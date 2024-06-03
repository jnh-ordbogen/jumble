from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TranslationPipeline
import random
from Tictoc import Tictoc
import re
import torch

class ChunkTranslator:

    def __init__(self, device, delimiter=' ') -> None:
        self.device = f'cuda:{device}'
        self.delimiter = delimiter
        self.frequency_dict = {}
        self.chunk_dict = {} # Every chunk has a dictionary of possible translations and their scores, such that unnecessary translations are not repeated
        self.num_use_cache = 0
        self.num_use_inference = 0
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-da-en").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")

    # Returns a list of five different translations of the input chunk list
    def translate_multi_return(self, chunk_list, entity_indexes, num_return) -> list:
        outputs = [self._do_translation(chunk_list, entity_indexes) for _ in range(num_return)]
        outputs = list(set(outputs))
        return outputs

    def get_top_cached_translations(self, num_translations):
        sorted_keys = sorted(self.frequency_dict, key=self.frequency_dict.get, reverse=True)
        top_keys = sorted_keys[:num_translations]
        top_values = [self.frequency_dict[key] for key in top_keys]
        top_translations = [(key, value) for key, value in zip(top_keys, top_values)]
        return top_translations

    def get_num_cached_translations(self):
        return len(self.chunk_dict)
    
    def get_num_use_cache(self):
        return self.num_use_cache
    
    def get_num_use_inference(self):
        return self.num_use_inference

    #@Tictoc.tictoc
    def translate(self, chunk_list, entity_indexes=[], num_return=5) -> str:
        if len(chunk_list) == 1:
            return ''
        
        chunks_indeces_to_ignore = []
        chunks_indeces_to_inference = []
        for i, chunk in enumerate(chunk_list):
            if i in entity_indexes:
                chunks_indeces_to_ignore.append(i)
            elif chunk in self.chunk_dict:
                chunks_indeces_to_ignore.append(i)
            else:
                chunks_indeces_to_inference.append(i)

        self.num_use_cache += len(chunks_indeces_to_ignore) - len(entity_indexes)
        self.num_use_inference += len(chunks_indeces_to_inference)

        to_translate = [chunk_list[i] for i in chunks_indeces_to_inference]
        if len(to_translate) != 0:
            translated_dicts = self._translate_chunks(to_translate)

            for index in chunks_indeces_to_inference:
                self.chunk_dict[chunk_list[index]] = translated_dicts.pop(0)
                self.frequency_dict[chunk_list[index]] = 0

        translated_chunk_dicts = []
        for i, chunk in enumerate(chunk_list):
            if i in entity_indexes:
                translated_chunk_dicts.append(self._certain_token_to_dict(chunk))
            else:
                translated_chunk_dicts.append(self.chunk_dict[chunk])

        #for i, chunk in enumerate(chunk_list):
        #    if i in entity_indexes:
        #        translated_chunk_dicts.append(self._certain_token_to_dict(chunk))
        #    else:
        #        res_dict = self._lookup_or_translate(chunk)
        #        translated_chunk_dicts.append(res_dict)
        
        
        return list(set([self._random_sample(chunk_list, translated_chunk_dicts, entity_indexes) for _ in range(num_return)]))

    
    #@Tictoc.tictoc
    def _random_sample(self, chunk_list, translated_chunk_dicts, entity_indexes=[]):
        selected_chunks = [self._select_key_from_probabilities(chunk_dict) for chunk_dict in translated_chunk_dicts]
        new_selected_chunks = []
        for i, (chunk, selected) in enumerate(zip(chunk_list, selected_chunks)):
            if not i in entity_indexes:
                #selected = self._correct_word(word=selected)
                if chunk[0].isupper():
                    selected = selected.capitalize()
            new_selected_chunks.append(selected)
        selected_chunks = new_selected_chunks
        if len(selected_chunks) <= 2:
            output = self._translate_full_sentence(self.delimiter.join(chunk_list))
        else:
            output = selected_chunks[0].capitalize() + self.delimiter + self.delimiter.join(selected_chunks[1:])

        output = re.sub(r'\s+([^\w\s])', r'\1', output)

        return output
    
    def _lookup_or_translate(self, chunk):
        #if len(self.chunk_dict) >= 10000:
            #print("Pruning")
            #print(len(self.chunk_dict))
            #self.frequency_dict, self.chunk_dict = self.prune_lower_half_from_both(self.frequency_dict, self.chunk_dict)
            #print("Pruned")
            #print(len(self.chunk_dict))
        if not chunk in self.chunk_dict:
            self.chunk_dict[chunk] = self._translate_chunk(chunk)
            self.frequency_dict[chunk] = 0
            #self.num_use_inference += 1
        #else:
            #print("Cache hit")
        #    self.num_use_cache += 1
        self.frequency_dict[chunk] += 1
        return self.chunk_dict[chunk]
    
    #@Tictoc.tictoc
    def prune_lower_half_from_both(self, d1, d2):
        # Sort the first dictionary by value and get the keys
        sorted_keys = sorted(d1, key=d1.get)
        
        # Calculate the midpoint to find the threshold for the lower 50%
        midpoint = len(sorted_keys) // 2
        
        # Keys to keep (top 50% based on the first dictionary)
        keys_to_keep = set(sorted_keys[midpoint:])
        
        # Filter both dictionaries to keep only the entries with keys in 'keys_to_keep'
        pruned_d1 = {k: v for k, v in d1.items() if k in keys_to_keep}
        pruned_d2 = {k: v for k, v in d2.items() if k in keys_to_keep}
        
        return pruned_d1, pruned_d2

    def _certain_token_to_dict(self, token):
        return {token: 1}

    # Stochastically selects a key from a dictionary of translations and their scores
    def _select_key_from_probabilities(self, probabilities):
        #print(probabilities)
        normalized_probs = [2 ** (prob) for prob in probabilities.values()]
        total_prob = sum(normalized_probs)
        normalized_probs = [prob / total_prob for prob in normalized_probs]
        try:
            selected_key = random.choices(list(probabilities.keys()), weights=normalized_probs, k=1)[0]
        except:
            selected_key = ''
        return selected_key

    # Takes a chunk and returns a dictionary of possible translations and their probability scores
    #@Tictoc.tictoc
    def _translate_chunks(self, chunk):

        outputs_list = self._inference(chunk)
        
        translations = []
        for outputs in outputs_list:
            #for (seq, score) in zip(sequences, scores):
                #print('Sequence:', self.tokenizer.decode(seq, skip_special_tokens=True), score.item())
            translations.append(self._build_output_dict(outputs, chunk))
        #print(translations)
        return translations

    #@Tictoc.tictoc
    def _translate_full_sentence(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs,
                                    max_new_tokens=len(sentence.split(' ')) * 2,
                                    num_return_sequences=1,
                                    num_beams=10,
                                    early_stopping=True)
        # Decode the first (and only) generated sequence to text
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    # Builds a dictionary of translations and their scores
    #@Tictoc.tictoc
    def _build_output_dict(self, outputs, chunk):

        translations = {}
        #print('build_output_dict', outputs)
        for idx, output in enumerate(outputs):
            translation = self._output_chunk_cleaner(output)
            score = outputs[output]
            #score = outputs.sequences_scores[idx].item()
            if translation not in translations and self._is_good(chunk, translation) and translation != '': #and self._is_valid_translation(translation, chunk):
                translations[translation] = score
        #print(translations)
        return translations
    
    def _is_good(self, chunk, translation):
        if len(chunk) == 1:
            return True
        if len(translation) == 1:
            return False
        return True

    # Checks if the translation is valid
    def _is_valid_translation(self, translation, chunk):
        acceptable_lengths = [len(chunk.split(' ')), len(chunk.split(' ')) + 1]
        return translation != chunk and translation != '' #and len(translation.split(' ')) in acceptable_lengths

    # Generates a translation for a chunk
    #@Tictoc.tictoc
    #def _inference(self, chunk):
    #    inputs = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
    #    outputs = self.model.generate(**inputs, 
    #                     max_new_tokens=len(chunk.split(' '))*2, 
    #                     num_return_sequences=10, 
    #                     num_beams=10, 
    #                     early_stopping=True, 
    #                     return_dict_in_generate=True, 
    #                     output_scores=True)
    #    return outputs


    def _inference(self, chunks):
        # Tokenize all chunks together
        inputs = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generate outputs
        outputs = self.model.generate(**inputs, 
                                    max_new_tokens=max(len(chunk.split(' '))*2 for chunk in chunks), 
                                    num_return_sequences=10, 
                                    num_beams=10, 
                                    early_stopping=True, 
                                    return_dict_in_generate=True, 
                                    output_scores=True)
        
        # Get the number of sequences per input
        num_return_sequences = outputs.sequences.shape[0] // len(chunks)

        # Iterate over generated sequences and scores
        generated_sequences = outputs.sequences.tolist()
        sequence_scores = outputs.sequences_scores.tolist()
        outputs = []

        for i in range(len(chunks)):
            output = {}
            for j in range(num_return_sequences):
                sequence = self.tokenizer.decode(generated_sequences[i * num_return_sequences + j], skip_special_tokens=True)
                score = sequence_scores[i * num_return_sequences + j]
                output[sequence] = score
            outputs.append(output)
        return outputs


    """ def _inference(self, chunks):
        # Tokenize all chunks together
        inputs = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Generate outputs
        outputs = self.model.generate(**inputs, 
                                    #max_new_tokens=10, 
                                    num_return_sequences=10, 
                                    num_beams=10, 
                                    early_stopping=True, 
                                    return_dict_in_generate=True, 
                                    output_scores=True)
        # Split sequences and sequence scores
        batch_size = len(chunks)
        print('shape',outputs.sequences.shape[0])
        chunk_size = outputs.sequences.shape[0] // batch_size
        sequence_chunks = torch.split(outputs.sequences, chunk_size)
        score_chunks = torch.split(outputs.sequences_scores, chunk_size)
        for sequence_chunk in sequence_chunks:
            print('sequence_chunk', self.tokenizer.decode(sequence_chunk, skip_special_tokens=True))
        ## Print sequences and sequence scores
        #for sequences, scores in zip(sequence_chunks, score_chunks):
        #    print('Sequence Scores:', scores)
        #    for sequence in sequences:
        #        print(self.tokenizer.decode(sequence, skip_special_tokens=True))
        
        return zip(sequence_chunks, score_chunks) """


    
    # Removes unwanted characters from a chunk
    def _output_chunk_cleaner(self, chunk):
        remove_chars = str.maketrans('', '', '♪-•;,.!?()[]{}<>:«»\'\"\#%&/\\■`')
        chunk = chunk.translate(remove_chars).strip().lower()
        return chunk
    


        # Takes a list of chunks and returns a single translation
"""     def translate(self, chunk_list, entity_indexes=[]) -> str:
        if len(chunk_list) == 1:
            return ''
        translated_chunk_dicts = []
        #print('chunk_list', chunk_list)

        self._translate_chunk(chunk_list)
        
        #for i, chunk in enumerate(chunk_list):
        #    if i in entity_indexes:
        #        translated_chunk_dicts.append(self._certain_token_to_dict(chunk))
        #    else:
        #        res_dict = self._lookup_or_translate(chunk)
        #        translated_chunk_dicts.append(res_dict)
        selected_chunks = [self._select_key_from_probabilities(chunk_dict) for chunk_dict in translated_chunk_dicts]
        new_selected_chunks = []
        for i, (chunk, selected) in enumerate(zip(chunk_list, selected_chunks)):
            if chunk[0].isupper() and not i in entity_indexes:
                selected = selected.capitalize()
            new_selected_chunks.append(selected)
        selected_chunks = new_selected_chunks
        if len(selected_chunks) <= 2:
            output = self._translate_full_sentence(self.delimiter.join(chunk_list))
        else:
            output = selected_chunks[0].capitalize() + self.delimiter + self.delimiter.join(selected_chunks[1:])

        output = re.sub(r'\s+([^\w\s])', r'\1', output)

        return output
 """