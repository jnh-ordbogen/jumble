from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TranslationPipeline
import random
from Tictoc import Tictoc

device = "cuda:2"

class ChunkTranslator:

    def __init__(self, delimiter=' ') -> None:
        self.delimiter = delimiter
        self.frequency_dict = {}
        self.chunk_dict = {} # Every chunk has a dictionary of possible translations and their scores, such that unnecessary translations are not repeated
        #self.num_use_cache = 0
        #self.num_use_inference = 0
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-da-en").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")

#        self.pipe = TranslationPipeline(model=self.model, tokenizer=self.tokenizer, task="translation")

    # Returns a list of five different translations of the input chunk list
    def translate_multi_return(self, chunk_list, entity_indexes, num_return) -> list:
        outputs = [self._do_translation(chunk_list, entity_indexes) for _ in range(num_return)]
        outputs = list(set(outputs))
        return outputs
    
    # Returns a single translation of the input chunk list
    def translate(self, chunk_list, entity_indexes) -> str:
        output = self._do_translation(chunk_list, entity_indexes)
        return output

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
    
    # Takes a list of chunks and returns a single translation
    def _do_translation(self, chunk_list, entity_indexes=[]):
        if len(chunk_list) == 1:
            return ''
        translated_chunk_dicts = []
        for i, chunk in enumerate(chunk_list):
            if i in entity_indexes:
                translated_chunk_dicts.append(self._certain_token_to_dict(chunk))
            else:
                res_dict = self._lookup_or_translate(chunk)
                translated_chunk_dicts.append(res_dict)
        selected_chunks = [self._select_key_from_probabilities(chunk_dict) for chunk_dict in translated_chunk_dicts]
        if len(selected_chunks) <= 2:
            output = self._translate_full_sentence(self.delimiter.join(chunk_list))
        else:
            output = selected_chunks[0].capitalize() + self.delimiter + self.delimiter.join(selected_chunks[1:])
        return output
    
    def _lookup_or_translate(self, chunk):
        if len(self.chunk_dict) >= 1000:
            #print("Pruning")
            #print(len(self.chunk_dict))
            self.frequency_dict, self.chunk_dict = self.prune_lower_half_from_both(self.frequency_dict, self.chunk_dict)
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
        normalized_probs = [2 ** (-prob) for prob in probabilities.values()]
        total_prob = sum(normalized_probs)
        normalized_probs = [prob / total_prob for prob in normalized_probs]
        try:
            selected_key = random.choices(list(probabilities.keys()), weights=normalized_probs, k=1)[0]
        except:
            selected_key = ''
        return selected_key

    # Takes a chunk and returns a dictionary of possible translations and their probability scores
    def _translate_chunk(self, chunk):
        outputs = self._inference(chunk)
        translations = self._build_output_dict(outputs, chunk)
        return translations

    def _translate_full_sentence(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(device)
        outputs = self.model.generate(**inputs,
                                    max_new_tokens=len(sentence.split(' ')) * 2,
                                    num_return_sequences=1,
                                    num_beams=10,
                                    early_stopping=True)
        # Decode the first (and only) generated sequence to text
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    # Builds a dictionary of translations and their scores
    def _build_output_dict(self, outputs, chunk):
        translations = {}
        for idx, output in enumerate(outputs.sequences):
            translation = self.tokenizer.decode(output, skip_special_tokens=True)
            translation = self._output_chunk_cleaner(translation)
            score = outputs.sequences_scores[idx].item()
            if translation not in translations and self._is_valid_translation(translation, chunk):
                translations[translation] = score
        return translations
    
    # Checks if the translation is valid
    def _is_valid_translation(self, translation, chunk):
        return translation != chunk and translation != ''

    # Generates a translation for a chunk
    #@Tictoc.tictoc
    def _inference(self, chunk):
        inputs = self.tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        outputs = self.model.generate(**inputs, 
                         max_new_tokens=len(chunk.split(' '))*2, 
                         num_return_sequences=10, 
                         num_beams=10, 
                         early_stopping=True, 
                         return_dict_in_generate=True, 
                         output_scores=True)
        return outputs
    
    # Removes unwanted characters from a chunk
    def _output_chunk_cleaner(self, chunk):
        remove_chars = str.maketrans('', '', '♪-•;,.!?()[]{}<>:«»\'\"\#%&/\\■`')
        chunk = chunk.translate(remove_chars).strip().lower()
        return chunk