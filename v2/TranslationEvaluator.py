import csv
from tqdm import tqdm
from SentenceSplitter import SentenceSplitter
from ChunkTranslator import ChunkTranslator
import evaluate
from Tictoc import Tictoc

class TranslationEvaluator:
    def __init__(self, device, output_path):
        self.device = device

        self.splitter = SentenceSplitter(device)
        self.translator = ChunkTranslator(device, delimiter=' ')

        self.bert = evaluate.load('bertscore')
        self.rouge = evaluate.load('rouge')
        
        self.max_num_splits = 1
        self.max_num_translations = 10
        self.output_lines_filename = output_path

    def get_num_use_cache(self):
        return self.translator.get_num_use_cache()
    
    def get_num_use_translator(self):
        return self.translator.get_num_use_inference()

    def get_bad_translations(self, src):
        possible_splits, index_lists = self.splitter.get_multiple_splits(src, self.max_num_splits)
        split_err = {}
        # print(possible_splits)
        for split, indexes in zip(possible_splits, index_lists):
            
            split_str = ' | '.join(split)
            
            # print(split_str)
            # error_sentence = translator.translate(split, indexes)
            #print(split_str)
            error_sentences = self.translator.translate(split, indexes, 10)
            split_err[split_str] = error_sentences
        return split_err
    
    def get_bad_translations_batched(self, src):
        possible_splits, index_lists = self.splitter.get_multiple_splits(src, self.max_num_splits)
        split_err = {}
        for split, indexes in zip(possible_splits, index_lists):
            
            split_str = ' | '.join(split)
            
            # print(split_str)
            # error_sentence = translator.translate(split, indexes)
            #print(split_str)
            error_sentences = self.translator.translate(split, indexes, 10)
            split_err[split_str] = error_sentences
        return split_err

    def write_dict(self, danish_sentences, correct_sentences):
        bad_translations_dict = {}
        pairs = list(zip(danish_sentences, correct_sentences))
        for src, tgt in pairs:
            bad_translations = self.get_bad_translations(src)
            for bad in bad_translations.values():
                for b in bad:
                    bad_translations_dict[b] = tgt
        return bad_translations_dict
    
    def write_batches(self, danish_sentences, correct_sentences, batch_size=100):
        batches = []
        pairs = list(zip(danish_sentences, correct_sentences))
        num_pairs = len(pairs)

        for batch_idx in range(0, num_pairs, batch_size):
            batch_pairs = pairs[batch_idx:batch_idx + batch_size]
            batches.append(batch_pairs)

        inputs = []
        for batch in batches:
            input_list = []
            for pair in batch:
                input_list.append(pair)
                input_list.append(('[BATCH_TOKEN]','[BATCH_TOKEN]'))
            inputs.append(input_list)

        for batch in inputs:
            bad_translations_dict = self.get_bad_translations_batched(batch)

        return result

        for src, tgt in pairs:
            bad_translations = self.get_bad_translations(src)
            for bad in bad_translations.values():
                for b in bad:
                    bad_translations_dict[b] = tgt
        return bad_translations_dict
    
    def get_results(self, input_dict):
        #print(input_dict)
        bad_translations = list(input_dict.keys())
        ground_truths = list(input_dict.values())
        bert_scores, rouge_scores = self.get_individual_scores(bad_translations, ground_truths)
        edit_distances = self._edit_distances(bad_translations, ground_truths)
        results = []
        for bad_translation, bert_score, edit_distance in zip(bad_translations, bert_scores, edit_distances):
            results.append((bad_translation, bert_score, edit_distance))
        return results

    def write_csv(self, results, bad_translations_dict, is_idiom=False):
        with open(self.output_lines_filename, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            #writer.writerow(['Ground truth', 'Bad Translation', 'BERT Score', 'rouge Score'])
            for result in results:
                """ if (is_idiom and result[1] >= 0.87) or (not is_idiom and result[1] >= 0.89): """
                ground_truth = bad_translations_dict[result[0]]
                writer.writerow([ground_truth, result[0], result[1], result[2]])

    def write_csv_stripped(self, bad_translations_dict):
        with open(self.output_lines_filename, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            #writer.writerow(['Ground truth', 'Bad Translation', 'BERT Score', 'rouge Score'])
            for (key, value) in zip(bad_translations_dict.keys(), bad_translations_dict.values()):
                writer.writerow([key, value])

    #@Tictoc.tictoc
    def process(self, danish_sentences, correct_sentences):
        bad_translations_dict = self.write_dict(danish_sentences, correct_sentences)
        self.write_csv_stripped(bad_translations_dict)
        #results = self.get_results(bad_translations_dict)
        #self.write_csv(results, bad_translations_dict, is_idiom)

    def get_individual_scores(self, predictions, references):
        bert_scores = []
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            bert_scores.append(self.bert.compute(predictions=[pred], references=[ref], lang='en')['f1'][0])
            rouge_scores.append(0)
            #rouge_scores.append(self.rouge.compute(predictions=[pred], references=[ref])['rouge1'])
        return bert_scores, rouge_scores
    
    def _levenshtein_distance(self, s1, s2):
        """
        Calculate the Levenshtein distance between two strings.
        """
        m, n = len(s1), len(s2)

        # Create a matrix to store the distances
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # Initialize the first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        # Fill in the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                #if m > 2 and n > 2:
                #    cost = 0 if (self._levenshtein_distance(s1[i - 1], s2[j - 1]) <= 1) else 1
                #else:
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,      # deletion
                            dp[i][j - 1] + 1,      # insertion
                            dp[i - 1][j - 1] + cost)  # substitution
        return dp[m][n]

    def _word_level_edit_distance(self, sentence1, sentence2):
        """
        Calculate the word-level edit distance between two sentences.
        """
        # Tokenize the sentences into words
        words1 = sentence1.lower().split()
        words2 = sentence2.lower().split()
        # Calculate the Levenshtein distance between the tokenized sequences
        distance = self._levenshtein_distance(words1, words2)

        norm_distance = distance / max(len(words1), len(words2))

        return norm_distance

    def _edit_distances(self, sentences1, sentences2):
        return [self._word_level_edit_distance(s1, s2) for s1, s2 in zip(sentences1, sentences2)]