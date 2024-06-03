import csv
from tqdm import tqdm
from SentenceSplitter import SentenceSplitter
from ChunkTranslator import ChunkTranslator
from BertScoreAnalyzer import BertScoreAnalyzer

class TranslationEvaluator:
    def __init__(self, danish_sentences, correct_sentences):
        self.danish_sentences = danish_sentences
        self.correct_sentences = correct_sentences
        # Assuming these classes are defined elsewhere in your project
        self.splitter = SentenceSplitter()
        self.translator = ChunkTranslator(delimiter=' ')
        self.analyzer = BertScoreAnalyzer()
        self.max_num_splits = 2
        self.max_num_translations = 5
        self.output_lines_filename = 'bad_idioms.csv'

    def get_bad_translations(self, src):
        possible_splits, index_lists = self.splitter.get_multiple_splits(src, self.max_num_splits)
        split_err = {}
        for split, indexes in zip(possible_splits, index_lists):
            split_str = ' | '.join(split)
            error_sentences = self.translator.translate_multi_return(split, indexes, self.max_num_translations)
            split_err[split_str] = error_sentences
        return split_err

    def write_dict(self):
        bad_translations_dict = {}
        pairs = list(zip(self.danish_sentences, self.correct_sentences))
        for src, tgt in pairs:
            bad_translations = self.get_bad_translations(src)
            for bad in bad_translations.values():
                for b in bad:
                    bad_translations_dict[b] = tgt
        return bad_translations_dict

    def get_results(self, input_dict):
        bad_translations = list(input_dict.keys())
        ground_truths = list(input_dict.values())
        bert_scores = self.analyzer.get_bert_scores_batch(bad_translations, ground_truths)
        results = []
        for bad_translation, bert_score in zip(bad_translations, bert_scores):
            results.append((bad_translation, bert_score))
        return results

    def write_csv(self, results, bad_translations_dict, is_idiom=False):
        with open(self.output_lines_filename, mode='a', newline='') as csv_file:
            #fieldnames = ['Ground truth', 'Bad Translation', 'BERT Score']
            writer = csv.writer(csv_file)
            #writer.writerow(fieldnames)
            for result in results:
                if (is_idiom and result[1] >= 0.87) or (not is_idiom and result[1] >= 0.89):
                    ground_truth = bad_translations_dict[result[0]]
                    writer.writerow([ground_truth, result[0], result[1]])

    def process(self, is_idiom=False):
        bad_translations_dict = self.write_dict()
        results = self.get_results(bad_translations_dict)
        self.write_csv(results, bad_translations_dict, is_idiom)