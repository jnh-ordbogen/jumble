import random
from Tokenizer import Tokenizer
import re
from DateTranslator import DateTranslator

class SentenceSplitter:

    def __init__(self) -> None:
        self.tokenizer = Tokenizer()
        self.date_translator = DateTranslator()

    def split(self, sentence, max_length=4):
        named_entity_indexes, words = self.tokenizer.tokenize_sentence(sentence)
        sublists, indexes = self._split_list_at_indexes(words, named_entity_indexes)
        new_sublists = []
        new_indexes = []
        for i in range(len(sublists)):
            if i in indexes:
                if self._likely_to_be_a_date(sublists[i][0]):
                    sublists[i][0] = self.date_translator.translate(sublists[i][0])
                new_sublists.append(sublists[i][0])
                new_indexes.append(len(new_sublists)-1)
            else:
                sub_sublists = self._split_aux(sublists[i], max_length)
                for sub_sublist in sub_sublists:
                    new_sublists.append(sub_sublist)
        return new_sublists, new_indexes
    
    def _split_aux(self, word_list, max_length=5):
        
        segments = []
        i = 0

        while i < len(word_list):
            max_remaining_length = len(word_list) - i
            if max_remaining_length <= max_length:
                segment_size = max_remaining_length
            else:
                segment_size = random.choices([1, 2, 3], weights=[1, 2, 4], k=1)[0]
                segment_size = min(segment_size, max_length)

            segment = word_list[i:i+segment_size]
            segments.append(' '.join(segment))
            i += segment_size

        return segments
    
    def get_multiple_splits(self, sentence, num_splits):
        splits = []
        index_lists = []
        while len(splits) < num_splits:
            split, indexes = self.split(sentence)
            splits.append(split)
            index_lists.append(indexes)
        return splits, index_lists

    def _split_list_at_indexes(self, input_list, indexes):
        split_lists = []
        start = 0
        new_indexes = []

        for index in indexes:
            split_lists.append(input_list[start:index])
            split_lists.append([input_list[index]])
            new_indexes.append(len(split_lists)-1)
            start = index + 1

        split_lists.append(input_list[start:])  # Append the remaining elements after the last index

        return split_lists, new_indexes

    def _likely_to_be_a_date(self, word):
        pattern = r'\d'
        digit_matches = re.findall(pattern, word)
        return len(digit_matches) >= 2




#    def split_old(self, sentence):
#
#        words = [word for word in word_tokenize(sentence) if word not in string.punctuation]
#
#        segments = []
#        i = 0
#
#        while i < len(words):
#            segment_size = random.randint(1, 3)  # Random segment size of 1 to 3
#            segment = words[i:i+segment_size]    # Extract segment from the sentence
#            segments.append(' '.join(segment))   # Join the words to form the segment
#            i += segment_size                    # Move to the next segment
#
#        return segments
#