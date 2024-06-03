import spacy
import numpy as np

class Comparer:

    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def pos_tag(self, sentence):
        doc = self.nlp(sentence)
        return [(token.text, token.pos_) for token in doc]

    def levenshtein_distance(self, s1, s2):
        n, m = len(s1), len(s2)
        dp = np.zeros((n + 1, m + 1), dtype=int)

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        return dp[n][m]

    def align_tags(self, src_tags, tgt_tags):
        alignments = []

        for src_word, src_tag in src_tags:
            best_alignment = None
            min_distance = float('inf')

            for tgt_word, tgt_tag in tgt_tags:
                distance = self.levenshtein_distance(src_tag, tgt_tag)
                if distance < min_distance:
                    min_distance = distance
                    best_alignment = (src_word, src_tag, tgt_word, tgt_tag)

            alignments.append(best_alignment)

        return alignments

    def compare_errors(self, src_sentence, tgt_sentence):
        src_tags = self.pos_tag(src_sentence)
        tgt_tags = self.pos_tag(tgt_sentence)

        # Align the POS-tagged sentences
        alignments = self.align_tags(src_tags, tgt_tags)

        # Calculate the numeric score based on errors
        score = 0
        for a in alignments:
            print(a)
        for src_word, src_tag, tgt_word, tgt_tag in alignments:
            if src_word != tgt_word:
                # Assign a higher score for more severe errors (e.g., word replacement)
                # You can customize the scoring based on your specific criteria
                score += 2  # Example: 2 points for each error

        return score
