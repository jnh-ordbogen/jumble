import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import score
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

class BertScoreAnalyzer:
    def __init__(self):
        pass

    def get_bert_score(self, generated_sentence, ground_truth):
        P, R, F1 = score([generated_sentence], [ground_truth], lang="en", verbose=False)
        bert_score = F1.item()
        return bert_score
    
    def get_bert_scores_batch(self, generated_sentences, ground_truths):
        P, R, F1 = score(generated_sentences, ground_truths, lang="en", verbose=False)
        return F1.numpy()

    #def calculate_bert_scores(self, generated_sentences):
    #    bert_scores = []
    #    for generated in generated_sentences:
    #        bert_scores.append(self.calculate_bert_score(generated))

    #def get_max_bert_score(self):
    #    index = self.bert_scores.index(max(self.bert_scores))
    #    best_sentence = self.generated_sentences[index]
    #    best_score = self.bert_scores[index]
    #    return best_score, best_sentence

    #def get_min_bert_score(self):
    #    index = self.bert_scores.index(min(self.bert_scores))
    #    worst_sentence = self.generated_sentences[index]
    #    worst_score = self.bert_scores[index]
    #    return worst_score, worst_sentence

    def plot_bertscore_distribution(self, bert_scores, filename):
        plt.figure(figsize=(8, 6))
        sns.kdeplot(bert_scores, fill=True)
        plt.title('Distribution of BERTScores')
        plt.xlabel('BERTScore')
        plt.ylabel('Density')
        plt.savefig(filename)