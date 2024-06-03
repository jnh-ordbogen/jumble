import csv
import evaluate
bert_score = evaluate.load('bertscore')
from tqdm import tqdm

def read_rows(file_path):
    with open(file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [row for row in csv_reader]
    return rows

def get_unique_values(rows):
    values = [row['Ground truth'] for row in rows]
    unqiue_values = list(set(values))
    return unqiue_values

def get_related(rows, unique_value):
    related = []
    for row in rows:
        if row['Ground truth'] == unique_value:
            related.append(row['Bad translation'])
    return related

def get_bert_score(reference, hypothesis):
    return bert_score.compute(references=[reference], predictions=[hypothesis], lang="en")['precision'][0]

def get_bert_scores(reference, hypotheses):
    bert_scores = {}
    for hypothesis in hypotheses:
        bert_scores[hypothesis] = get_bert_score(reference, hypothesis)
    return bert_scores

def get_best_bert_scores(bert_scores_dict, num_best=1):
    best_bert_scores = sorted(bert_scores_dict.items(), key=lambda x: x[1], reverse=True)#[:num_best]
    return best_bert_scores

def get_best_sentence_pairs(file_path, num_best):
    rows = read_rows(file_path)
    unique_values = get_unique_values(rows)
    best_sentence_pairs = {}
    for unique_value in tqdm(unique_values[0:10]):
        related = get_related(rows, unique_value)
        bert_scores = get_bert_scores(unique_value, related)
        best_bert_scores = get_best_bert_scores(bert_scores, num_best)
        best_sentence_pairs[unique_value] = best_bert_scores
    return best_sentence_pairs

def write_best_sentence_pairs(best_sentence_pairs, file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Ground truth', 'Best Translation', 'bert Score'])
        for key, value in best_sentence_pairs.items():
            for pair in value:
                writer.writerow([key, pair[0], pair[1]])

if __name__ == "__main__":
    file_path = 'bad_idioms_2.csv'
    num_best = 2
    best_sentence_pairs = get_best_sentence_pairs(file_path, num_best)
    print(len(best_sentence_pairs))
    write_best_sentence_pairs(best_sentence_pairs, 'best_idioms_test_precision.csv')

