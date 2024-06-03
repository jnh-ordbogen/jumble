import csv
from tqdm import tqdm

def read_rows(file_path):
    with open(file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [row for row in csv_reader]
    return rows

def get_unique_values(rows):
    values = [row['Ground truth'] for row in rows]
    unique_values = list(set(values))
    return unique_values

def get_related(rows, unique_value):
    related = {}
    for row in rows:
        if row['Ground truth'] == unique_value:
            related[row['Bad translation']] = row['BERT precision']
    return related

def get_best_bert_scores(bert_scores_dict, num_best=2):
    best_bert_scores = sorted(bert_scores_dict.items(), key=lambda x: x[1], reverse=True)[:num_best]
    return best_bert_scores

def get_best_sentence_pairs(file_path, num_best):
    rows = read_rows(file_path)
    unique_values = get_unique_values(rows)
    best_sentence_pairs = {}
    for unique_value in tqdm(unique_values[0:10]):
        related = get_related(rows, unique_value)
        best_sentence_pairs[unique_value] = get_best_bert_scores(related, num_best)
    return best_sentence_pairs

def write_best_sentence_pairs(best_sentence_pairs, file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Ground truth', 'Best Translation', 'BERT precision'])
        for key, value in best_sentence_pairs.items():
            for pair in value:
                writer.writerow([key, pair[0], pair[1]])

if __name__ == "__main__":
    file_path = 'all_europarl_at_once.csv'
    num_best = 2
    best_sentence_pairs = get_best_sentence_pairs(file_path, num_best)
    print(len(best_sentence_pairs))
    write_best_sentence_pairs(best_sentence_pairs, 'best_europarl.csv')

