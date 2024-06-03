import csv
from tqdm import tqdm

def read_rows(file_path):
    rows = []
    with open(file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            rows.append(row)
    return rows

def get_best_bert_scores(bert_scores_dict, num_best=2):
    return sorted(bert_scores_dict.items(), key=lambda x: x[1], reverse=True)[:num_best]

def write_best_sentence_pairs(best_sentence_pairs, file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Ground truth', 'Best Translation', 'BERT precision'])
        for key, value in best_sentence_pairs.items():
            for pair in value:
                writer.writerow([key, pair[0], pair[1]])

def get_unique_values(rows):
    unique_dict = {}
    total = len(rows)
    for i, row in tqdm(enumerate(rows), total=total):
        if row['Ground truth'] not in unique_dict:
            unique_dict[row['Ground truth']] = [i]
        else:
            unique_dict[row['Ground truth']].append(i)

    return unique_dict

def get_best_sentence_pairs(file_path, num_best):
    best_sentence_pairs = {}

    rows = read_rows(file_path)

    # First pass: collect unique values
    
    unique_values = get_unique_values(rows)
    print('unique_values_length: ', len(unique_values))

    # Second pass: process related values
    for unique_value in tqdm(unique_values):
        related = {}
        inner_rows = []
        for idx in unique_values[unique_value]:
            inner_rows.append(rows[idx])
        for row in inner_rows:
            related[row['Bad translation']] = row['BERT precision']
        best_sentence_pairs[unique_value] = get_best_bert_scores(related, num_best)

    return best_sentence_pairs

if __name__ == "__main__":
    file_path = 'bad_idioms_with_score.csv'
    num_best = 8
    best_sentence_pairs = get_best_sentence_pairs(file_path, num_best)
    print(len(best_sentence_pairs))
    write_best_sentence_pairs(best_sentence_pairs, 'best_bad_idioms_8.csv')
