import csv
from evaluate import load
bertscore = load("bertscore")
from tqdm import tqdm

def get_ref_hyp(row):
    return row['Ground truth'], row['Bad translation']

def read_rows(file_path):
    with open(file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [row for row in csv_reader]
        refs = []
        hyps = []
        #rows = rows[0:1000]
        for row in rows:
            ref, hyp = get_ref_hyp(row)
            refs.append(ref)
            hyps.append(hyp)
    return refs, hyps

def get_bert_score(reference, hypothesis):
    return bertscore.compute(references=[reference], predictions=[hypothesis], lang="en", device='cuda:3')

if __name__ == "__main__":
    file_path = 'bad_idioms_2.csv'
    refs, hyps = read_rows(file_path)

    with open('all_idiom.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Ground truth', 'Bad translation', 'BERT precision', 'BERT recall', 'BERT f1'])
        lst = zip(refs, hyps)
        for ref, hyp in tqdm(lst, total=len(refs)):
            bert_score = get_bert_score(ref, hyp)
            precision = bert_score['precision'][0]
            recall = bert_score['recall'][0]
            f1 = bert_score['f1'][0]
            #print(bert_score)
            writer.writerow([ref, hyp, precision, recall, f1])