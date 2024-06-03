import csv
from tqdm import tqdm
import evaluate
evaluate.enable_progress_bar()
bertscore = evaluate.load("bertscore")

def get_ref_hyp(row):
    return row['Ground truth'], row['Bad Translation']

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

def get_bert_scores(references, hypotheses):
    return bertscore.compute(references=references, predictions=hypotheses, lang="en", device='cuda:3')

def write_results_to_csv(refs, hyps, precisions, recalls, f1s, csv_file):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for ref, hyp, precision, recall, f1 in zip(refs, hyps, precisions, recalls, f1s):
            writer.writerow([ref, hyp, precision, recall, f1])

if __name__ == "__main__":
    file_path = 'bad_europarl_full.csv'
    refs, hyps = read_rows(file_path)
    output_file = 'all_europarl_at_once.csv'

    # print('here')

    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Ground truth', 'Bad translation', 'BERT precision', 'BERT recall', 'BERT f1'])
    

    batch_size = 1000
    for i in tqdm(range(0, len(refs), batch_size)):
        batch_refs = refs[i:i+batch_size]
        batch_hyps = hyps[i:i+batch_size]
        # print(len(batch_hyps))
        # print(len(batch_refs))
        # Get BERT scores in batches
        bert_scores = get_bert_scores(batch_refs, batch_hyps)
        precisions = bert_scores['precision']
        recalls = bert_scores['recall']
        f1s = bert_scores['f1']
        
        # Append results to CSV
        write_results_to_csv(batch_refs, batch_hyps, precisions, recalls, f1s, output_file)
