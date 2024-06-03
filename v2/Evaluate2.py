import csv
import evaluate
evaluate.enable_progress_bar()
bertscore = evaluate.load("bertscore")

def get_ref_hyp(row):
    return row['Ground truth'], row['Bad translation']

def read_rows(file_path):
    with open(file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [row for row in csv_reader]
        refs, hyps = [ref, hyp for ref, hyp in map(get_ref_hyp, rows)]
    return refs, hyps

def get_bert_scores(references, hypotheses):
    return bertscore.compute(references=references, predictions=hypotheses, lang="en", device='cuda:3')

if __name__ == "__main__":
    file_path = 'bad_idioms_2.csv'
    refs, hyps = read_rows(file_path)

    with open('all_idiom_at_once.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Ground truth', 'Bad translation', 'BERT precision', 'BERT recall', 'BERT f1'])
        bert_scores = get_bert_scores(refs, hyps)
        precisions = bert_scores['precision']
        recalls = bert_scores['recall']
        f1s = bert_scores['f1']
        #print(bert_score)
        for ref, hyp, precision, recall, f1 in zip(refs, hyps, precisions, recalls, f1s):
            writer.writerow([ref, hyp, precision, recall, f1])