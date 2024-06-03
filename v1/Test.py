from SentenceSplitter import SentenceSplitter
from ChunkTranslator import ChunkTranslator
from BertScoreAnalyzer import BertScoreAnalyzer
import csv
import time
from tqdm import tqdm

with open('datasets/Europarl.da-en.da', mode='r', newline='') as txt_file:
    danish_sentences = txt_file.readlines()
    danish_sentences = [sentence.strip() for sentence in danish_sentences]

with open('datasets/Europarl.da-en.en', mode='r', newline='') as txt_file:
    correct_sentences = txt_file.readlines()
    correct_sentences = [sentence.strip() for sentence in correct_sentences]

#danish_sentences = danish_sentences[-1000:]
#correct_sentences = correct_sentences[-1000:]

splitter = SentenceSplitter()
translator = ChunkTranslator(delimiter=' ')
analyzer = BertScoreAnalyzer()
max_num_splits = 2
max_num_translations = 5
output_plot_filename = 'bertscore_distribution.png'
output_lines_filename = 'bad_translations0.csv'

def get_bad_translations(src):
    possible_splits, index_lists = splitter.get_multiple_splits(src, max_num_splits)
    split_err = {}
    for split, indexes in zip(possible_splits, index_lists):
        split_str = ' | '.join(split)
        error_sentences = translator.translate_multi_return(split, indexes, max_num_translations)
        split_err[split_str] = error_sentences
    return split_err

def get_bad_translation(src):
    chunks = splitter.split(src)
    error_sentence = translator.translate(chunks)
    return error_sentence

def write_dict(da_sentences, cor_sentences):
    bad_translations_dict = {}
    pairs = list(zip(da_sentences, cor_sentences))
    for src, tgt in tqdm(pairs):
        bad_translations = get_bad_translations(src)
        for bad in bad_translations.values():
            for b in bad:
                bad_translations_dict[b] = tgt
    return bad_translations_dict

def get_results(input_dict):
    bad_translations = list(input_dict.keys())
    ground_truths = list(input_dict.values())

    bert_scores = analyzer.get_bert_scores_batch(bad_translations, ground_truths)

    results = []
    for bad_translation, bert_score in zip(bad_translations, bert_scores):
        results.append((bad_translation, bert_score))

    return results


def write_csv(results, bad_translations_dict):
    with open(output_lines_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Ground truth', 'Bad Translation', 'BERT Score']
        writer = csv.writer(csv_file)
        
        writer.writerow(fieldnames)
        
        for result in results:
            if result[1] >= 0.90:
                ground_truth = bad_translations_dict[result[0]]
                writer.writerow([ground_truth, result[0], result[1]])



bad_translations_dict = write_dict(danish_sentences, correct_sentences)
results = get_results(bad_translations_dict)
write_csv(results, bad_translations_dict)



#start = time.time()
#end = time.time()
#all_bad = bad_translations_dict.keys()
#bad_per_sec = len(all_bad) / (end - start)


def get_results_old(dict):
    results = []

    for bad_translation, ground_truth in dict.items():
        bert_score = analyzer.get_bert_score(bad_translation, ground_truth)
        results.append((bad_translation, bert_score))
        
    return results



def get_bert_column_values(csv_file):
    bert_column_values = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                bert_column_values.append(row[1])

    bert_column_values = [float(i) for i in bert_column_values[1:]]

    return bert_column_values

all_bert_scores = [bert_score for _, bert_score in results]

num_over_90 = 0
for score in all_bert_scores:
    if score >= 0.9:
        num_over_90 += 1
percentage_over_90 = num_over_90 / len(all_bert_scores)
print(f'Percentage over 90: {percentage_over_90}')
analyzer.plot_bertscore_distribution(all_bert_scores, output_plot_filename)


danish_sentences = [
    "Genoptagelse af sessionen",
    "Jeg erklærer Europa-Parlamentets session, der blev afbrudt fredag den 17. december, for genoptaget. Endnu en gang vil jeg ønske Dem godt nytår, og jeg håber, De har haft en god ferie.",
    "Som De kan se, indfandt det store \"år 2000-problem\" sig ikke. Til gengæld har borgerne i en del af medlemslandene været ramt af meget forfærdelige naturkatastrofer.",
    "De har udtrykt ønske om en debat om dette emne i løbet af mødeperioden.",
    "I mellemtiden ønsker jeg - som også en del kolleger har anmodet om - at vi iagttager et minuts stilhed til minde om ofrene for bl.a. stormene i de medlemslande, der blev ramt.",
    "Jeg opfordrer Dem til stående at iagttage et minuts stilhed.",
    "Parlamentet iagttog stående et minuts stilhed",
    "Fru formand, en bemærkning til forretningsordenen.",
    "Gennem pressen og tv vil De være bekendt med en række bombeeksplosioner og drab i Sri Lanka.",
    "En af de personer, der blev myrdet for ganske nylig i Sri Lanka, var hr. Kumar Ponnambalam, der besøgte Europa-Parlamentet for få måneder siden."
]
correct_sentences = [
    "Resumption of the session",
    "I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.",
    "Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful.",
    "You have requested a debate on this subject in the course of the next few days, during this part-session.",
    "In the meantime, I should like to observe a minute's silence, as a number of Members have requested, on behalf of all the victims concerned, particularly those of the terrible storms, in the various countries of the European Union.",
    "Please rise, then, for this minute's silence.",
    "The House rose and observed a minute's silence",
    "Madam President, on a point of order.",
    "You will be aware from the press and television that there have been a number of bomb explosions and killings in Sri Lanka.",
    "One of the people assassinated very recently in Sri Lanka was Mr Kumar Ponnambalam, who had visited the European Parliament just a few months ago."
]