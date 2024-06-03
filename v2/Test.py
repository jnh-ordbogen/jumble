from SentenceSplitter import SentenceSplitter
from ChunkTranslator import ChunkTranslator

danish_sentences = [
    "Jeg erklærer Europa-Parlamentets session, der blev afbrudt fredag den 17. december, for genoptaget. Endnu en gang vil jeg ønske Dem godt nytår, og jeg håber, De har haft en god ferie.",
    "Alt dette er i tråd med de principper, vi altid har været tilhængere af.",
    "Tak, hr. Segni, det gør jeg med glæde.",
    "Det er således helt i tråd med den holdning, Europa-Parlamentet altid har indtaget.",
    "Fru formand, jeg vil gerne gøre Dem opmærksom på en sag, som Parlamentet har beskæftiget sig med gentagne gange.",
    "Det drejer sig om Alexander Nikitin."
]

correct_sentences = [
    "I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.",
    "This is all in accordance with the principles that we have always upheld.",
    "Thank you, Mr Segni, I shall do so gladly.",
    "Indeed, it is quite in keeping with the positions this House has always adopted.",
    "Madam President, I should like to draw your attention to a case in which this Parliament has consistently shown an interest.",
    "It is the case of Alexander Nikitin."
]

splitter = SentenceSplitter()
translator = ChunkTranslator(device=1, delimiter=' ')
max_num_splits = 1

def get_bad_translations(src):
    possible_splits, index_lists = splitter.get_multiple_splits(src, max_num_splits)
    split_err = {}
    # print(possible_splits)
    for split, indexes in zip(possible_splits, index_lists):
        split_str = ' | '.join(split)
        print(split_str)
        # error_sentence = translator.translate(split, indexes)
        error_sentences = translator.translate_multi(split, indexes, 10)
        split_err[split_str] = error_sentences
    return split_err

for danish, correct in zip(danish_sentences, correct_sentences):
    print('Danish: ' + danish)
    print('Correct: ' + correct)
    print('Bad translations:')
    bad_translations = get_bad_translations(danish)
    for bad_translation in bad_translations.keys():
        for b in bad_translations[bad_translation]:
            print(b)
        #print(bad_translations[bad_translation])
    print()