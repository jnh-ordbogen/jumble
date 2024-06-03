from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token
from spacy.tokens import Doc
from spacy.util import filter_spans
MONTHS = [
    'januar',
    'februar',
    'marts',
    'april',
    'maj',
    'juni',
    'juli',
    'august',
    'september',
    'oktober',
    'november',
    'december'
]

DOT_DATE_LONG_YEAR_REGEX = r"(?:(?:[12][0-9])|(?:3[01])|(?:[1-9]))\.(?:(?:1[0-2])|(?:[1-9]))\.\d\d\d\d"
DOT_DATE_SHORT_YEAR_REGEX = r"(?:(?:[12][0-9])|(?:3[01])|(?:[1-9]))\.(?:(?:1[0-2])|(?:[1-9]))\.\d\d"
SLASH_DATE_NO_YEAR_REGEX = r"(?:(?:[12][0-9])|(?:3[01])|(?:0[1-9]))\/(?:(?:1[0-2])|(?:[1-9]))"
DASH_DATE_REGEX = r"\d\d\d\d-(?:(?:1[0-2])|(?:0[1-9]))-(?:(?:[12][0-9])|(?:3[01])|(?:0[1-9]))"
ONLY_DAY_DATE_REGEX = r"(?:(?:[12][0-9])|(?:3[01])|(?:[1-9]))\."
DATE_RANGE_LONG_REGEX = r"(?:(?:[12][0-9])|(?:3[01])|(?:[1-9]))\.(?:(?:1[0-2])|(?:[1-9]))\.-(?:(?:[12][0-9])|(?:3[01])|(?:[1-9]))\.(?:(?:1[0-2])|(?:[1-9]))\."
DATE_RANGE_SHORT_REGEX = r"(?:(?:[12][0-9])|(?:3[01])|(?:[1-9]))\.-(?:(?:[12][0-9])|(?:3[01])|(?:[1-9]))\."
SIMPLE_DATE_A_REGEX = r"d\. \d+\."
SIMPLE_DATE_B_REGEX = r"den \d+\."


# https://spacy.io/usage/rule-based-matching

# NOTE! Due to how spacy tokenizes a date range, the following two regexes are missing the final "." to be a correct format
DATE_RANGE_LONG_REGEX_S = DATE_RANGE_LONG_REGEX[:-2]
DATE_RANGE_SHORT_REGEX_S = DATE_RANGE_SHORT_REGEX[:-2]

@Language.factory("date_factory")
def date_merger(nlp, name):
    return DateFactory(nlp.vocab)

class DateFactory:
    def __init__(self, vocab):
        patterns = [
            # [
            #     # 12.10.-1.11.
            #     { 'TEXT': { 'REGEX': DATE_RANGE_LONG_REGEX_S }, 'POS': { 'IN': [ 'NUM', 'ADJ' ] } },
            #     { 'TEXT': '.', 'POS': { 'IN': [ 'NUM', 'X', 'PUNCT' ]} }
            # ],
            # [
            #     # 15.-17.
            #     { 'TEXT': { 'REGEX': DATE_RANGE_SHORT_REGEX_S }, 'POS': { 'IN': [ 'ADJ', 'DET' ]} },
            #     { 'TEXT': '.', 'POS': 'ADJ' },
            # ],
            [
                # Den 15. august 2020
                # d. 15. august 2020
                { 'LOWER': { 'IN': [ 'd.', 'den'] }, 'POS': { 'IN' : [ 'DET', 'ADP' ]} },
                { 'TEXT': { 'REGEX': r'\d+\.$' }, 'POS': 'ADJ' },
                { 'LOWER': { 'IN': MONTHS }, 'POS': 'NOUN' },
                { 'TEXT': { 'REGEX': r'\d\d\d\d$' }, 'POS': 'NUM' },
            ],
            [
                # Den 15. august
                # d. 15. august
                { 'LOWER': { 'IN': [ 'd.', 'den'] }, 'POS': { 'IN' : [ 'DET', 'ADP' ]} },
                { 'TEXT': { 'REGEX': r'\d+\.$' }, 'POS': 'ADJ' },
                { 'LOWER': { 'IN': MONTHS }, 'POS': 'NOUN' },
            ],
            [
                # Den 15.
                # d. 15.
                { 'LOWER': { 'IN': [ 'd.', 'den'] }, 'POS': { 'IN' : [ 'DET', 'ADP' ]} },
                { 'TEXT': { 'REGEX': r'\d+\.$' }, 'POS': 'ADJ' },
            ],
            [
                # 31.12.2023
                { 'TEXT': { 'REGEX': DOT_DATE_LONG_YEAR_REGEX }, 'POS': 'NUM' },
            ],
            [
                # 31.12.23
                { 'TEXT': { 'REGEX': DOT_DATE_SHORT_YEAR_REGEX }, 'POS': 'NUM' },
            ],
            [
                # 2013-01-08
                { 'TEXT': { 'REGEX': DASH_DATE_REGEX }, 'POS': 'NUM' },
            ],
            [
                # 31/12 23
                { 'TEXT': { 'REGEX': SLASH_DATE_NO_YEAR_REGEX }, 'POS': 'NUM' },
                { 'TEXT': { 'REGEX': r'(?:(?:\d\d)|(?:\d\d\d\d))$' }, 'POS': 'NUM' },
            ],
            [
                # 7. august
                { 'TEXT': { 'REGEX': ONLY_DAY_DATE_REGEX }, 'POS': 'ADJ' },
                { 'LOWER': { 'IN': MONTHS }, 'POS': 'NOUN' },
            ],
        ]
        try:
            Token.set_extension("date_token", default=False)
        except:
            pass
        self.matcher = Matcher(vocab)
        self.matcher.add("DATE_TOKEN", patterns)

    def __call__(self, doc: Doc):
        matches = self.matcher(doc)
        spans = []

        for _, start, end in matches:
            spans.append(doc[start:end])
            
        with doc.retokenize() as retokenizer:
            for span in filter_spans(spans):
                retokenizer.merge(span)
                for token in span:
                    token._.date_token = True
        return doc
