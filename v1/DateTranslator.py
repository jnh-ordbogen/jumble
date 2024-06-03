test_dates = [
    'den 15. august 2023',
    'den 7. juli 2019',
    'd. 20. september 2022',
    'den 10. maj 2018',
    'den 3. november 2021',
    'd. 25. juni 2020',
    'den 18. oktober',
    'd. 31. december',
    'den 12. december',
    'd. 29. marts 2024',
    '8.10.2011',
    '10/9 2023',
    '5/3 2022',
    '3.4.2023',
    'den 1. juli 2023',
    '25/5 2012',
    'd. 18. februar 1987',
    '7. august 2025',
    '15/10 2002',
    '15/11 2025',
    '30.1.2023',
    'den 5. til den 10. juni 2024',
    'den 8. til den 12. august 2023',
    'd. 12. maj 2025 ',
    '12.10.-1.11.',
    'den 8.-15. marts',
    '20.6.-5.7.',
    '1.-5. juni',
    '1.7.-15.8.',
    'd. 25. juli til d. 2. august',
    '1.11.-30.12.',
    'd. 20. maj',
    'd. 12.-18. februar',
    '8.9.2021',
    '15. oktober',
    '20.11.2017',
    '5.2.2019',
    '3.6.2023',
    'd. 4. juli 1983',
    '7.3.2014',
    '20.10.2025',
    '2013-01-08',
    '2025-06-15',
    '1941-09-17',
    'den 15. oktober',
    '1999-05-28',
    '2008-08-26',
    'd. 25. marts 2025',
    'den 10. september',
    '5.-15. juli',
    '15/11 25',
    '15/11-25',
    '30.1.23',
    '8.9.21 ',
    '20.11.17',
    '5.2.19',
    '3.6.23',
    '7.3.14',
    '20.10.25',
    '20130108',
    '13-01-08',
    '20250615',
    '25-06-15',
    '19410917',
    '410917',
    '19990528',
    '99-05-28',
    '20080826',
    '08-08-26',
    '14. NOV 2023',
    '12. NOV'
]

import re
import string

class DateTranslator:
    def __init__(self) -> None:
        self.month_map = {
            'januar': 'january',
            'februar': 'february',
            'marts': 'march',
            'april': 'april',
            'maj': 'may',
            'juni': 'june',
            'juli': 'july',
            'august': 'august',
            'september': 'september',
            'oktober': 'october',
            'november': 'november',
            'december': 'december',
            'Januar': 'January',
            'Februar': 'February',
            'Marts': 'March',
            'April': 'April',
            'Maj': 'May',
            'Juni': 'June',
            'Juli': 'July',
            'August': 'August',
            'September': 'September',
            'Oktober': 'October',
            'November': 'November',
            'December': 'December'
        }

    def translate(self, date):
        date = date.replace('d.', 'the').replace('den', 'the').replace('til', 'to').replace('til den', 'to the')
        tokens = date.split(' ')
        
        tokens = self._insert_of(tokens)

        new_tokens = []
        for t in tokens:
            if t in self.month_map:
                t = self.month_map[t]
            if self._is_day(t):
                t = self._remove_non_digits(t)
                t = self.ordinal(int(t))
            new_tokens.append(t)
        return ' '.join(new_tokens)

    def _insert_of(self, tokens):
        for i, token in enumerate(tokens):
            if self._is_day(token) and i < len(tokens) - 1 and tokens[i+1] in self.month_map:
                tokens.insert(i+1, 'of')
        return tokens

    def _is_day(self, token):
        #token = self._remove_punctuation(token)
        token = self._remove_non_digits(token)
        return self._has_digit(token) and int(token) > 0 and int(token) <= 31 and not self._is_year(token)

    def _has_digit(self, token):
        pattern = r'\d'
        digit_matches = re.findall(pattern, token)
        return len(digit_matches) > 0

    def _is_year(self, token):
        pattern = r'\d'
        digit_matches = re.findall(pattern, token)
        return len(digit_matches) >= 4

    def ordinal(self, n: int) -> str:
        """
        derive the ordinal numeral for a given number n
        """
        return f"{n:d}{'tsnrhtdd'[(n//10%10!=1)*(n%10<4)*n%10::4]}"
    
    def _remove_punctuation(self, token):
        return token.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_non_digits(self, s):
        return ''.join(re.findall(r'\d+', s))