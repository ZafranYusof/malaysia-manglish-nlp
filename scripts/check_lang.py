import sys
sys.path.insert(0, '.')
from manglish_nlp import detect_language

LANGUAGE_DATA = [
    ("aku nak pergi makan nasi", "bm"),
    ("saya suka belajar di universiti", "bm"),
    ("dia kerja kat pejabat setiap hari", "bm"),
    ("makanan kat sini memang sedap", "bm"),
    ("I want to go to the store", "en"),
    ("The weather is really nice today", "en"),
    ("She works at the hospital", "en"),
    ("Can you help me with this", "en"),
    ("aku nak go buy some food then balik", "manglish"),
    ("dia cakap nak meet up kat mall", "manglish"),
    ("best gila the movie, kau kena try la", "manglish"),
    ("aku dah tired sangat nak rest kejap", "manglish"),
    ("confirm la dia coming tonight", "manglish"),
    ("serious ke you nak quit job tu", "manglish"),
]

for text, expected in LANGUAGE_DATA:
    result = detect_language(text)
    status = "OK" if result['language'] == expected else "FAIL"
    if status == "FAIL":
        print(f'{status}: "{text}"')
        print(f'  Expected: {expected}, Got: {result["language"]}')
        print(f'  BM:{result["bm_ratio"]}, EN:{result["en_ratio"]}, Manglish:{result["manglish_markers"]}, Confidence:{result["confidence"]}')
        print()
