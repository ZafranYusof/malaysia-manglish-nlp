import sys
sys.path.insert(0, '.')
from manglish_nlp.emotion import detect_emotion

EMOTION_DATA = [
    ("gila happy aku dapat result bagus", "happy"),
    ("best sangat holiday kali ni", "happy"),
    ("seronok gila main game ni", "happy"),
    ("enjoy betul concert semalam", "happy"),
    ("sedih gila dengar berita tu", "sad"),
    ("kecewa sangat dengan dia", "sad"),
    ("rindu gila kat family", "sad"),
    ("nangis je aku tengok movie tu", "sad"),
    ("bengang betul la service dia", "angry"),
    ("geram gila dengan orang macam ni", "angry"),
    ("marah sangat sebab kena tipu", "angry"),
    ("fed up dah dengan attitude dia", "angry"),
    ("cuak gila nak exam esok", "fear"),
    ("takut sangat nak present", "fear"),
    ("nervous gila first day kerja", "fear"),
    ("risau pasal result nanti", "fear"),
    ("terkejut gila dia datang", "surprise"),
    ("tak sangka dia boleh buat macam tu", "surprise"),
    ("sayang sangat kat dia", "love"),
    ("rindu gila kat awak", "love"),
    ("jijik gila tengok benda tu", "disgust"),
    ("geli sangat pegang benda tu", "disgust"),
]

for text, expected in EMOTION_DATA:
    result = detect_emotion(text)
    if result['emotion'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {result["emotion"]}')
        print(f'  Scores: {result["scores"]}')
        print(f'  Words: {result["words_found"]}')
        print()
