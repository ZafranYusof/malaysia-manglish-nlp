import sys
sys.path.insert(0, '.')
from malaysian_manglish_nlp.emotion import detect_emotion
from malaysian_manglish_nlp.profanity import detect_profanity
from malaysian_manglish_nlp import sentiment

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

PROFANITY_DATA = [
    ("kau ni bodoh ke apa", True),
    ("pergi mampus la kau", True),
    ("babi la orang tu", True),
    ("celaka punya orang", True),
    ("sial betul hari ni", True),
    ("stupid gila decision dia", True),
    ("wtf is wrong with you", True),
    ("mak kau hijau", True),
    ("makanan sedap gila", False),
    ("aku nak pergi kedai", False),
    ("best la movie tu", False),
    ("dia memang pandai", False),
    ("gila cantik rumah dia", False),
    ("serious ke kau cakap", False),
    ("confirm la dia datang", False),
]

print("=== EMOTION FAILS ===")
for text, expected in EMOTION_DATA:
    result = detect_emotion(text)
    if result['emotion'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {result["emotion"]}')
        print(f'  Scores: {result["scores"]}')
        print()

print("=== PROFANITY FAILS ===")
for text, expected in PROFANITY_DATA:
    result = detect_profanity(text)
    if result['is_toxic'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected toxic={expected}, Got toxic={result["is_toxic"]}')
        print(f'  Words: {result["words_found"]}, Severity: {result["severity"]}')
        print()

print("=== SENTIMENT FAILS ===")
SENTIMENT_DATA = [
    ("best gila makanan dia", "positive"),
    ("sedap sangat nasi goreng tu", "positive"),
    ("power la presentation kau", "positive"),
    ("terbaik service kat sini", "positive"),
    ("mahal gila tapi tak sedap", "negative"),
    ("lambat gila delivery dia", "negative"),
    ("tak best langsung movie tu", "negative"),
]
for text, expected in SENTIMENT_DATA:
    result = sentiment(text)
    if result['sentiment'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {result["sentiment"]}')
        print(f'  Score: {result["score"]}, Pos: {result["positive_words"]}, Neg: {result["negative_words"]}')
        print()
