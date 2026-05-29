import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'tests'))
from manglish_nlp import detect_language
from manglish_nlp.sarcasm import detect_sarcasm
from manglish_nlp.emotion import detect_emotion
from manglish_nlp.dialect import detect_dialect
from benchmark_expanded import LANGUAGE_DATA, SARCASM_DATA, EMOTION_DATA, DIALECT_DATA, SENTIMENT_DATA
from manglish_nlp import sentiment

print("=== LANGUAGE FAILS ===")
for text, expected in LANGUAGE_DATA:
    r = detect_language(text)
    if r['language'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {r["language"]}')
        print(f'  BM:{r["bm_ratio"]}, EN:{r["en_ratio"]}, Manglish:{r["manglish_markers"]}')
        print()

print("=== SARCASM FAILS ===")
for text, expected in SARCASM_DATA:
    r = detect_sarcasm(text)
    if r['is_sarcastic'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {r["is_sarcastic"]}')
        print(f'  Confidence: {r["confidence"]}, Signals: {r["signals"]}')
        print()

print("=== EMOTION FAILS ===")
for text, expected in EMOTION_DATA:
    r = detect_emotion(text)
    if r['emotion'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {r["emotion"]}')
        print()

print("=== DIALECT FAILS ===")
for text, expected in DIALECT_DATA:
    r = detect_dialect(text)
    if r['dialect'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {r["dialect"]}')
        print(f'  Scores: {r["scores"]}')
        print()

print("=== SENTIMENT FAILS ===")
for text, expected in SENTIMENT_DATA:
    r = sentiment(text)
    if r['sentiment'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {r["sentiment"]}')
        print(f'  Score: {r["score"]}, Pos: {r["positive_words"][:3]}, Neg: {r["negative_words"][:3]}')
        print()
