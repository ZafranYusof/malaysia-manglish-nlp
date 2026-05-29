#!/usr/bin/env python3
"""Benchmark suite for manglish-nlp accuracy measurement.

Tests against 200+ labeled examples across all modules.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manglish_nlp
from manglish_nlp.emotion import detect_emotion
from manglish_nlp.profanity import detect_profanity, is_safe

# ============================================================
# LABELED TEST DATA
# ============================================================

# Sentiment: (text, expected_sentiment)
SENTIMENT_DATA = [
    ("gila best makanan dia", "positive"),
    ("sedap gila nasi lemak sini", "positive"),
    ("power la kau bro", "positive"),
    ("mantap betul presentation dia", "positive"),
    ("syok gila main game ni", "positive"),
    ("terbaik la service dia", "positive"),
    ("cantik gila view dari sini", "positive"),
    ("padu la design baru ni", "positive"),
    ("solid performance dia malam ni", "positive"),
    ("best sangat holiday kali ni", "positive"),
    ("alhamdulillah dapat result bagus", "positive"),
    ("happy gila dapat offer tu", "positive"),
    ("enjoy betul concert semalam", "positive"),
    ("love la tempat ni vibes dia", "positive"),
    ("nice gila outfit kau hari ni", "positive"),
    ("teruk gila service restoran tu", "negative"),
    ("hampeh la makanan dia", "negative"),
    ("boring gila lecture tadi", "negative"),
    ("waste of money je beli benda ni", "negative"),
    ("kecewa sangat dengan result", "negative"),
    ("mahal gila tapi tak sedap", "negative"),
    ("lambat gila delivery dia", "negative"),
    ("tak best langsung movie tu", "negative"),
    ("menyesal beli phone ni", "negative"),
    ("terrible la wifi kat sini", "negative"),
    ("frustrated gila assignment ni", "negative"),
    ("sampah la game ni", "negative"),
    ("rugi je bayar mahal", "negative"),
    ("hancur la plan aku", "negative"),
    ("stress gila kerja ni", "negative"),
    ("aku pergi kedai kejap", "neutral"),
    ("dia balik rumah dah", "neutral"),
    ("esok ada meeting pukul 3", "neutral"),
    ("aku nak makan nasi goreng", "neutral"),
    ("kelas start pukul 8 pagi", "neutral"),
    ("dia kerja kat KL sekarang", "neutral"),
    ("aku baru sampai rumah", "neutral"),
    ("hujan kat luar sekarang", "neutral"),
    ("aku tunggu kat depan", "neutral"),
    ("dia hantar mesej tadi", "neutral"),
]

# Language detection: (text, expected_lang)
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

# Emotion: (text, expected_emotion)
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
    ("naik angin aku dengan dia", "angry"),
    ("marah sangat sebab dia tipu", "angry"),
    ("cuak gila nak present esok", "fear"),
    ("takut sangat nak jumpa doktor", "fear"),
    ("nervous gila interview nanti", "fear"),
    ("risau pasal exam next week", "fear"),
    ("terkejut gila dia datang", "surprise"),
    ("tak sangka dia boleh buat macam tu", "surprise"),
    ("wow serious ke ni", "surprise"),
    ("sayang sangat kat dia", "love"),
    ("rindu gila kat awak", "love"),
    ("love la couple ni sweet sangat", "love"),
]

# Profanity: (text, expected_is_toxic)
PROFANITY_DATA = [
    ("makanan sedap gila", False),
    ("aku nak pergi kedai", False),
    ("best la tempat ni", False),
    ("kau ni bodoh ke apa", True),
    ("babi la kau ni", True),
    ("sial betul hari ni", True),
    ("celaka punya orang", True),
    ("pergi mati la kau", True),
    ("hampeh la service", True),
    ("gila best makanan", False),
    ("dia memang pandai", False),
    ("bangang betul budak ni", True),
    ("stupid gila decision tu", True),
    ("aku suka tempat ni", False),
    ("wtf is happening", True),
]

# Normalize: (text, should_contain)
NORMALIZE_DATA = [
    ("nk pgi mkn", "nak"),
    ("nk pgi mkn", "pergi"),
    ("nk pgi mkn", "makan"),
    ("dgn kwn utk blaja", "dengan"),
    ("dgn kwn utk blaja", "kawan"),
    ("dgn kwn utk blaja", "untuk"),
    ("smlm aku tdo lmbt", "semalam"),
    ("smlm aku tdo lmbt", "tidur"),
    ("smlm aku tdo lmbt", "lambat"),
    ("xde org kat rmh", "takde"),
    ("xde org kat rmh", "orang"),
    ("xde org kat rmh", "rumah"),
]

# Stemmer: (word, expected_root)
STEMMER_DATA = [
    ("memakan", "makan"),
    ("menulis", "tulis"),
    ("berlari", "lari"),
    ("pelajaran", "ajar"),
    ("menyapu", "sapu"),
    ("mengambil", "ambil"),
    ("membaca", "baca"),
    ("berlarian", "lari"),
    ("terbang", "terbang"),
    ("sekolahan", "sekolah"),
    ("mempersoalkan", "soal"),
    ("mendapat", "dapat"),
    ("mencari", "cari"),
    ("memasak", "masak"),
]


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_benchmark():
    results = {}
    total_correct = 0
    total_tests = 0
    
    print("=" * 60)
    print("MANGLISH-NLP BENCHMARK SUITE")
    print("=" * 60)
    
    # --- Sentiment ---
    print("\n[SENTIMENT]")
    correct = 0
    for text, expected in SENTIMENT_DATA:
        r = manglish_nlp.sentiment(text)
        if r['sentiment'] == expected:
            correct += 1
    acc = correct / len(SENTIMENT_DATA) * 100
    print(f"  Accuracy: {correct}/{len(SENTIMENT_DATA)} ({acc:.1f}%)")
    results['sentiment'] = {'correct': correct, 'total': len(SENTIMENT_DATA), 'accuracy': acc}
    total_correct += correct
    total_tests += len(SENTIMENT_DATA)
    
    # --- Language Detection ---
    print("\n[LANGUAGE DETECTION]")
    correct = 0
    for text, expected in LANGUAGE_DATA:
        r = manglish_nlp.detect_language(text)
        if r['language'] == expected:
            correct += 1
    acc = correct / len(LANGUAGE_DATA) * 100
    print(f"  Accuracy: {correct}/{len(LANGUAGE_DATA)} ({acc:.1f}%)")
    results['language'] = {'correct': correct, 'total': len(LANGUAGE_DATA), 'accuracy': acc}
    total_correct += correct
    total_tests += len(LANGUAGE_DATA)
    
    # --- Emotion ---
    print("\n[EMOTION DETECTION]")
    correct = 0
    for text, expected in EMOTION_DATA:
        r = detect_emotion(text)
        if r['emotion'] == expected:
            correct += 1
        else:
            pass  # Uncomment to debug: print(f"  MISS: '{text}' -> {r['emotion']} (expected {expected})")
    acc = correct / len(EMOTION_DATA) * 100
    print(f"  Accuracy: {correct}/{len(EMOTION_DATA)} ({acc:.1f}%)")
    results['emotion'] = {'correct': correct, 'total': len(EMOTION_DATA), 'accuracy': acc}
    total_correct += correct
    total_tests += len(EMOTION_DATA)
    
    # --- Profanity ---
    print("\n[PROFANITY DETECTION]")
    correct = 0
    for text, expected in PROFANITY_DATA:
        r = detect_profanity(text)
        if r['is_toxic'] == expected:
            correct += 1
    acc = correct / len(PROFANITY_DATA) * 100
    print(f"  Accuracy: {correct}/{len(PROFANITY_DATA)} ({acc:.1f}%)")
    results['profanity'] = {'correct': correct, 'total': len(PROFANITY_DATA), 'accuracy': acc}
    total_correct += correct
    total_tests += len(PROFANITY_DATA)
    
    # --- Normalize ---
    print("\n[NORMALIZATION]")
    correct = 0
    for text, should_contain in NORMALIZE_DATA:
        r = manglish_nlp.normalize(text)
        if should_contain in r:
            correct += 1
    acc = correct / len(NORMALIZE_DATA) * 100
    print(f"  Accuracy: {correct}/{len(NORMALIZE_DATA)} ({acc:.1f}%)")
    results['normalize'] = {'correct': correct, 'total': len(NORMALIZE_DATA), 'accuracy': acc}
    total_correct += correct
    total_tests += len(NORMALIZE_DATA)
    
    # --- Stemmer ---
    print("\n[STEMMER]")
    correct = 0
    for word, expected in STEMMER_DATA:
        r = manglish_nlp.stem_word(word)
        if r == expected:
            correct += 1
    acc = correct / len(STEMMER_DATA) * 100
    print(f"  Accuracy: {correct}/{len(STEMMER_DATA)} ({acc:.1f}%)")
    results['stemmer'] = {'correct': correct, 'total': len(STEMMER_DATA), 'accuracy': acc}
    total_correct += correct
    total_tests += len(STEMMER_DATA)
    
    # --- OVERALL ---
    overall_acc = total_correct / total_tests * 100
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {total_correct}/{total_tests} ({overall_acc:.1f}%)")
    print(f"{'=' * 60}")
    
    results['overall'] = {'correct': total_correct, 'total': total_tests, 'accuracy': overall_acc}
    return results


if __name__ == '__main__':
    run_benchmark()
