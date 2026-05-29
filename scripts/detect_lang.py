#!/usr/bin/env python3
"""
Language detection for Malaysian text.
Classifies text as BM, EN, or Manglish (mixed).
Usage: python detect_lang.py "aku nak pergi makan, then balik rumah"
"""

import sys
import re
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(SCRIPT_DIR, '..', 'references', 'shortform-dict.json')

# Common BM words (high frequency, unambiguous)
BM_MARKERS = {
    'saya', 'aku', 'awak', 'kamu', 'dia', 'mereka', 'kami', 'kita',
    'yang', 'dan', 'atau', 'tetapi', 'tapi', 'untuk', 'dengan', 'dari',
    'ini', 'itu', 'ada', 'tidak', 'tak', 'bukan', 'sudah', 'belum',
    'akan', 'sedang', 'telah', 'boleh', 'perlu', 'harus', 'mahu', 'nak',
    'pergi', 'datang', 'buat', 'makan', 'minum', 'tidur', 'kerja',
    'rumah', 'sekolah', 'universiti', 'kedai', 'jalan', 'kereta',
    'baik', 'bagus', 'besar', 'kecil', 'banyak', 'sikit', 'semua',
    'apa', 'siapa', 'mana', 'bila', 'kenapa', 'macam', 'berapa',
    'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'lapan',
    'pagi', 'petang', 'malam', 'hari', 'minggu', 'bulan', 'tahun',
    'juga', 'pun', 'lagi', 'sahaja', 'sangat', 'memang', 'kalau',
    'sebab', 'pasal', 'lepas', 'sebelum', 'selepas', 'dalam', 'luar',
}

# Common EN words (high frequency, unambiguous)
EN_MARKERS = {
    'the', 'is', 'are', 'was', 'were', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall',
    'this', 'that', 'these', 'those', 'what', 'which', 'who',
    'where', 'when', 'why', 'how', 'because', 'since', 'although',
    'but', 'and', 'or', 'not', 'very', 'really', 'just', 'also',
    'then', 'than', 'after', 'before', 'between', 'through',
    'about', 'with', 'from', 'into', 'during', 'until',
    'again', 'further', 'once', 'here', 'there', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'only', 'same', 'so', 'too', 'can', 'cannot',
}

# Manglish-specific markers (code-switching indicators)
MANGLISH_MARKERS = {
    'la', 'lah', 'lor', 'leh', 'meh', 'geh', 'weh', 'wei', 'eh',
    'kan', 'kot', 'je', 'jer', 'aje', 'bro', 'sis', 'boss',
    'confirm', 'frust', 'syok', 'best', 'power', 'solid',
}


def detect_language(text):
    """Detect if text is BM, EN, or Manglish."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    
    if not words:
        return {"language": "unknown", "bm_ratio": 0, "en_ratio": 0, "confidence": 0}
    
    # Load shortforms as BM indicators too
    try:
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shortforms = set(data['shortforms'].keys())
    except Exception:
        shortforms = set()
    
    bm_count = 0
    en_count = 0
    manglish_count = 0
    
    for word in words:
        if word in MANGLISH_MARKERS:
            manglish_count += 1
        elif word in BM_MARKERS or word in shortforms:
            bm_count += 1
        elif word in EN_MARKERS:
            en_count += 1
    
    total = len(words)
    bm_ratio = (bm_count + manglish_count * 0.5) / total
    en_ratio = (en_count + manglish_count * 0.5) / total
    
    # Classification logic
    if manglish_count >= 2 or (bm_count > 0 and en_count > 0):
        language = "manglish"
        confidence = min(0.95, (bm_count + en_count + manglish_count) / total)
    elif bm_ratio > 0.4 and en_ratio < 0.1:
        language = "bm"
        confidence = bm_ratio
    elif en_ratio > 0.4 and bm_ratio < 0.1:
        language = "en"
        confidence = en_ratio
    elif bm_ratio > en_ratio:
        language = "bm"
        confidence = bm_ratio
    elif en_ratio > bm_ratio:
        language = "en"
        confidence = en_ratio
    else:
        language = "manglish"
        confidence = 0.5
    
    return {
        "language": language,
        "bm_ratio": round(bm_ratio, 3),
        "en_ratio": round(en_ratio, 3),
        "manglish_markers": manglish_count,
        "confidence": round(confidence, 3),
        "word_count": total
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python detect_lang.py <text>")
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    result = detect_language(text)
    print(json.dumps(result, indent=2))
