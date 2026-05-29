"""Language detection for Malaysian text."""

from __future__ import annotations

from typing import Any, Dict

import re
import json
import os
from malaysian_manglish_nlp.utils import get_shortforms, get_particles
from malaysian_manglish_nlp.cache import cached

# Common BM words
_BM_MARKERS = {
    'saya', 'aku', 'awak', 'kamu', 'dia', 'mereka', 'kami', 'kita',
    'yang', 'dan', 'atau', 'tetapi', 'tapi', 'untuk', 'dengan', 'dari',
    'ini', 'itu', 'ada', 'tidak', 'tak', 'bukan', 'sudah', 'belum',
    'akan', 'sedang', 'telah', 'boleh', 'perlu', 'harus', 'mahu', 'nak',
    'pergi', 'datang', 'buat', 'makan', 'minum', 'tidur', 'kerja',
    'rumah', 'sekolah', 'universiti', 'kedai', 'jalan', 'kereta',
    'baik', 'bagus', 'besar', 'kecil', 'banyak', 'sikit', 'semua',
    'apa', 'siapa', 'mana', 'bila', 'kenapa', 'macam', 'berapa',
    'pagi', 'petang', 'malam', 'hari', 'minggu', 'bulan', 'tahun',
    'juga', 'pun', 'lagi', 'sahaja', 'sangat', 'memang', 'kalau',
    'sebab', 'pasal', 'lepas', 'sebelum', 'selepas', 'dalam', 'luar',
    # Additional BM markers
    'orang', 'budak', 'kawan', 'keluarga', 'anak', 'abang', 'kakak', 'adik',
    'hendak', 'ingin', 'suka', 'sayang', 'benci', 'takut', 'malu',
    'cantik', 'pandai', 'rajin', 'malas', 'senang', 'susah',
    'cepat', 'lambat', 'dekat', 'jauh', 'tinggi', 'rendah',
    'masuk', 'keluar', 'naik', 'turun', 'duduk', 'berdiri',
    'ambil', 'letak', 'hantar', 'terima', 'bayar', 'beli', 'jual',
    'dengar', 'tengok', 'lihat', 'nampak', 'rasa', 'fikir', 'tahu',
    'kenal', 'ingat', 'lupa', 'faham', 'cuba', 'tolong', 'minta',
    'duit', 'wang', 'harga', 'murah', 'mahal', 'percuma',
    'betul', 'salah', 'benar', 'palsu', 'bagus', 'teruk',
    'panas', 'sejuk', 'hujan', 'panjang', 'pendek',
    'putih', 'hitam', 'merah', 'biru', 'hijau', 'kuning',
    'nasi', 'ayam', 'ikan', 'sayur', 'buah', 'air', 'susu',
    'oleh', 'pada', 'kepada', 'daripada', 'antara', 'tanpa',
    'walaupun', 'supaya', 'sehingga', 'ketika', 'apabila',
    'setiap', 'sedikit', 'beberapa', 'selalu', 'kadang',
    'masih', 'hampir', 'agak', 'amat', 'terlalu',
}

# Common EN words (expanded)
_EN_MARKERS = {
    'the', 'is', 'are', 'was', 'were', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall',
    'this', 'that', 'these', 'those', 'what', 'which', 'who',
    'where', 'when', 'why', 'how', 'because', 'since', 'although',
    'but', 'and', 'or', 'not', 'very', 'really', 'just', 'also',
    'then', 'than', 'after', 'before', 'between', 'through',
    'about', 'with', 'from', 'into', 'during', 'until',
    'some', 'any', 'many', 'much', 'few', 'more', 'most',
    # Additional EN markers
    'they', 'them', 'their', 'there', 'here', 'where',
    'been', 'being', 'going', 'coming', 'doing', 'making',
    'want', 'need', 'like', 'love', 'hate', 'think', 'know',
    'said', 'told', 'asked', 'gave', 'took', 'went', 'came',
    'good', 'bad', 'big', 'small', 'new', 'old', 'long', 'short',
    'every', 'each', 'both', 'either', 'neither', 'another',
    'still', 'already', 'yet', 'never', 'always', 'often',
    'however', 'therefore', 'moreover', 'furthermore', 'meanwhile',
    'actually', 'basically', 'literally', 'definitely', 'probably',
    'something', 'anything', 'nothing', 'everything', 'someone',
    'because', 'although', 'unless', 'whether', 'while',
    'its', 'our', 'your', 'his', 'her', 'my',
    # Common EN verbs/nouns used in Manglish code-switching
    'go', 'buy', 'eat', 'sleep', 'work', 'play', 'meet', 'call',
    'send', 'come', 'get', 'give', 'take', 'make', 'see', 'try',
    'help', 'ask', 'tell', 'say', 'talk', 'walk', 'run', 'drive',
    'wait', 'stop', 'start', 'finish', 'quit', 'join', 'leave',
    'food', 'shop', 'mall', 'store', 'house', 'room', 'car',
    'job', 'school', 'class', 'office', 'meeting', 'movie',
    'phone', 'game', 'book', 'money', 'time', 'place', 'way',
    'friend', 'family', 'people', 'man', 'woman', 'girl', 'boy',
    'today', 'tomorrow', 'yesterday', 'tonight', 'morning', 'night',
    'tired', 'busy', 'free', 'late', 'early', 'fast', 'slow',
    'happy', 'sad', 'angry', 'scared', 'bored', 'excited',
    'serious', 'sure', 'maybe', 'okay', 'fine', 'cool', 'nice',
    'rest', 'break', 'lunch', 'dinner', 'breakfast', 'snack',
    'you', 'me', 'him', 'her', 'us', 'we',
    'up', 'down', 'out', 'in', 'off', 'on', 'back', 'away',
    'now', 'later', 'soon', 'again', 'once', 'twice',
    'restaurant', 'hotel', 'airport', 'station', 'market',
    'lately', 'recently', 'finally', 'suddenly', 'usually',
    'weekend', 'holiday', 'vacation', 'trip', 'plan',
    'cancel', 'confirm', 'update', 'download', 'upload',
}

# Manglish particles (expanded)
_MANGLISH_MARKERS = {
    'la', 'lah', 'lor', 'leh', 'meh', 'geh', 'weh', 'wei', 'eh',
    'kan', 'kot', 'je', 'jer', 'aje', 'bro', 'sis', 'boss',
    'confirm', 'frust', 'syok', 'best', 'power', 'solid',
    'gila', 'siot', 'bodoh', 'damn', 'walao', 'aiyo', 'aiyoh',
    'mamak', 'tapau', 'dabao', 'yum', 'jom', 'gostan',
    'potong', 'kena', 'kantoi', 'cabut', 'blah', 'chao',
    'cincai', 'kiasu', 'kaypoh', 'paiseh', 'shiok',
    'alamak', 'aduh', 'amboi', 'celaka', 'cis',
}


@cached(maxsize=1024)
def detect_language(text: str) -> Dict[str, Any]:
    """Detect if text is BM, EN, or Manglish (code-switched).
    
    Args:
        text: Input text.
    
    Returns:
        dict: Detection result with keys:
            - language (str): 'bm', 'en', 'manglish', or 'unknown'
            - bm_ratio (float): Ratio of BM words (0-1)
            - en_ratio (float): Ratio of EN words (0-1)
            - manglish_markers (int): Count of Manglish-specific markers
            - confidence (float): Detection confidence (0-1)
            - word_count (int): Total word count
    
    Example:
        >>> malaysian_manglish_nlp.detect_language("aku nak pergi makan")
        {'language': 'bm', 'bm_ratio': 1.0, 'en_ratio': 0.0, ...}
        >>> malaysian_manglish_nlp.detect_language("I want to eat some food")
        {'language': 'en', 'bm_ratio': 0.0, 'en_ratio': 0.667, ...}
        >>> malaysian_manglish_nlp.detect_language("aku nak go makan then balik la")
        {'language': 'manglish', ...}
    """
    words = re.findall(r'[a-zA-Z]+', text.lower())
    
    if not words:
        return {"language": "unknown", "bm_ratio": 0, "en_ratio": 0,
                "manglish_markers": 0, "confidence": 0, "word_count": 0}
    
    shortforms = set(get_shortforms().keys())
    particles = set(get_particles().keys())
    
    bm_count = 0
    en_count = 0
    manglish_count = 0
    unknown_count = 0
    
    for word in words:
        if word in _MANGLISH_MARKERS or word in particles:
            manglish_count += 1
        elif word in _BM_MARKERS or word in shortforms:
            bm_count += 1
        elif word in _EN_MARKERS:
            en_count += 1
        else:
            unknown_count += 1
    
    total = len(words)
    recognized = bm_count + en_count + manglish_count
    
    # Ratios based on recognized words only (ignore unknown)
    recognized_total = max(recognized, 1)
    bm_ratio = bm_count / recognized_total
    en_ratio = en_count / recognized_total
    manglish_ratio = manglish_count / recognized_total
    
    # Classification with better thresholds
    if manglish_count >= 2 and (bm_count > 0 or en_count > 0):
        language = "manglish"
        confidence = min(0.95, recognized / total)
    elif bm_count > 0 and en_count > 0 and min(bm_count, en_count) / max(bm_count, en_count) > 0.3:
        # Significant mix of both = manglish
        language = "manglish"
        confidence = min(0.9, recognized / total)
    elif manglish_count >= 1 and (bm_count > 0 and en_count > 0):
        language = "manglish"
        confidence = min(0.85, recognized / total)
    elif manglish_count >= 1 and en_count > 0 and unknown_count > 0:
        # Has manglish marker + english + unknown words (likely BM slang) = mixed
        language = "mixed"
        confidence = min(0.8, recognized / total)
    elif bm_ratio > 0.6:
        language = "bm"
        confidence = min(0.95, bm_ratio)
    elif en_ratio > 0.6:
        language = "en"
        confidence = min(0.95, en_ratio)
    elif bm_count > en_count:
        language = "bm"
        confidence = bm_ratio
    elif en_count > bm_count:
        language = "en"
        confidence = en_ratio
    elif manglish_count > 0:
        language = "manglish"
        confidence = 0.6
    else:
        language = "unknown"
        confidence = 0.3
    
    return {
        "language": language,
        "bm_ratio": round(bm_count / max(total, 1), 3),
        "en_ratio": round(en_count / max(total, 1), 3),
        "manglish_markers": manglish_count,
        "confidence": round(confidence, 3),
        "word_count": total,
    }
