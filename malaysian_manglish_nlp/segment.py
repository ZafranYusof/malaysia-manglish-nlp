"""Code-switching segmenter for Manglish text."""

from __future__ import annotations

from typing import Dict, List

import re
from malaysian_manglish_nlp.utils import get_shortforms, get_particles

_BM_WORDS = {
    'saya', 'aku', 'awak', 'kamu', 'dia', 'mereka', 'kami', 'kita',
    'yang', 'dan', 'atau', 'tetapi', 'tapi', 'untuk', 'dengan', 'dari',
    'ini', 'itu', 'ada', 'tidak', 'tak', 'bukan', 'sudah', 'belum',
    'akan', 'sedang', 'telah', 'boleh', 'perlu', 'harus', 'mahu', 'nak',
    'pergi', 'datang', 'buat', 'makan', 'minum', 'tidur', 'kerja',
    'rumah', 'sekolah', 'kedai', 'jalan', 'kereta',
    'baik', 'bagus', 'besar', 'kecil', 'banyak', 'sikit', 'semua',
    'apa', 'siapa', 'mana', 'bila', 'kenapa', 'macam', 'berapa',
    'pagi', 'petang', 'malam', 'hari', 'minggu', 'bulan', 'tahun',
    'juga', 'pun', 'lagi', 'sahaja', 'sangat', 'memang', 'kalau',
    'orang', 'anak', 'budak', 'kawan', 'abang', 'kakak', 'adik',
    'cantik', 'comel', 'pandai', 'rajin', 'malas', 'penat',
    'suka', 'sayang', 'benci', 'takut', 'gembira', 'sedih',
    'dekat', 'jauh', 'cepat', 'lambat', 'senang', 'susah',
    'nasi', 'air', 'roti', 'ayam', 'ikan', 'sayur', 'buah',
    'duit', 'harga', 'murah', 'mahal', 'bayar', 'beli', 'jual',
}

_EN_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'this', 'that', 'these', 'those', 'what', 'which', 'who',
    'where', 'when', 'why', 'how', 'because', 'since', 'although',
    'but', 'and', 'or', 'not', 'very', 'really', 'just', 'also',
    'then', 'than', 'after', 'before', 'between', 'through',
    'go', 'going', 'went', 'come', 'came', 'get', 'got',
    'make', 'made', 'take', 'took', 'give', 'gave', 'say', 'said',
    'know', 'knew', 'see', 'saw', 'want', 'need', 'like', 'love',
    'buy', 'bought', 'eat', 'ate', 'drink', 'sleep', 'work',
    'good', 'bad', 'big', 'small', 'new', 'old', 'nice', 'cool',
    'already', 'still', 'never', 'always', 'sometimes', 'maybe',
    'here', 'there', 'now', 'today', 'tomorrow', 'yesterday',
    'some', 'any', 'many', 'much', 'few', 'more', 'most',
}


def segment(text: str) -> List[str]:
    """Segment text into BM/EN spans (alias for segment_text).
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Segmentation result.
    """
    return segment_text(text)


def segment_text(text: str) -> Dict[str, Any]:
    """Identify BM vs EN segments in code-switched text.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Result with keys:
            - segments (list): List of {'text', 'lang', 'word_count'}
            - switch_count (int): Number of language switches
            - total_segments (int): Total segment count
            - dominant_lang (str): Most used language
    
    Example:
        >>> malaysian_manglish_nlp.segment("aku nak buy some groceries then balik rumah")
        {'segments': [
            {'text': 'aku nak', 'lang': 'BM', 'word_count': 2},
            {'text': 'buy some groceries then', 'lang': 'EN', 'word_count': 4},
            {'text': 'balik rumah', 'lang': 'BM', 'word_count': 2}
        ], 'switch_count': 2, ...}
    """
    words = re.findall(r"[\w']+|[^\w\s]", text)
    
    shortforms = set(get_shortforms().keys())
    particles = set(get_particles().keys())
    
    # Tag each word
    tagged = []
    for word in words:
        lower = word.lower()
        if lower in _BM_WORDS or lower in shortforms or lower in particles:
            tagged.append({'word': word, 'lang': 'BM'})
        elif lower in _EN_WORDS:
            tagged.append({'word': word, 'lang': 'EN'})
        elif re.match(r'^[^\w]$', word):
            tagged.append({'word': word, 'lang': 'PUNCT'})
        elif lower.endswith(('kan', 'nya', 'lah', 'kah')):
            tagged.append({'word': word, 'lang': 'BM'})
        elif lower.endswith(('ing', 'tion', 'ness', 'ment', 'able', 'ous')):
            tagged.append({'word': word, 'lang': 'EN'})
        else:
            tagged.append({'word': word, 'lang': 'UNK'})
    
    # Group into segments
    segments = []
    current_lang = None
    current_words = []
    
    for item in tagged:
        lang = item['lang']
        if lang in ('PUNCT', 'UNK'):
            current_words.append(item['word'])
            continue
        
        if lang != current_lang and current_lang is not None:
            segments.append({
                'text': ' '.join(current_words),
                'lang': current_lang,
                'word_count': len([w for w in current_words if re.match(r'\w+', w)])
            })
            current_words = []
        
        current_lang = lang
        current_words.append(item['word'])
    
    if current_words:
        segments.append({
            'text': ' '.join(current_words),
            'lang': current_lang or 'UNK',
            'word_count': len([w for w in current_words if re.match(r'\w+', w)])
        })
    
    # Stats
    switch_points = sum(1 for i in range(1, len(segments)) if segments[i]['lang'] != segments[i-1]['lang'])
    
    dominant = 'UNK'
    if segments:
        lang_counts = {}
        for s in segments:
            lang_counts[s['lang']] = lang_counts.get(s['lang'], 0) + s['word_count']
        dominant = max(lang_counts, key=lang_counts.get)
    
    return {
        'segments': segments,
        'switch_count': switch_points,
        'total_segments': len(segments),
        'dominant_lang': dominant,
    }
