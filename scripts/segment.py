#!/usr/bin/env python3
"""
Code-switching segmenter for Manglish text.
Identifies BM vs EN segments within a single sentence.
Usage: python segment.py "aku nak pergi buy some groceries then balik rumah"
"""

import sys
import re
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(SCRIPT_DIR, '..', 'references', 'shortform-dict.json')

# High-confidence BM words
BM_WORDS = {
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
    'atas', 'bawah', 'depan', 'belakang', 'antara', 'setiap',
    'orang', 'anak', 'budak', 'kawan', 'abang', 'kakak', 'adik',
    'mak', 'ayah', 'cikgu', 'doktor', 'polis',
    'cantik', 'comel', 'pandai', 'rajin', 'malas', 'penat',
    'suka', 'sayang', 'benci', 'takut', 'gembira', 'sedih',
    'dekat', 'jauh', 'cepat', 'lambat', 'senang', 'susah',
    'baru', 'lama', 'muda', 'tua', 'tinggi', 'rendah',
    'putih', 'hitam', 'merah', 'biru', 'hijau', 'kuning',
    'nasi', 'air', 'roti', 'ayam', 'ikan', 'sayur', 'buah',
    'duit', 'harga', 'murah', 'mahal', 'bayar', 'beli', 'jual',
}

# High-confidence EN words
EN_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'this', 'that', 'these', 'those', 'what', 'which', 'who',
    'where', 'when', 'why', 'how', 'because', 'since', 'although',
    'but', 'and', 'or', 'not', 'very', 'really', 'just', 'also',
    'then', 'than', 'after', 'before', 'between', 'through',
    'about', 'with', 'from', 'into', 'during', 'until',
    'some', 'any', 'many', 'much', 'few', 'more', 'most',
    'go', 'going', 'went', 'gone', 'come', 'came', 'coming',
    'get', 'got', 'getting', 'make', 'made', 'take', 'took',
    'give', 'gave', 'say', 'said', 'tell', 'told', 'think', 'thought',
    'know', 'knew', 'see', 'saw', 'want', 'need', 'like', 'love',
    'buy', 'bought', 'eat', 'ate', 'drink', 'sleep', 'work',
    'house', 'school', 'shop', 'car', 'phone', 'money', 'food',
    'good', 'bad', 'big', 'small', 'new', 'old', 'long', 'short',
    'already', 'still', 'never', 'always', 'sometimes', 'maybe',
    'here', 'there', 'now', 'today', 'tomorrow', 'yesterday',
    'please', 'thanks', 'sorry', 'hello', 'bye', 'okay',
    'actually', 'basically', 'literally', 'seriously', 'honestly',
}


def segment_text(text):
    """Segment text into BM and EN spans."""
    words = re.findall(r"[\w']+|[^\w\s]", text)
    
    # Load shortforms as BM indicators
    try:
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shortforms = set(data['shortforms'].keys())
        particles = set(data.get('particles', {}).keys())
    except Exception:
        shortforms = set()
        particles = set()
    
    # Tag each word
    tagged = []
    for word in words:
        lower = word.lower()
        if lower in BM_WORDS or lower in shortforms or lower in particles:
            tagged.append({'word': word, 'lang': 'BM'})
        elif lower in EN_WORDS:
            tagged.append({'word': word, 'lang': 'EN'})
        elif re.match(r'^[^\w]$', word):
            tagged.append({'word': word, 'lang': 'PUNCT'})
        else:
            # Heuristic: check word endings
            if lower.endswith(('kan', 'nya', 'lah', 'kah')):
                tagged.append({'word': word, 'lang': 'BM'})
            elif lower.endswith(('ing', 'tion', 'ness', 'ment', 'able', 'ous', 'ive')):
                tagged.append({'word': word, 'lang': 'EN'})
            else:
                tagged.append({'word': word, 'lang': 'UNK'})
    
    # Group into segments
    segments = []
    current_lang = None
    current_words = []
    
    for item in tagged:
        lang = item['lang']
        if lang == 'PUNCT' or lang == 'UNK':
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
    
    # Final segment
    if current_words:
        segments.append({
            'text': ' '.join(current_words),
            'lang': current_lang or 'UNK',
            'word_count': len([w for w in current_words if re.match(r'\w+', w)])
        })
    
    # Calculate switch points
    switch_points = 0
    for i in range(1, len(segments)):
        if segments[i]['lang'] != segments[i-1]['lang']:
            switch_points += 1
    
    return {
        'segments': segments,
        'switch_count': switch_points,
        'total_segments': len(segments),
        'dominant_lang': max(set(s['lang'] for s in segments), key=lambda l: sum(s['word_count'] for s in segments if s['lang'] == l)) if segments else 'UNK'
    }


def format_segments(result):
    """Pretty format segments for display."""
    lines = []
    for seg in result['segments']:
        marker = '[BM]' if seg['lang'] == 'BM' else '[EN]' if seg['lang'] == 'EN' else '[??]'
        lines.append(f"  {marker} {seg['text']}")
    
    lines.append(f"\n  Switches: {result['switch_count']}")
    lines.append(f"  Dominant: {result['dominant_lang']}")
    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python segment.py <text>")
        print('Example: python segment.py "aku nak pergi buy some groceries then balik rumah"')
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    result = segment_text(text)
    
    if '--json' in sys.argv:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_segments(result))
