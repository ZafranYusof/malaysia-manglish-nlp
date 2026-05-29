"""Contextual spelling correction.

Uses surrounding words (bigram/trigram context) to improve correction accuracy.
Better than pure edit-distance for ambiguous cases.
"""

from __future__ import annotations

from typing import Dict

import re
from malaysian_manglish_nlp.spelling import correct_word, _BM_DICTIONARY
from malaysian_manglish_nlp.utils import get_shortforms

# Common bigrams (word pairs that frequently appear together)
_COMMON_BIGRAMS = {
    ('nak', 'pergi'), ('nak', 'makan'), ('nak', 'tidur'), ('nak', 'balik'),
    ('nak', 'buat'), ('nak', 'tanya'), ('nak', 'cakap'), ('nak', 'beli'),
    ('tak', 'nak'), ('tak', 'boleh'), ('tak', 'ada'), ('tak', 'tahu'),
    ('tak', 'faham'), ('tak', 'sempat'), ('tak', 'kisah'), ('tak', 'payah'),
    ('dah', 'makan'), ('dah', 'balik'), ('dah', 'siap'), ('dah', 'sampai'),
    ('dah', 'tidur'), ('dah', 'habis'), ('dah', 'lama'), ('dah', 'bayar'),
    ('aku', 'nak'), ('aku', 'tak'), ('aku', 'dah'), ('aku', 'pergi'),
    ('dia', 'nak'), ('dia', 'tak'), ('dia', 'dah'), ('dia', 'pergi'),
    ('kau', 'nak'), ('kau', 'tak'), ('kau', 'dah'), ('kau', 'pergi'),
    ('pergi', 'makan'), ('pergi', 'kerja'), ('pergi', 'sekolah'),
    ('pergi', 'kedai'), ('pergi', 'rumah'), ('pergi', 'kelas'),
    ('makan', 'nasi'), ('makan', 'roti'), ('makan', 'ayam'),
    ('beli', 'barang'), ('beli', 'makanan'), ('beli', 'tiket'),
    ('bayar', 'duit'), ('bayar', 'hutang'), ('bayar', 'bil'),
    ('balik', 'rumah'), ('balik', 'kerja'), ('balik', 'lambat'),
    ('sampai', 'rumah'), ('sampai', 'sana'), ('sampai', 'sekarang'),
    ('macam', 'mana'), ('macam', 'tu'), ('macam', 'ni'),
    ('kat', 'rumah'), ('kat', 'kedai'), ('kat', 'sekolah'), ('kat', 'sini'),
    ('dekat', 'rumah'), ('dekat', 'sini'), ('dekat', 'sana'),
    ('lepas', 'tu'), ('lepas', 'makan'), ('lepas', 'kerja'),
    ('boleh', 'tak'), ('boleh', 'pergi'), ('boleh', 'buat'),
    ('kena', 'pergi'), ('kena', 'buat'), ('kena', 'bayar'),
    ('hari', 'ni'), ('hari', 'tu'), ('hari', 'esok'),
    ('pukul', 'berapa'), ('berapa', 'ringgit'), ('berapa', 'lama'),
    ('terima', 'kasih'), ('minta', 'maaf'), ('tak', 'pe'),
}

# Common trigrams
_COMMON_TRIGRAMS = {
    ('aku', 'nak', 'pergi'), ('aku', 'nak', 'makan'), ('aku', 'nak', 'balik'),
    ('aku', 'tak', 'nak'), ('aku', 'tak', 'boleh'), ('aku', 'tak', 'tahu'),
    ('dia', 'tak', 'nak'), ('dia', 'dah', 'balik'), ('dia', 'nak', 'pergi'),
    ('tak', 'nak', 'pergi'), ('tak', 'boleh', 'buat'), ('tak', 'ada', 'masa'),
    ('nak', 'pergi', 'makan'), ('nak', 'balik', 'rumah'),
    ('macam', 'mana', 'nak'), ('kat', 'mana', 'tu'),
    ('dah', 'lama', 'tak'), ('dah', 'sampai', 'rumah'),
}


def _bigram_score(word1: str, word2: str) -> float:
    """Score how likely two words appear together."""
    pair = (word1.lower(), word2.lower())
    if pair in _COMMON_BIGRAMS:
        return 2.0
    # Reverse check
    if (pair[1], pair[0]) in _COMMON_BIGRAMS:
        return 1.5
    return 0.0


def _trigram_score(word1: str, word2: str, word3: str) -> float:
    """Score how likely three words appear together."""
    triple = (word1.lower(), word2.lower(), word3.lower())
    if triple in _COMMON_TRIGRAMS:
        return 3.0
    return 0.0


def correct_contextual(text: str, max_distance: int = 2) -> Dict[str, Any]:
    """Contextual spelling correction using surrounding words.
    
    Unlike basic `correct()`, this considers neighboring words to pick
    the best correction when multiple candidates exist.
    
    Parameters:
        text (str): Input text.
        max_distance (int): Max edit distance for candidates.
    
    Returns:
        dict: Result with 'corrected', 'changes', 'original'.
    
    Example:
        >>> correct_contextual("aku nk prgi mkn nsi")
        {'corrected': 'aku nak pergi makan nasi', ...}
        >>> correct_contextual("dia dh smpai rmh")
        {'corrected': 'dia dah sampai rumah', ...}
    """
    shortforms = set(get_shortforms().keys())
    all_valid = _BM_DICTIONARY | shortforms
    
    words = text.split()
    corrected = []
    changes = []
    
    for i, word in enumerate(words):
        punct = ''
        clean_word = word
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
            clean_word = word[:-1]
        
        lower = clean_word.lower()
        
        # Skip if valid or too short
        if lower in all_valid or len(lower) <= 2 or re.search(r'\d', lower):
            corrected.append(word)
            continue
        
        # Get candidates
        result = correct_word(lower, max_distance=max_distance, top_n=10)
        
        if result['is_valid']:
            corrected.append(word)
            continue
        
        if not result['suggestions']:
            corrected.append(word)
            continue
        
        # Score candidates using context
        best_word = result['suggestions'][0]['word']
        best_score = -1
        
        prev_word = words[i-1].lower().strip('.,!?;:') if i > 0 else ''
        next_word = words[i+1].lower().strip('.,!?;:') if i < len(words) - 1 else ''
        prev2_word = words[i-2].lower().strip('.,!?;:') if i > 1 else ''
        
        for suggestion in result['suggestions']:
            candidate = suggestion['word']
            dist = suggestion['distance']
            
            # Base score (prefer shorter distance)
            score = (max_distance + 1 - dist) * 2
            
            # Bigram bonus
            if prev_word:
                score += _bigram_score(prev_word, candidate)
            if next_word:
                score += _bigram_score(candidate, next_word)
            
            # Trigram bonus
            if prev_word and next_word:
                score += _trigram_score(prev_word, candidate, next_word)
            if prev2_word and prev_word:
                score += _trigram_score(prev2_word, prev_word, candidate)
            
            # Prefer dictionary words over shortforms
            if candidate in _BM_DICTIONARY:
                score += 1.0
            
            if score > best_score:
                best_score = score
                best_word = candidate
        
        corrected.append(best_word + punct)
        changes.append({
            'original': clean_word,
            'corrected': best_word,
            'position': i,
            'context_score': round(best_score, 2),
        })
    
    return {
        'corrected': ' '.join(corrected),
        'changes': changes,
        'original': text,
    }
