"""Malaysian dialect detection and normalization.

Supports: Standard BM, Kelantanese, Terengganu, Negeri Sembilan, Kedah, Sarawak.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import re

# Dialect markers
_DIALECTS = {
    'kelantan': {
        'markers': {
            'demo', 'ambo', 'kawe', 'ore', 'kito', 'gapo', 'guano', 'mano',
            'bilo', 'nok', 'abe', 'mek', 'ning', 'tehe', 'gak',
            'make', 'nasi daghe', 'ayoh', 'mari', 'maghi', 'tubik',
            'ghoyak', 'kecek', 'bakpo', 'sapa', 'hok', 'nyo',
        },
        'patterns': [
            r'\w+o[kh]$',  # barang->bare, makan->make (final vowel shift)
            r'\w+e$',      # -ang -> -e pattern
        ],
        'normalize': {
            'demo': 'mereka', 'ambo': 'saya', 'kawe': 'saya', 'ore': 'orang',
            'kito': 'kita', 'gapo': 'apa', 'guano': 'macam mana',
            'mano': 'mana', 'bilo': 'bila', 'nok': 'nak', 'tok': 'tidak',
            'make': 'makan', 'maghi': 'mari', 'tubik': 'keluar',
            'ghoyak': 'cakap', 'kecek': 'cakap', 'bakpo': 'kenapa',
            'hok': 'yang', 'nyo': 'dia',
        },
    },
    'terengganu': {
        'markers': {
            'mung', 'kite', 'die', 'guane', 'gane', 'mane',
            'bile', 'dok', 'doh', 'ning',
            'makang', 'ikang', 'ayong', 'budok', 'pitih', 'ngate',
            'suke', 'gile', 'sokmo', 'kelih', 'nok',
        },
        'patterns': [
            r'\w+ng$',  # makan->makang, ikan->ikang
            r'\w+ok$',  # budak->budok
        ],
        'normalize': {
            'mung': 'kamu', 'kite': 'kita', 'die': 'dia',
            'guane': 'macam mana', 'gane': 'macam mana', 'mane': 'mana',
            'bile': 'bila', 'nok': 'nak', 'dok': 'tidak', 'doh': 'sudah',
            'makang': 'makan', 'ikang': 'ikan', 'budok': 'budak',
            'pitih': 'duit', 'sokmo': 'selalu', 'kelih': 'tengok',
            'ngate': 'mengata',
        },
    },
    'negeri_sembilan': {
        'markers': {
            'den', 'dio', 'apo', 'bilo', 'gapo',
            'doh', 'yo', 'oi', 'ghoman', 'boghipuk',
            'uwan', 'oncu', 'atok', 'bonda',
            'mongak', 'togok', 'losau', 'polak', 'bosa',
        },
        'patterns': [
            r'\w+o$',  # dia->dio, apa->apo
        ],
        'normalize': {
            'den': 'saya', 'dio': 'dia', 'apo': 'apa', 'mano': 'mana',
            'bilo': 'bila', 'doh': 'sudah', 'ghoman': 'macam mana',
            'boghipuk': 'berkelahi', 'mongak': 'menangis',
            'togok': 'tengok', 'losau': 'lapar', 'polak': 'penat',
            'bosa': 'besar',
        },
    },
    'kedah': {
        'markers': {
            'hang', 'depa', 'awat', 'pasai', 'mai', 'pi', 'dok',
            'hampa', 'kami', 'satgi', 'naa', 'laa', 'daa',
            'loqlaq', 'cemuih', 'pikiaq', 'habaq', 'tengok',
            'pulon', 'ghasa', 'toksah', 'takdak',
        },
        'patterns': [
            r'\w+aq$',  # pikiaq, loqlaq
        ],
        'normalize': {
            'hang': 'kamu', 'depa': 'mereka', 'awat': 'kenapa',
            'pasai': 'pasal', 'mai': 'mari', 'pi': 'pergi',
            'dok': 'duduk', 'hampa': 'kamu semua', 'satgi': 'nanti',
            'cemuih': 'jijik', 'pikiaq': 'fikir', 'habaq': 'beritahu',
            'pulon': 'pula', 'toksah': 'tak payah', 'takdak': 'takde',
        },
    },
    'sabah': {
        'markers': {
            'bah', 'ko', 'mau', 'sia', 'sana', 'sini', 'bilang',
            'kasi', 'sudah', 'nda', 'tida', 'bikin', 'pigi',
            'siok', 'aramai', 'kunun', 'buli', 'tinguk',
        },
        'patterns': [],
        'normalize': {
            'bah': 'lah', 'ko': 'kamu', 'mau': 'nak', 'sia': 'saya',
            'nda': 'tidak', 'tida': 'tidak', 'bikin': 'buat',
            'pigi': 'pergi', 'kasi': 'beri', 'bilang': 'cakap',
            'buli': 'boleh', 'tinguk': 'tengok', 'kunun': 'konon',
        },
    },
    'sarawak': {
        'markers': {
            'kamek', 'kitak', 'sida', 'apa', 'mana', 'bila', 'nang',
            'dolok', 'agik', 'sik', 'mok', 'nemu', 'berik', 'polah',
            'madah', 'nanga', 'rindu', 'kelaka', 'nyamai', 'iboh',
        },
        'patterns': [],
        'normalize': {
            'kamek': 'saya', 'kitak': 'kamu', 'sida': 'mereka',
            'nang': 'memang', 'dolok': 'dulu', 'agik': 'lagi',
            'sik': 'tidak', 'mok': 'mahu', 'nemu': 'jumpa',
            'berik': 'beri', 'polah': 'buat', 'madah': 'cakap',
            'nanga': 'tengok', 'nyamai': 'sedap', 'iboh': 'jangan',
            'kelaka': 'lawak',
        },
    },
}


def detect_dialect(text: str) -> Dict[str, Any]:
    """Detect Malaysian dialect in text.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Result with keys:
            - dialect (str): Detected dialect or 'standard'
            - confidence (float): Detection confidence (0-1)
            - markers_found (list): Dialect markers detected
            - scores (dict): Score per dialect
    
    Example:
        >>> detect_dialect("ambo nok make nasi daghe")
        {'dialect': 'kelantan', 'confidence': 0.9, ...}
        >>> detect_dialect("kamek sik mok polah ya")
        {'dialect': 'sarawak', 'confidence': 0.85, ...}
        >>> detect_dialect("aku nak pergi makan")
        {'dialect': 'standard', 'confidence': 1.0, ...}
    """
    lower = text.lower()
    words = set(re.findall(r'[a-zA-Z]+', lower))
    
    scores = {}
    markers_found = {}
    
    for dialect, data in _DIALECTS.items():
        matched = words & data['markers']
        score = len(matched)
        
        # Pattern matching
        for pattern in data.get('patterns', []):
            matches = re.findall(pattern, lower)
            score += len(matches) * 0.3
        
        scores[dialect] = score
        if matched:
            markers_found[dialect] = list(matched)[:5]
    
    # Determine result
    max_score = max(scores.values()) if scores else 0
    
    if max_score < 1:
        return {
            'dialect': 'standard',
            'confidence': 1.0,
            'markers_found': {},
            'scores': scores,
        }
    
    best_dialect = max(scores, key=scores.get)
    total_words = len(words) if words else 1
    confidence = min(0.95, max_score / max(total_words * 0.3, 1))
    
    return {
        'dialect': best_dialect,
        'confidence': round(confidence, 3),
        'markers_found': markers_found,
        'scores': {k: round(v, 2) for k, v in scores.items()},
    }


def normalize_dialect(text: str, source_dialect: Optional[str] = None) -> str:
    """Normalize dialect text to standard BM.
    
    Parameters:
        text (str): Dialect text.
        source_dialect (str): Source dialect (auto-detect if None).
    
    Returns:
        dict: Result with 'normalized' text and 'dialect' detected.
    
    Example:
        >>> normalize_dialect("ambo nok make nasi")
        {'normalized': 'saya nak makan nasi', 'dialect': 'kelantan'}
        >>> normalize_dialect("kamek sik mok polah")
        {'normalized': 'saya tidak mahu buat', 'dialect': 'sarawak'}
    """
    if source_dialect is None:
        detection = detect_dialect(text)
        source_dialect = detection['dialect']
    
    if source_dialect == 'standard' or source_dialect not in _DIALECTS:
        return {'normalized': text, 'dialect': 'standard', 'changes': []}
    
    normalize_map = _DIALECTS[source_dialect]['normalize']
    words = text.split()
    result = []
    changes = []
    
    for word in words:
        lower = word.lower().strip('.,!?;:')
        punct = ''
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
        
        if lower in normalize_map:
            normalized = normalize_map[lower]
            result.append(normalized + punct)
            changes.append({'from': lower, 'to': normalized})
        else:
            result.append(word)
    
    return {
        'normalized': ' '.join(result),
        'dialect': source_dialect,
        'changes': changes,
    }


def get_dialect_info(dialect: str) -> Dict[str, Any]:
    """Get information about a dialect.
    
    Parameters:
        dialect (str): Dialect name.
    
    Returns:
        dict: Info about the dialect.
    """
    if dialect not in _DIALECTS:
        return {'error': f'Unknown dialect: {dialect}. Available: {list(_DIALECTS.keys())}'}
    
    data = _DIALECTS[dialect]
    return {
        'name': dialect,
        'markers_count': len(data['markers']),
        'normalize_count': len(data['normalize']),
        'sample_markers': list(data['markers'])[:10],
    }


def available_dialects() -> List[str]:
    """List available dialects.
    
    Returns:
        list[str]: Available dialect names.
    """
    return list(_DIALECTS.keys())
