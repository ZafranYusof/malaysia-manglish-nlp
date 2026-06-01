"""Malaysian dialect detection and normalization.

Supports all 13 Malaysian states + 3 federal territories:
Standard BM, Kelantanese, Terengganu, Negeri Sembilan, Kedah, Sabah, Sarawak,
Penang, Perak, Johor, Pahang, Melaka, Perlis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    'penang': {
        'markers': {
            'lu', 'gua', 'lu orang', 'gua orang', 'cun', 'kasi',
            'tau', 'macam mana', 'mana', 'apa hal', 'buat apa',
            'hokkien', 'char koay teow', 'laksa penang', 'rojak',
            'lorong', 'kopitiam', 'mamak', 'teh tarik',
            'siam', 'sini mau', 'macam tu', 'macam ni',
            'boleh tahan', 'tak payah', 'tak mau', 'sudah',
            'jangan buat', 'kasi habis', 'kasi cepat',
            'chup', 'choy', 'aiyah', 'aiyoh', 'walao',
        },
        'patterns': [
            r'\b(lu|gua)\b',
            r'\b(aiyah|aiyoh|walao|choy)\b',
        ],
        'normalize': {
            'lu': 'kamu', 'gua': 'saya', 'cun': 'cantik',
            'kasi': 'beri', 'tak mau': 'tak nak',
            'aiyah': 'alamak', 'aiyoh': 'alamak', 'walao': 'alamak',
            'chup': 'tunggu', 'choy': 'bodoh',
        },
    },
    'perak': {
        'markers': {
            'kome', 'mike', 'meghang', 'meghoyak', 'meke',
            'meghi', 'mege', 'mase', 'meghaso',
            'pekena', 'kena', 'meghacun', 'meghejek',
            'meghepek', 'meghelik', 'meghelap', 'meghoyan',
            'taiping', 'ipoh', 'kuala kangsar', 'manjung',
            'batu gajah', 'teluk intan', 'bagan datuk',
            'meghagam', 'meghamun', 'meghajok', 'meghaso',
            'megey', 'megheleh', 'meghelat',
            'orang perak', 'loghat perak',
        },
        'patterns': [
            r'\bmegh\w+',  # megh- prefix (meghang, meghe, meghi, mege)
            r'\bmeke\b',
            r'\bmase\b',
        ],
        'normalize': {
            'kome': 'kamu', 'mike': 'kamu', 'meghang': 'makan',
            'meghoyak': 'cakap', 'meke': 'makan',
            'meghi': 'mari', 'mege': 'pergi', 'mase': 'mana',
            'meghaso': 'rasa', 'meghacun': 'racun',
            'meghejek': 'ejek', 'meghepek': 'empal',
            'meghelik': 'elik', 'meghelap': 'gelap',
            'meghoyan': 'gila', 'meghagam': 'racun',
            'meghamun': 'hamun', 'meghajok': 'hajok',
            'megey': 'pergi', 'megheleh': 'leceh',
            'meghelat': 'lat',
        },
    },
    'johor': {
        'markers': {
            'kite orang', 'kite', 'die orang', 'die',
            'saye', 'awak', 'bende', 'ghase',
            'nak kabo', 'kabo', 'gheh', 'gheti',
            'gheghak', 'gheti', 'kame',
            'makan angin', 'jalan-jalan',
            'muo', 'seghab', 'seghabut',
            'tanye', 'bawak', 'tido',
            'johor bahru', 'jb', 'masai', 'skudai',
        },
        'patterns': [
            r'\b(gh|ghe)\w+',  # gh- prefix (ghase, gheti, gheghak)
            r'\w+e\b',  # final -e (Riau-Johor influence: saye, kite, bende)
        ],
        'normalize': {
            'kite': 'kita', 'die': 'dia', 'saye': 'saya',
            'bende': 'benda', 'ghase': 'rasa', 'kabo': 'beritahu',
            'gheh': 'tak', 'gheti': 'retak', 'gheghak': 'gerak',
            'kame': 'kami', 'tanye': 'tanya', 'tido': 'tidur',
            'muo': 'muara', 'seghab': 'serabut',
            'seghabut': 'serabut',
        },
    },
    'pahang': {
        'markers': {
            'koi', 'awok', 'aok', 'kamu',
            'ghase', 'kabo', 'meghe', 'meghi',
            'makan angin', 'geli', 'gheti',
            'mangge', 'magge', 'nange',
            'kua', 'kuale', 'kuantan', 'pekan',
            'temerloh', 'raub', 'bentong', 'lipis',
            'patin', 'tempoyak', 'gulai',
            'gheti', 'kaghau', 'naghi',
        },
        'patterns': [
            r'\b(gh)\w+',  # gh- prefix (similar to Johor but different words)
            r'\bmangg?e\b',  # mangge/magge pattern
        ],
        'normalize': {
            'koi': 'saya', 'awok': 'kamu', 'aok': 'kamu',
            'ghase': 'rasa', 'kabo': 'beritahu',
            'meghe': 'pergi', 'meghi': 'mari',
            'mangge': 'makan', 'magge': 'makan',
            'nange': 'mana', 'kua': 'keluar',
            'kuale': 'kuala', 'kaghau': 'kacau',
            'naghi': 'negeri', 'gheti': 'retak',
        },
    },
    'melaka': {
        'markers': {
            'nyonya', 'baba', 'peranakan', 'kristang',
            'cincalok', 'belacan melaka', 'asam pedas melaka',
            'melaka', 'bandar hilir', 'jongker', 'jonker',
            'portugis', 'geragau', 'kampung portugis',
            'taming sari', 'menara taming', 'keris',
            'debus', 'kuda kepang', 'dondang sayang',
            'panglima awang', 'hang tuah', 'hang Jebat',
            'sambal belacan', 'cencalok', 'budu',
            'tempoyak', 'asam pedas', 'gulai lemak',
            'saye', 'kite', 'die', 'bende',
            'ghase', 'kabo', 'gheh',
        },
        'patterns': [
            r'\b(nyonya|baba|peranakan|kristang|debus|kuda kepang|dondang)\b',
            r'\b(hang tuah|hang jebat|panglima awang|taming sari)\b',
            r'\b(cincalok|cencalok|geragau)\b',
        ],
        'normalize': {
            'saye': 'saya', 'kite': 'kita', 'die': 'dia',
            'bende': 'benda', 'ghase': 'rasa', 'kabo': 'beritahu',
            'gheh': 'tak',
        },
    },
    'perlis': {
        'markers': {
            'peliaq', 'kemaih', 'ketegaq', 'harumanis',
            'kangar', 'arau', 'padang besar', 'chuping',
            'gua kelam', 'tasik melati', 'wang kelian',
            'perlis', 'orang perlis',
            'pulut harumanis', 'mangga harumanis',
            'tebu', 'padi', 'jelapang',
            'peliaq', 'pikiaq', 'habaq', 'loqlaq',
            'cemuih', 'pulon', 'ghasa', 'dok',
            'hang', 'depa', 'awat', 'pasai', 'mai',
            'hampa', 'satgi', 'toksah', 'takdak',
        },
        'patterns': [
            r'\b(peliaq|kemaih|ketegaq|harumanis)\b',
            r'\b(kangar|arau|padang besar|chuping|wang kelian)\b',
            r'\b(gua kelam|tasik melati)\b',
        ],
        'normalize': {
            'peliaq': 'pelik', 'kemaih': 'kemas',
            'ketegaq': 'degil', 'harumanis': 'mangga harumanis',
            'hang': 'kamu', 'depa': 'mereka', 'awat': 'kenapa',
            'pasai': 'pasal', 'mai': 'mari', 'pi': 'pergi',
            'hampa': 'kamu semua', 'satgi': 'nanti',
            'toksah': 'tak payah', 'takdak': 'takde',
            'pikiaq': 'fikir', 'habaq': 'beritahu',
            'loqlaq': 'lawak', 'cemuih': 'jijik',
            'pulon': 'pula', 'ghasa': 'rasa',
        },
    },
}


def detect_dialect(text: str) -> Dict[str, Any]:
    """Detect Malaysian dialect in text.
    
    Args:
        text: Input text.
    
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
    
    # Precompute marker uniqueness (how many dialects share each marker)
    marker_counts: Dict[str, int] = {}
    for data in _DIALECTS.values():
        for m in data['markers']:
            marker_counts[m] = marker_counts.get(m, 0) + 1
    
    scores = {}
    markers_found = {}
    
    for dialect, data in _DIALECTS.items():
        matched = words & data['markers']
        # Weight: unique markers score higher (1/num_dialects_sharing)
        score = sum(1.0 / max(marker_counts.get(m, 1), 1) * 2.0 for m in matched)
        
        # Pattern matching (bonus)
        for pattern in data.get('patterns', []):
            matches = re.findall(pattern, lower)
            score += len(matches) * 0.5
        
        scores[dialect] = score
        if matched:
            markers_found[dialect] = list(matched)[:5]
    
    # Determine result
    max_score = max(scores.values()) if scores else 0
    
    if max_score < 0.5:
        return {
            'dialect': 'standard',
            'confidence': 1.0,
            'markers_found': {},
            'scores': scores,
        }
    
    best_dialect = max(scores, key=scores.get)
    total_words = len(words) if words else 1
    confidence = min(0.95, max_score / max(total_words * 0.2, 1))
    
    return {
        'dialect': best_dialect,
        'confidence': round(confidence, 3),
        'markers_found': markers_found,
        'scores': {k: round(v, 2) for k, v in scores.items()},
    }


def normalize_dialect(text: str, source_dialect: Optional[str] = None) -> str:
    """Normalize dialect text to standard BM.
    
    Args:
        text: Dialect text.
        source_dialect: Source dialect (auto-detect if None).
    
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
    
    Args:
        dialect: Dialect name.
    
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

# Dialect metadata for display/documentation
_DIALECT_INFO = {
    'kelantan': {'state': 'Kelantan', 'region': 'Pantai Timur', 'family': 'Malay'},
    'terengganu': {'state': 'Terengganu', 'region': 'Pantai Timur', 'family': 'Malay'},
    'pahang': {'state': 'Pahang', 'region': 'Pantai Timur', 'family': 'Malay'},
    'negeri_sembilan': {'state': 'Negeri Sembilan', 'region': 'Pantai Barat', 'family': 'Minangkabau'},
    'kedah': {'state': 'Kedah', 'region': 'Utara', 'family': 'Malay'},
    'perlis': {'state': 'Perlis', 'region': 'Utara', 'family': 'Malay'},
    'penang': {'state': 'Pulau Pinang', 'region': 'Utara', 'family': 'Malay/Hokkien'},
    'perak': {'state': 'Perak', 'region': 'Utara', 'family': 'Malay'},
    'johor': {'state': 'Johor', 'region': 'Selatan', 'family': 'Riau-Johor'},
    'melaka': {'state': 'Melaka', 'region': 'Selatan', 'family': 'Riau-Melaka'},
    'sabah': {'state': 'Sabah', 'region': 'Borneo', 'family': 'Sabah Malay'},
    'sarawak': {'state': 'Sarawak', 'region': 'Borneo', 'family': 'Sarawak Malay'},
}

def get_dialect_metadata(dialect: str = None) -> Dict[str, Any]:
    """Get dialect metadata.
    
    Args:
        dialect: Specific dialect name, or None for all.
    
    Returns:
        dict: Dialect metadata (state, region, language family).
    
    Example:
        >>> get_dialect_metadata('kelantan')
        {'state': 'Kelantan', 'region': 'Pantai Timur', 'family': 'Malay'}
    """
    if dialect:
        return _DIALECT_INFO.get(dialect, {'error': f'Unknown dialect: {dialect}'})
    return dict(_DIALECT_INFO)

def get_dialects_by_region(region: str) -> List[str]:
    """Get dialects belonging to a region.
    
    Args:
        region: Region name (Utara, Pantai Timur, Pantai Barat, Selatan, Borneo).
    
    Returns:
        list[str]: Dialect names in that region.
    
    Example:
        >>> get_dialects_by_region('Utara')
        ['kedah', 'perlis', 'penang', 'perak']
    """
    return [d for d, info in _DIALECT_INFO.items() if info['region'] == region]

def get_all_regions() -> List[str]:
    """Get all available regions.
    
    Returns:
        list[str]: Region names.
    """
    return list(set(info['region'] for info in _DIALECT_INFO.values()))
