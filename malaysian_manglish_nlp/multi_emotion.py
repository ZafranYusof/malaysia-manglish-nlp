"""Multi-label emotion detection for Malaysian text.

Detects multiple emotions simultaneously with confidence scores.
Supports 8 base emotions: happy, sad, angry, fear, surprise, disgust, love, neutral.
Recognizes common emotion co-occurrence patterns.

Rule-based with Malaysian slang awareness, zero external dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import re

_RE_WORDS = re.compile(r'[a-zA-Z]+')

# ============================================================
# Emotion lexicons (BM + EN + Manglish slang)
# ============================================================

_EMOTION_LEXICON: Dict[str, Dict[str, Any]] = {
    'happy': {
        'words': {
            'gembira', 'happy', 'seronok', 'syok', 'best', 'power', 'padu',
            'mantap', 'solid', 'terbaik', 'hebat', 'awesome', 'amazing',
            'enjoy', 'fun', 'excited', 'yeay', 'yay', 'hore', 'alhamdulillah',
            'grateful', 'bersyukur', 'bangga', 'proud', 'satisfied', 'puas',
            'lega', 'relieved', 'chill', 'relax', 'santai', 'peaceful',
            'blessed', 'rezeki', 'nasib', 'lucky', 'bertuah',
            'suka', 'gemar', 'minat', 'ske', 'hehe', 'hihi',
            'gempak', 'tiptop', 'superb', 'excellent', 'brilliant',
            'legend', 'goat', 'fire', 'lit', 'slaps', 'bussin',
            'wholesome', 'heartwarming', 'inspiring', 'motivated',
            'berbaloi', 'worth', 'valuable', 'meaningful',
            'berjaya', 'menang', 'win', 'achieve', 'accomplished',
            'sihat', 'segar', 'fresh', 'energetic', 'semangat',
            'lawak', 'kelakar', 'funny', 'hilarious', 'lol', 'lmao',
            'haha', 'wkwk', 'rofl',
            'cantik', 'gorgeous', 'stunning', 'beautiful',
            'delicious', 'yummy', 'tasty', 'lazat', 'enak', 'sedap',
            'perfect', 'flawless', 'smooth', 'lancar',
        },
        'patterns': [
            (r'\b(best|power|padu|mantap)\s*(gila|giler|siot|dow)', 1.5),
            (r'woo+h', 1.0), (r'ye+s+', 1.0), (r'ya+y+', 1.0),
            (r'ha+ha+', 1.0), (r'wk+wk+', 1.0),
        ],
        'weight': 1.0,
    },
    'sad': {
        'words': {
            'sedih', 'sad', 'pilu', 'sayu', 'duka', 'kecewa', 'disappointed',
            'hampa', 'down', 'depressed', 'murung', 'suram', 'lonely',
            'sunyi', 'sepi', 'kehilangan', 'lost',
            'menyesal', 'regret', 'rugi', 'malang', 'sial',
            'nangis', 'menangis', 'cry', 'crying', 'tears', 'sebak',
            'terharu', 'touched', 'broken', 'patah', 'luluh', 'hancur',
            'give up', 'putus asa', 'hopeless', 'helpless',
            'sakit hati', 'kecil hati', 'terasa', 'terluka', 'luka',
            'merana', 'derita', 'sengsara', 'azab',
            'homesick', 'nostalgia', 'kenangan', 'memori',
            'ditinggal', 'gone', 'left',
            'sobs', 'sob', 'huhu', 'emo',
        },
        'patterns': [
            (r':\(', 1.5), (r'T_T', 1.5), (r'huhu+', 1.0),
            (r';\(', 1.5), (r'sob+s?', 1.0),
            (r'rindu\s*(gila|sgt|sangat)?\s*(kat|dekat)?\s*(family|keluarga|mak|ayah|abang|kakak|adik|arwah|kampung)', 2.5),
        ],
        'weight': 1.0,
    },
    'angry': {
        'words': {
            'marah', 'angry', 'bengang', 'geram', 'baran', 'triggered',
            'annoyed', 'irritated', 'frustrated',
            'meluat', 'muak', 'benci', 'hate',
            'bodoh', 'bangang', 'bebal', 'stupid', 'idiot', 'sial',
            'babi', 'celaka', 'damn', 'wtf', 'wth',
            'unfair', 'zalim', 'kejam', 'jahat',
            'menyampah', 'terrible', 'horrible',
            'hampeh', 'sampah', 'trash', 'rubbish', 'useless',
            'furious', 'rage', 'livid', 'outraged', 'fuming',
            'pissed', 'mad', 'heated', 'tilted',
            'kurang ajar', 'biadab', 'muka tebal',
            'dengki', 'hasad', 'iri', 'jealous', 'envious',
            'pembuli', 'bully', 'toxic', 'manipulative',
            'fed', 'frust', 'angin',
        },
        'patterns': [
            (r'!{2,}', 1.5), (r'geram\s*(gila|sgt|sangat)', 1.5),
            (r'bengang\s*(gila|sgt)', 1.5), (r'f+u+c+k+', 2.0),
            (r'wtf+', 1.5), (r'fed\s*up', 1.5),
            (r'panas\s*hati', 1.5), (r'naik\s*angin', 1.5),
        ],
        'weight': 1.0,
    },
    'fear': {
        'words': {
            'takut', 'scared', 'afraid', 'fear', 'seram', 'gerun', 'ngeri',
            'cuak', 'gabra', 'nervous', 'anxious', 'worried', 'risau',
            'bimbang', 'gelisah', 'restless', 'panic', 'panik',
            'bahaya', 'dangerous', 'threat', 'ancaman',
            'trauma', 'phobia', 'fobia', 'creepy', 'horror',
            'menakutkan', 'terrifying', 'scary', 'spooky',
            'paranoid', 'suspense', 'thriller', 'eerie', 'haunted',
            'hantu', 'ghost', 'jin', 'syaitan', 'iblis',
            'nightmare', 'insomnia',
            'goosebumps', 'merinding',
            'helpless', 'vulnerable', 'terdedah',
        },
        'patterns': [
            (r'cuak\s*(gila|sgt|dow)', 1.5),
            (r'takut\s*(gila|sgt)', 1.5),
        ],
        'weight': 1.0,
    },
    'surprise': {
        'words': {
            'terkejut', 'surprised', 'shocked', 'stunned', 'speechless',
            'unexpected', 'unbelievable', 'incredible',
            'wow', 'omg', 'alamak', 'astaga', 'astaghfirullah',
            'serious', 'really',
            'insane', 'crazy', 'wild',
            'tergamam', 'ternganga', 'terpegun', 'amazed',
            'sangka',
        },
        'patterns': [
            (r'wh?at\?+', 1.5), (r'hah\?+', 1.5), (r'eh\?+', 1.0),
            (r'serious(ly)?\?', 1.0), (r'tak\s*sangka', 2.0),
            (r'for\s*real', 1.0), (r'no\s*way', 1.5),
        ],
        'weight': 1.0,
    },
    'disgust': {
        'words': {
            'jijik', 'disgusting', 'gross', 'eww', 'yuck', 'geli',
            'meluat', 'muak', 'loya', 'mual', 'nausea',
            'kotor', 'dirty', 'filthy', 'busuk', 'bau',
            'cringe', 'awkward',
            'memalukan', 'embarrassing', 'shameful',
        },
        'patterns': [
            (r'ew+', 1.5), (r'yuck+', 1.5), (r'ugh+', 1.0),
            (r'jijik\s*(gila|sgt|sangat)', 1.5),
        ],
        'weight': 1.2,
    },
    'love': {
        'words': {
            'sayang', 'love', 'cinta', 'kasih', 'rindu', 'miss',
            'comel', 'cute', 'adorable', 'sweet', 'romantic',
            'crush', 'admire', 'kagum', 'terpesona',
            'bahagia', 'bliss', 'soulmate', 'jodoh', 'pasangan',
            'hubby', 'wifey', 'babe', 'baby', 'dear', 'darling',
            'syg', 'iloveyou', 'ily', 'muah', 'xoxo',
            'heart', 'jiwa', 'nyawa',
        },
        'patterns': [
            (r'<3+', 2.0), (r'love\s*you', 2.0),
            (r'sayang\s*(kau|ko|awak|you)', 2.0),
            (r'rindu\s*(kau|ko|awak|dia|you|sangat|sgt)', 2.0),
        ],
        'weight': 1.2,
    },
}

# Intensifiers that boost emotion score
_INTENSIFIERS = {
    'gila', 'giler', 'gile', 'sangat', 'sgt', 'memang', 'mmg',
    'betul', 'btl', 'habis', 'teramat', 'super', 'ultra',
    'very', 'really', 'so', 'damn', 'totally', 'extremely',
    'teruk', 'tahap', 'level', 'max', 'overr',
}

# ============================================================
# Co-occurrence patterns
# ============================================================

_CO_OCCURRENCE_PATTERNS: Dict[str, Dict[str, Any]] = {
    'bittersweet': {
        'emotions': {'happy', 'sad'},
        'description': 'Mixed joy and sadness, nostalgia or farewell',
    },
    'anxious': {
        'emotions': {'fear', 'sad'},
        'description': 'Worry mixed with sadness, dread',
    },
    'frustrated_love': {
        'emotions': {'angry', 'love'},
        'description': 'Love-hate relationship, passionate frustration',
    },
    'jealous_rage': {
        'emotions': {'angry', 'sad'},
        'description': 'Jealousy-driven anger and hurt',
    },
    'surprised_joy': {
        'emotions': {'surprise', 'happy'},
        'description': 'Pleasant surprise, unexpected happiness',
    },
    'shocked_horror': {
        'emotions': {'surprise', 'fear'},
        'description': 'Shock and fear, frightening surprise',
    },
    'disgusted_angry': {
        'emotions': {'disgust', 'angry'},
        'description': 'Revulsion mixed with anger',
    },
    'loving_sadness': {
        'emotions': {'love', 'sad'},
        'description': 'Missing someone, longing',
    },
    'excited_fear': {
        'emotions': {'happy', 'fear'},
        'description': 'Thrill, adrenaline rush, nervous excitement',
    },
    'surprised_disgust': {
        'emotions': {'surprise', 'disgust'},
        'description': 'Shocked by something gross or unpleasant',
    },
}


# ============================================================
# Core logic
# ============================================================

def _detect_co_occurrence(
    emotion_scores: Dict[str, float],
    threshold: float,
) -> Optional[str]:
    """Detect co-occurrence pattern from emotion scores.

    Args:
        emotion_scores: Mapping of emotion to confidence score.
        threshold: Minimum score to consider emotion active.

    Returns:
        Name of co-occurrence pattern, or None.
    """
    active = {e for e, s in emotion_scores.items() if s >= threshold and e != 'neutral'}

    if len(active) < 2:
        return None

    best_match: Optional[str] = None
    best_overlap = 0

    for name, pattern in _CO_OCCURRENCE_PATTERNS.items():
        overlap = len(active & pattern['emotions'])
        if overlap >= 2 and overlap > best_overlap:
            best_match = name
            best_overlap = overlap

    return best_match


def _compute_raw_scores(text: str) -> Tuple[Dict[str, float], Dict[str, List[str]], bool]:
    """Compute raw emotion scores from text.

    Returns:
        (raw_scores, words_found, has_intensifier)
    """
    lower = text.lower()
    words = set(_RE_WORDS.findall(lower))

    scores: Dict[str, float] = {}
    words_found: Dict[str, List[str]] = {}
    has_intensifier = bool(words & _INTENSIFIERS)

    for emotion, data in _EMOTION_LEXICON.items():
        matched = words & data['words']
        score = len(matched) * data['weight']

        for pattern_entry in data['patterns']:
            pat, pat_weight = pattern_entry
            if re.search(pat, lower):
                score += pat_weight

        if matched and has_intensifier:
            score *= 1.5

        scores[emotion] = score
        if matched:
            words_found[emotion] = list(matched)[:5]

    return scores, words_found, has_intensifier


def _normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize raw scores to [0, 1] range."""
    total = sum(raw_scores.values())
    if total > 0:
        return {k: round(v / total, 4) for k, v in raw_scores.items()}
    result = {k: 0.0 for k in raw_scores}
    result['neutral'] = 1.0
    return result


# ============================================================
# Public API
# ============================================================

def detect_multi_emotion(text: str, threshold: float = 0.2) -> Dict[str, Any]:
    """Detect multiple emotions in text simultaneously.

    Args:
        text: Input text.
        threshold: Minimum confidence to include emotion (default 0.2).

    Returns:
        dict: Result with keys:
            - emotions (list[dict]): Emotions above threshold, sorted by confidence
            - dominant (str): Highest confidence emotion
            - is_multi (bool): True if more than 1 emotion above threshold
            - co_occurrence (str|None): Named pattern if detected
            - raw_scores (dict): All 8 emotion raw scores

    Example:
        >>> detect_multi_emotion("sedih tapi grateful")
        {'emotions': [{'emotion': 'sad', ...}, {'emotion': 'happy', ...}], ...}
        >>> detect_multi_emotion("gila best la")
        {'emotions': [{'emotion': 'happy', ...}], ...}
    """
    if not text or not text.strip():
        return {
            'emotions': [{'emotion': 'neutral', 'confidence': 1.0}],
            'dominant': 'neutral',
            'is_multi': False,
            'co_occurrence': None,
            'raw_scores': {
                'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0,
                'surprise': 0.0, 'disgust': 0.0, 'love': 0.0, 'neutral': 1.0,
            },
        }

    raw_scores, words_found, has_intensifier = _compute_raw_scores(text)
    normalized = _normalize_scores(raw_scores)

    # Ensure neutral is in raw_scores
    if 'neutral' not in raw_scores:
        raw_scores['neutral'] = 0.0

    # Build emotion list filtered by threshold
    emotions_list: List[Dict[str, float]] = []
    for emotion, conf in normalized.items():
        if emotion == 'neutral':
            continue
        if conf >= threshold:
            emotions_list.append({
                'emotion': emotion,
                'confidence': round(conf, 4),
            })

    # Sort by confidence descending
    emotions_list.sort(key=lambda x: x['confidence'], reverse=True)

    # Determine dominant
    if emotions_list:
        dominant = emotions_list[0]['emotion']
    else:
        dominant = 'neutral'
        emotions_list = [{'emotion': 'neutral', 'confidence': round(normalized.get('neutral', 1.0), 4)}]

    # Multi-label check
    is_multi = len([e for e in emotions_list if e['emotion'] != 'neutral']) > 1

    # Co-occurrence
    co_occurrence = _detect_co_occurrence(normalized, threshold)

    # Build raw_scores output (all 8 emotions)
    raw_out: Dict[str, float] = {}
    for emo in ('happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'love', 'neutral'):
        raw_out[emo] = round(raw_scores.get(emo, 0.0), 4)

    return {
        'emotions': emotions_list,
        'dominant': dominant,
        'is_multi': is_multi,
        'co_occurrence': co_occurrence,
        'raw_scores': raw_out,
    }


def detect_multi_emotion_batch(texts: List[str], threshold: float = 0.2) -> List[Dict[str, Any]]:
    """Detect multi-label emotions for multiple texts.

    Args:
        texts: List of input texts.
        threshold: Minimum confidence threshold.

    Returns:
        list[dict]: Multi-emotion results per text.
    """
    return [detect_multi_emotion(t, threshold=threshold) for t in texts]


def get_co_occurrence_patterns() -> Dict[str, Dict[str, Any]]:
    """Return known emotion co-occurrence patterns.

    Returns:
        dict: Mapping of pattern name to emotion set and description.

    Example:
        >>> patterns = get_co_occurrence_patterns()
        >>> patterns['bittersweet']['emotions']
        {'happy', 'sad'}
    """
    return {
        name: {
            'emotions': set(data['emotions']),
            'description': data['description'],
        }
        for name, data in _CO_OCCURRENCE_PATTERNS.items()
    }
