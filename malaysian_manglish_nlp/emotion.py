"""Emotion detection for Malaysian text.

Detects 8 emotions: happy, sad, angry, fear, surprise, disgust, love, neutral.
Rule-based with Malaysian slang awareness.
"""

from __future__ import annotations

from typing import Dict, List

import re
from malaysian_manglish_nlp.utils import get_shortforms

# Emotion lexicons (BM + EN + Manglish slang)
_EMOTIONS = {
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
            'haha', 'wkwk', 'xD', 'rofl',
            'cantik', 'gorgeous', 'stunning', 'beautiful',
            'delicious', 'yummy', 'tasty', 'lazat', 'enak', 'sedap',
            'perfect', 'flawless', 'smooth', 'lancar',
        },
        'patterns': [r'\b(best|power|padu|mantap)\s*(gila|giler|siot|dow)', r'woo+h', r'ye+s', r'ya+y', r'ha+ha+', r'wk+wk+'],
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
        'patterns': [r':\(', r'T_T', r'huhu+', r';\(', r'sob+s?', (r'rindu\s*(gila|sgt|sangat)?\s*(kat|dekat)?\s*(family|keluarga|mak|ayah|abang|kakak|adik|arwah|kampung)', 2.5)],
        'weight': 1.0,
    },
    'angry': {
        'words': {
            'marah', 'angry', 'bengang', 'geram', 'naik angin', 'panas hati',
            'baran', 'triggered', 'annoyed', 'irritated', 'frustrated',
            'meluat', 'muak', 'sick of', 'benci', 'hate',
            'bodoh', 'bangang', 'bebal', 'stupid', 'idiot', 'sial',
            'babi', 'celaka', 'damn', 'wtf', 'wth',
            'unfair', 'tak adil', 'zalim', 'kejam', 'jahat',
            'menyampah', 'terrible', 'horrible',
            'hampeh', 'sampah', 'trash', 'rubbish', 'useless',
            'furious', 'rage', 'livid', 'outraged', 'fuming',
            'pissed', 'mad', 'heated', 'tilted',
            'kurang ajar', 'biadab', 'tak sedar diri', 'muka tebal',
            'dengki', 'hasad', 'iri', 'jealous', 'envious',
            'pembuli', 'bully', 'toxic', 'manipulative',
            'penindas', 'oppressor', 'abuser',
            'fed', 'frust', 'angin', 'menyampah',
        },
        'patterns': [r'!{2,}', r'geram\s*(gila|sgt|sangat)', r'bengang\s*(gila|sgt)', r'f+u+c+k+', r'wtf+', r'fed\s*up', r'panas\s*hati', r'naik\s*angin'],
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
            'nightmare', 'mimpi ngeri', 'insomnia',
            'goosebumps', 'merinding', 'bulu roma',
            'helpless', 'vulnerable', 'terdedah',
        },
        'patterns': [r'cuak\s*(gila|sgt|dow)', r'takut\s*(gila|sgt)'],
        'weight': 1.0,
    },
    'surprise': {
        'words': {
            'terkejut', 'surprised', 'shocked', 'stunned', 'speechless',
            'unexpected', 'unbelievable', 'incredible',
            'wow', 'omg', 'alamak', 'astaga', 'astaghfirullah',
            'serious', 'betul ke', 'really', 'for real', 'no way',
            'insane', 'crazy', 'mad', 'wild',
            'tergamam', 'ternganga', 'terpegun', 'amazed',
            'sangka', 'expect',
        },
        'patterns': [r'wh?at\?+', r'hah\?+', r'eh\?+', r'serious(ly)?\?', r'tak\s*sangka'],
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
        'patterns': [r'ew+', r'yuck+', r'ugh+', r'jijik\s*(gila|sgt|sangat)'],
        'weight': 1.2,  # Slightly higher weight to win ties vs angry
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
        'patterns': [r'<3+', r'love\s*you', r'sayang\s*(kau|ko|awak|you)', r'rindu\s*(kau|ko|awak|dia|you|sangat|sgt)'],
        'weight': 1.2,  # Higher weight so rindu+context wins over sad
    },
}

# Intensifiers that boost emotion score
_INTENSIFIERS = {
    'gila', 'giler', 'gile', 'sangat', 'sgt', 'memang', 'mmg',
    'betul', 'btl', 'habis', 'teramat', 'super', 'ultra',
    'very', 'really', 'so', 'damn', 'totally', 'extremely',
    'teruk', 'tahap', 'level', 'max', 'overr',
}


def detect_emotion(text: str) -> Dict[str, Any]:
    """Detect emotion in text.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Result with keys:
            - emotion (str): Primary emotion detected
            - scores (dict): Score for each emotion (0-1)
            - confidence (float): Confidence of primary emotion
            - intensified (bool): Whether intensifiers were present
            - words_found (list): Emotion words detected
    
    Example:
        >>> detect_emotion("gila best la makanan dia")
        {'emotion': 'happy', 'confidence': 0.85, ...}
        >>> detect_emotion("sedih gila aku dengar berita tu")
        {'emotion': 'sad', 'confidence': 0.9, ...}
        >>> detect_emotion("bengang betul la service dia")
        {'emotion': 'angry', 'confidence': 0.9, ...}
    """
    lower = text.lower()
    words = set(re.findall(r'[a-zA-Z]+', lower))
    
    scores = {}
    words_found = {}
    has_intensifier = bool(words & _INTENSIFIERS)
    
    for emotion, data in _EMOTIONS.items():
        # Word matching
        matched = words & data['words']
        score = len(matched) * data['weight']
        
        # Pattern matching (supports string or (string, weight) tuples)
        for pattern in data['patterns']:
            if isinstance(pattern, tuple):
                pat, pat_weight = pattern
            else:
                pat, pat_weight = pattern, 1.5
            if re.search(pat, lower):
                score += pat_weight
        
        # Intensifier boost
        if matched and has_intensifier:
            score *= 1.5
        
        scores[emotion] = score
        if matched:
            words_found[emotion] = list(matched)[:5]
    
    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        normalized = {k: round(v / total, 3) for k, v in scores.items()}
    else:
        normalized = {k: 0.0 for k in scores}
        normalized['neutral'] = 1.0
    
    # Primary emotion
    if total == 0:
        primary = 'neutral'
        confidence = 1.0
    else:
        primary = max(scores, key=scores.get)
        confidence = round(normalized[primary], 3)
    
    return {
        'emotion': primary,
        'scores': normalized,
        'confidence': confidence,
        'intensified': has_intensifier,
        'words_found': words_found,
    }


def detect_emotions_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Detect emotions for multiple texts.
    
    Parameters:
        texts (list[str]): Input texts.
    
    Returns:
        list[dict]: Emotion results per text.
    """
    return [detect_emotion(t) for t in texts]


def emotion_summary(texts: List[str]) -> Dict[str, Any]:
    """Get emotion distribution summary for a collection of texts.
    
    Parameters:
        texts (list[str]): Input texts.
    
    Returns:
        dict: Summary with emotion counts and percentages.
    
    Example:
        >>> emotion_summary(["best gila", "sedih la", "ok je"])
        {'total': 3, 'distribution': {'happy': 1, 'sad': 1, 'neutral': 1}, ...}
    """
    results = detect_emotions_batch(texts)
    
    counts = {}
    for r in results:
        e = r['emotion']
        counts[e] = counts.get(e, 0) + 1
    
    total = len(texts)
    percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}
    
    return {
        'total': total,
        'distribution': counts,
        'percentages': percentages,
        'dominant_emotion': max(counts, key=counts.get) if counts else 'neutral',
    }
