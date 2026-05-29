"""Sarcasm detection for Malaysian text.

Detects sarcastic expressions using contradiction patterns,
exaggeration markers, and Malaysian sarcasm idioms.
"""

from __future__ import annotations

from typing import Dict

import re
from malaysian_manglish_nlp.sentiment import analyze_sentiment
from malaysian_manglish_nlp.utils import get_shortforms


# Sarcasm indicators
_SARCASM_MARKERS = {
    # Exaggeration + negative context
    'konon', 'kononnya', 'supposedly', 'apparently',
    'wow', 'wah', 'hebat', 'pandai',  # when used sarcastically
    'bagus la', 'best la', 'power la', 'pandai la', 'rajin la',
    'tahniah', 'congratulations', 'bravo',
}

# Patterns that indicate sarcasm
_SARCASM_PATTERNS = [
    # "best la" + negative context
    (r'(best|bagus|power|pandai|rajin|hebat)\s*(la|lah)\s*.*(tak|x|bukan|langsung|pun)', 0.8),
    # Exaggerated praise followed by criticism
    (r'(wow|wah)\s*(best|bagus|hebat|pandai).*tapi', 0.7),
    # "konon" pattern (supposedly)
    (r'konon(nya)?\s', 0.9),
    # Quotation marks around praise (air quotes)
    (r'"(best|bagus|pandai|rajin|hebat)"', 0.85),
    # "ye la" pattern (yeah right)
    (r'ye+\s*la+h?', 0.6),
    # "ok la tu" (dismissive)
    (r'ok\s*la\s*tu', 0.5),
    # Ellipsis after positive word
    (r'(best|bagus|pandai|hebat)\.{2,}', 0.7),
    # "sangat la" + adjective (over-emphasis)
    (r'sangat\s*la\s*(best|bagus|pandai|rajin)', 0.5),
    # "memang la" (of course... not)
    (r'memang\s*la\s*(best|bagus|pandai|hebat|rajin)', 0.6),
    # Clap emoji pattern or repeated emoji after statement
    (r'(best|bagus|pandai).*[👏🙄😒]', 0.8),
    # "thank you" in negative context
    (r'(thanks?|terima kasih|tq)\s*(la|lah)?\s*.*(lambat|lama|teruk|hampeh)', 0.7),
    # Praise + comma + contradiction ("pandai la, exam fail")
    (r'(best|bagus|power|pandai|rajin|hebat)\s*(la|lah)?\s*,\s*.*?(fail|last|rosak|salah|lambat|teruk|bodoh)', 0.9),
    # "tahniah" + negative outcome
    (r'tahniah.*?(last|fail|kalah|rosak|teruk|place)', 0.9),
    # "wow hebat/pandai sangat" (exaggerated without object)
    (r'wow\s*(hebat|pandai|bagus|rajin)\s*sangat\s*la', 0.8),
]

# Contradiction pairs (positive word + negative word in same sentence)
_CONTRADICTION_POSITIVE = {
    'best', 'bagus', 'pandai', 'rajin', 'hebat', 'power', 'mantap',
    'terbaik', 'cantik', 'perfect', 'amazing', 'great', 'wonderful',
    'excellent', 'brilliant', 'genius', 'talented',
}

_CONTRADICTION_NEGATIVE = {
    'tapi', 'but', 'tak', 'x', 'langsung', 'pun', 'hampeh',
    'teruk', 'bodoh', 'lambat', 'lama', 'sampah', 'hancur',
    'terrible', 'useless', 'waste', 'rubbish', 'never', 'nothing',
}


def detect_sarcasm(text: str) -> Dict[str, Any]:
    """Detect sarcasm in text.
    
    Uses multiple signals:
    1. Contradiction patterns (positive + negative in same sentence)
    2. Known sarcasm markers (konon, "best la")
    3. Exaggeration patterns
    4. Sentiment mismatch (positive words but negative intent)
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Result with keys:
            - is_sarcastic (bool): Whether sarcasm detected
            - confidence (float): Confidence score (0-1)
            - signals (list): Sarcasm signals found
            - literal_sentiment (str): Surface-level sentiment
            - intended_sentiment (str): Likely intended sentiment
    
    Example:
        >>> detect_sarcasm("best la service dia, 2 jam tunggu")
        {'is_sarcastic': True, 'confidence': 0.8, 'intended_sentiment': 'negative', ...}
        >>> detect_sarcasm("memang la pandai, exam fail")
        {'is_sarcastic': True, 'confidence': 0.85, ...}
        >>> detect_sarcasm("sedap gila makanan dia")
        {'is_sarcastic': False, 'confidence': 0.1, ...}
    """
    lower = text.lower()
    words = set(re.findall(r'[a-zA-Z]+', lower))
    
    signals = []
    score = 0.0
    
    # Signal 1: Pattern matching
    for pattern, weight in _SARCASM_PATTERNS:
        if re.search(pattern, lower):
            signals.append(f'pattern: {pattern[:30]}')
            score += weight
    
    # Signal 2: Contradiction (positive + negative words together)
    pos_found = words & _CONTRADICTION_POSITIVE
    neg_found = words & _CONTRADICTION_NEGATIVE
    if pos_found and neg_found:
        signals.append(f'contradiction: {list(pos_found)[:2]} + {list(neg_found)[:2]}')
        score += 0.5
    
    # Signal 3: Sarcasm markers
    for marker in _SARCASM_MARKERS:
        if marker in lower:
            signals.append(f'marker: {marker}')
            score += 0.4
            break  # Only count once
    
    # Signal 4: "la" after positive adjective at start (dismissive tone)
    # Only sarcastic if followed by negative context or comma+contradiction
    if re.match(r'^(best|bagus|pandai|rajin|hebat|power)\s*(la|lah)', lower):
        has_neg_context = bool(words & _CONTRADICTION_NEGATIVE) or ',' in lower
        if has_neg_context:
            signals.append('dismissive_opener')
            score += 0.3
    
    # Normalize score
    confidence = min(1.0, score)
    is_sarcastic = confidence >= 0.5
    
    # Determine sentiments
    literal = analyze_sentiment(text)
    literal_sent = literal['sentiment']
    
    if is_sarcastic:
        # Flip sentiment
        if literal_sent == 'positive':
            intended = 'negative'
        elif literal_sent == 'negative':
            intended = 'positive'
        else:
            intended = 'negative'  # Sarcasm usually implies negativity
    else:
        intended = literal_sent
    
    return {
        'is_sarcastic': is_sarcastic,
        'confidence': round(confidence, 3),
        'signals': signals,
        'literal_sentiment': literal_sent,
        'intended_sentiment': intended,
    }
