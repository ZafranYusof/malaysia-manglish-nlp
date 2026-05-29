"""Profanity and toxicity detection for Malaysian text.

Detects Malaysian profanity, insults, and toxic content.
Supports severity levels and censoring.
"""

from __future__ import annotations

from typing import Dict

import re


# Malaysian profanity (severity 1-3: mild, moderate, severe)
_PROFANITY = {
    # Severe (3)
    'babi': 3, 'pukimak': 3, 'kimak': 3, 'lancau': 3, 'pantat': 3,
    'sial': 3, 'haram jadah': 3, 'sundal': 3, 'pelacur': 3,
    'puki': 3, 'tetek': 3, 'konek': 3, 'butoh': 3,
    'fuck': 3, 'shit': 3, 'bitch': 3, 'asshole': 3, 'dick': 3,
    'motherfucker': 3, 'cibai': 3, 'cipap': 3, 'jubur': 3,
    'mak kau': 3, 'bapak kau': 3, 'mak ko': 3,
    
    # Moderate (2)
    'bodoh': 2, 'bangang': 2, 'bebal': 2, 'bengap': 2, 'dunggu': 2,
    'goblok': 2, 'tolol': 2, 'bahlol': 2, 'hampas': 2,
    'celaka': 2, 'jahanam': 2, 'terkutuk': 2, 'laknat': 2,
    'damn': 2, 'hell': 2, 'crap': 2, 'idiot': 2, 'moron': 2,
    'stupid': 2, 'dumb': 2, 'retard': 2,
    'sampah': 2, 'hampeh': 2, 'taik': 2, 'tahi': 2,
    'gampang': 2, 'bangsat': 2, 'keparat': 2, 'bedebah': 2,
    'setan': 2, 'syaitan': 2, 'iblis': 2,
    'anjing': 2, 'beruk': 2, 'monyet': 2, 'lembu': 2,
    
    # Mild (1) — NOTE: 'gila' removed, it's primarily an intensifier in Manglish
    'siot': 1, 'cis': 1, 'ish': 1, 'dey': 1,
    'sewel': 1, 'tak betul': 1, 'sakai': 1,
    'noob': 1, 'loser': 1, 'lame': 1, 'pathetic': 1,
    'wtf': 1, 'wth': 1, 'omfg': 1, 'stfu': 1,
    'pergi mati': 1, 'mampus': 1, 'mampos': 1,
    'pemalas': 1, 'pembohong': 1, 'penipu': 1, 'pengkhianat': 1,
}

# Toxic patterns (regex)
_TOXIC_PATTERNS = [
    (r'mak\s*(kau|ko|hang|kamu)', 3),  # Mak insults
    (r'bapak\s*(kau|ko|hang|kamu)', 3),
    (r'pergi\s*(mati|mampos|mampus)', 2),
    (r'bunuh\s*(diri|kau)', 3),
    (r'(kau|ko)\s*(bodoh|bangang|bebal|bengap)', 2),
    (r'(muka|rupa)\s*(macam|mcm)\s*(babi|monyet|anjing)', 3),
    (r'tak\s*guna\s*(punya|nye)', 1),
    (r'buang\s*(masa|tebiat)', 1),
]

# Leetspeak/evasion patterns (expanded)
_EVASION_MAP = {
    'b4bi': 'babi', 'b@bi': 'babi', 'bab1': 'babi', 'b4b1': 'babi',
    'bbb': 'babi', 'bbi': 'babi',
    'b0d0h': 'bodoh', 'bod0h': 'bodoh', 'b0doh': 'bodoh', 'bdoh': 'bodoh',
    's1al': 'sial', 'si@l': 'sial', 'sy4l': 'sial',
    'fck': 'fuck', 'f*ck': 'fuck', 'fuk': 'fuck', 'phuck': 'fuck',
    'fk': 'fuck', 'f**k': 'fuck', 'fcuk': 'fuck', 'fuq': 'fuck',
    'sh1t': 'shit', 'sh!t': 'shit', 'sht': 'shit', 'sh*t': 'shit',
    'b1tch': 'bitch', 'b!tch': 'bitch', 'btch': 'bitch', 'b*tch': 'bitch',
    'stpd': 'stupid', 'st00pid': 'stupid', 'stup1d': 'stupid',
    'a$$': 'ass', 'a**': 'ass', '@ss': 'ass',
    'd1ck': 'dick', 'd!ck': 'dick', 'dck': 'dick',
    'p*ki': 'puki', 'puk1': 'puki', 'pk1mak': 'pukimak',
    'c1bai': 'cibai', 'cb': 'cibai', 'c1b41': 'cibai',
    'l4ncau': 'lancau', 'lncau': 'lancau', 'lanc4u': 'lancau',
    'k1mak': 'kimak', 'k1m4k': 'kimak', 'kmk': 'kimak',
    'bngng': 'bangang', 'bngap': 'bengap',
    'mmpus': 'mampus', 'mmpos': 'mampos',
    'clk': 'celaka', 'c3laka': 'celaka',
    'anjg': 'anjing', 'anjng': 'anjing', 'anj1ng': 'anjing',
    'bgst': 'bangsat', 'bngst': 'bangsat',
}


def detect_profanity(text: str) -> Dict[str, Any]:
    """Detect profanity and toxic content in text.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Result with keys:
            - is_toxic (bool): Whether text contains profanity
            - severity (int): Max severity found (0-3)
            - words_found (list): Profane words detected
            - score (float): Toxicity score (0-1)
            - categories (list): Types of toxicity found
    
    Example:
        >>> detect_profanity("kau ni bodoh ke apa")
        {'is_toxic': True, 'severity': 2, 'words_found': ['bodoh'], ...}
        >>> detect_profanity("makanan sedap gila")
        {'is_toxic': False, 'severity': 0, ...}
    """
    lower = text.lower()
    
    # Check evasion patterns first
    for evasion, real in _EVASION_MAP.items():
        lower = lower.replace(evasion, real)
    
    words_found = []
    max_severity = 0
    total_score = 0
    categories = set()
    
    # Word-level detection
    text_words = set(re.findall(r'[a-zA-Z]+', lower))
    for word, severity in _PROFANITY.items():
        if ' ' in word:
            # Multi-word phrase
            if word in lower:
                words_found.append(word)
                max_severity = max(max_severity, severity)
                total_score += severity
                if severity >= 3:
                    categories.add('severe_profanity')
                elif severity >= 2:
                    categories.add('insult')
                else:
                    categories.add('mild')
        else:
            if word in text_words:
                words_found.append(word)
                max_severity = max(max_severity, severity)
                total_score += severity
                if severity >= 3:
                    categories.add('severe_profanity')
                elif severity >= 2:
                    categories.add('insult')
                else:
                    categories.add('mild')
    
    # Pattern-level detection
    for pattern, severity in _TOXIC_PATTERNS:
        if re.search(pattern, lower):
            max_severity = max(max_severity, severity)
            total_score += severity
            categories.add('toxic_pattern')
    
    # Normalize score (0-1)
    score = min(1.0, total_score / 5.0)
    
    return {
        'is_toxic': max_severity > 0,
        'severity': max_severity,
        'severity_label': ['clean', 'mild', 'moderate', 'severe'][min(max_severity, 3)],
        'words_found': words_found,
        'score': round(score, 3),
        'categories': sorted(categories),
    }


def censor(text: str, replacement: str = '*', level: str = 1) -> str:
    """Censor profanity in text.
    
    Parameters:
        text (str): Input text.
        replacement (str): Character to replace with (default: '*').
        level (int): Minimum severity to censor (1=all, 2=moderate+, 3=severe only).
    
    Returns:
        str: Censored text.
    
    Example:
        >>> censor("kau ni bodoh ke apa")
        'kau ni b***h ke apa'
        >>> censor("pergi la siot", level=2)
        'pergi la siot'  # siot is level 1, not censored
    """
    result = text
    
    for word, severity in sorted(_PROFANITY.items(), key=lambda x: len(x[0]), reverse=True):
        if severity < level:
            continue
        if ' ' in word:
            # Multi-word: censor whole phrase
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            censored = word[0] + replacement * (len(word) - 2) + word[-1]
            result = pattern.sub(censored, result)
        else:
            # Single word: keep first and last char
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            if len(word) <= 2:
                censored = replacement * len(word)
            else:
                censored = word[0] + replacement * (len(word) - 2) + word[-1]
            result = pattern.sub(censored, result)
    
    return result


def is_safe(text: str, threshold: float = 0) -> bool:
    """Quick check if text is safe (no profanity above threshold).
    
    Parameters:
        text (str): Input text.
        threshold (int): Max acceptable severity (0=no profanity, 1=mild ok, 2=moderate ok).
    
    Returns:
        bool: True if text is safe.
    
    Example:
        >>> is_safe("makanan sedap")
        True
        >>> is_safe("bodoh la kau")
        False
        >>> is_safe("siot la", threshold=1)
        True
    """
    result = detect_profanity(text)
    return result['severity'] <= threshold
