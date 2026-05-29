"""Stance detection for Malaysian Manglish text.

Determine if text supports, opposes, or is neutral toward a target/topic.
Handles Manglish-specific indicators, negation patterns, and sarcasm awareness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import re

# Stance indicator lexicons
_SUPPORT_INDICATORS = [
    'setuju', 'sokong', 'betul', 'agree', 'support', 'bagus', 'well done',
    'tahniah', '+1', 'yes', 'memang patut', 'tepat', 'correct', 'right',
    'mantap', 'terbaik', 'approved', 'thumbs up', 'aku sokong', 'i agree',
    'betul tu', 'memang betul', 'patut pun', 'good move', 'wise decision',
    'bijak', 'on point', 'exactly', 'precisely', 'damn right', 'fax',
    'facts', 'true', 'legit', 'real talk', 'preach', 'this',
]

_OPPOSE_INDICATORS = [
    'tak setuju', 'bantah', 'salah', 'disagree', 'oppose', 'bodoh',
    'stupid idea', 'tak patut', '-1', 'no way', 'nonsense', 'rubbish',
    'bullshit', 'bs', 'gila ke', 'are you crazy', 'terrible', 'worst',
    'fail', 'gagal', 'reject', 'tolak', 'haram', 'tak boleh terima',
    'unacceptable', 'ridiculous', 'absurd', 'mengarut', 'karut',
    'pembohong', 'liar', 'tipu', 'scam', 'aku bantah', 'i disagree',
    'wrong', 'silap', 'tak betul', 'merepek', 'bongok', 'bangang',
]

_NEUTRAL_INDICATORS = [
    'maybe', 'entah', 'tak sure', 'depends', 'both sides', 'on the fence',
    'not sure', 'tak tau', 'idk', 'ntah', 'boleh jadi', 'mungkin',
    'perhaps', 'could be', 'hard to say', 'susah nak cakap', 'fifty fifty',
    '50/50', 'ada pro ada con', 'dua-dua ada point', 'tengok la',
    'we shall see', 'wait and see', 'tgk dulu', 'belum pasti',
]

# Negation words that flip stance
_NEGATIONS = [
    'tak', 'tidak', 'bukan', 'takde', 'xde', 'x', 'no', 'not',
    "don't", "doesn't", "didn't", "won't", "can't", "cannot",
    'jangan', 'never', 'takkan', 'mustahil',
]

# Double negation patterns (flip back to original)
_DOUBLE_NEGATIONS = [
    'bukan tak', 'bukan tidak', 'tak boleh tak', 'cannot not',
    "can't deny", 'tak dinafikan', 'undeniably', 'tak boleh nafikan',
]

# Sarcasm markers (when combined with positive words, may indicate opposition)
_SARCASM_MARKERS = [
    'la tu', 'konon', 'kononnya', 'ye ke', 'oh ye ke', 'wow',
    'pandai', 'hebat', 'sure', 'right...', 'yeah right', 'oh really',
    'acah', 'perasan', 'poyo', 'haha', 'lol', 'lmao',
]


def _normalize_text(text: str) -> str:
    """Lowercase and strip for matching."""
    return text.lower().strip()


def _find_indicators(text: str, indicator_list: List[str]) -> List[str]:
    """Find which indicators appear in text, return list of matches."""
    normalized = _normalize_text(text)
    found = []
    # Sort by length descending so longer phrases match first
    sorted_indicators = sorted(indicator_list, key=len, reverse=True)
    for indicator in sorted_indicators:
        # Use word boundary matching to avoid substring false positives
        if len(indicator) <= 2:
            # Exact token match for very short indicators
            tokens = re.findall(r'\S+', normalized)
            if indicator in tokens:
                found.append(indicator)
        else:
            # Word boundary match to prevent 'ntah' matching inside 'bantah'
            pattern = r'(?:^|\b|\s)' + re.escape(indicator) + r'(?:\b|\s|$)'
            if re.search(pattern, normalized):
                found.append(indicator)
    return found


def _check_negation(text: str, indicator: Any) -> Dict[str, Any]:
    """Check if an indicator is negated in the text."""
    normalized = _normalize_text(text)
    idx = normalized.find(indicator.lower())
    if idx == -1:
        return False

    # Check for double negation first (cancels out)
    for dn in _DOUBLE_NEGATIONS:
        if dn in normalized:
            return False

    # Check if any negation word appears within 3 words before the indicator
    before_text = normalized[:idx].strip()
    before_words = before_text.split()
    last_words = before_words[-3:] if len(before_words) >= 3 else before_words

    for neg in _NEGATIONS:
        if neg in last_words:
            return True
    return False


def _detect_sarcasm(text: str) -> Dict[str, Any]:
    """Detect potential sarcasm in text."""
    normalized = _normalize_text(text)
    sarcasm_score = 0
    markers_found = []

    for marker in _SARCASM_MARKERS:
        if marker in normalized:
            sarcasm_score += 1
            markers_found.append(marker)

    # Ellipsis or excessive punctuation can indicate sarcasm
    if '...' in text:
        sarcasm_score += 0.5
    if re.search(r'[?!]{2,}', text):
        sarcasm_score += 0.5

    return sarcasm_score >= 1.5, markers_found


def _is_relevant_to_target(text: str, target: str) -> bool:
    """Check if text mentions or is relevant to the target."""
    if target is None:
        return True
    normalized = _normalize_text(text)
    target_lower = target.lower().strip()

    # Direct mention
    if target_lower in normalized:
        return True

    # Check individual words of multi-word target
    target_words = target_lower.split()
    if len(target_words) > 1:
        matches = sum(1 for w in target_words if w in normalized)
        if matches >= len(target_words) * 0.5:
            return True

    return False


def detect_stance(text: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Detect stance of text toward a target/topic.

    Args:
        text: Input text to analyze.
        target: The topic/entity to detect stance toward.
            If None, detects general stance of the text.

    Returns:
        dict: {
            "stance": "support" | "oppose" | "neutral",
            "confidence": float (0.0-1.0),
            "indicators": list[str] - matched indicators
        }

    Example:
        >>> detect_stance("Aku setuju dengan cadangan tu")
        {"stance": "support", "confidence": 0.85, "indicators": ["setuju"]}
        >>> detect_stance("Bodoh la idea ni", target="new policy")
        {"stance": "oppose", "confidence": 0.8, "indicators": ["bodoh"]}
    """
    if not text or not text.strip():
        return {"stance": "neutral", "confidence": 0.5, "indicators": []}

    # Check relevance to target
    if target and not _is_relevant_to_target(text, target):
        return {"stance": "neutral", "confidence": 0.3, "indicators": []}

    # Check for double negation first
    normalized_lower = _normalize_text(text)
    has_double_negation = False
    for dn in _DOUBLE_NEGATIONS:
        if dn in normalized_lower:
            has_double_negation = True
            break

    # Find indicators
    support_found = _find_indicators(text, _SUPPORT_INDICATORS)
    oppose_found = _find_indicators(text, _OPPOSE_INDICATORS)
    neutral_found = _find_indicators(text, _NEUTRAL_INDICATORS)

    # If double negation detected, the overall stance flips:
    # "bukan tak setuju" = actually support
    if has_double_negation:
        # Remove the oppose indicators that are part of double negation
        # and add support instead
        dn_oppose_to_remove = []
        for ind in oppose_found:
            # Check if this oppose indicator is part of a double negation
            for dn in _DOUBLE_NEGATIONS:
                if ind in dn or dn.endswith(ind):
                    dn_oppose_to_remove.append(ind)
                    break
        for ind in dn_oppose_to_remove:
            oppose_found.remove(ind)
        # Also remove support indicators that are substrings of the
        # double-negated oppose phrase (e.g. "setuju" inside "tak setuju")
        support_to_remove = []
        for s_ind in support_found:
            for o_ind in dn_oppose_to_remove:
                if s_ind in o_ind:
                    support_to_remove.append(s_ind)
                    break
        for ind in support_to_remove:
            support_found.remove(ind)
        # The double negation means support
        support_found.append("double_negation")
    else:
        # Handle single negation - negated support becomes oppose and vice versa
        negated_support = []
        for ind in support_found[:]:
            if _check_negation(text, ind):
                negated_support.append(ind)
                support_found.remove(ind)
                oppose_found.append(f"tak {ind}")

        negated_oppose = []
        for ind in oppose_found[:]:
            if ind not in [f"tak {s}" for s in negated_support]:
                if _check_negation(text, ind):
                    negated_oppose.append(ind)
                    oppose_found.remove(ind)
                    support_found.append(f"bukan {ind}")

    # Sarcasm detection - positive words + sarcasm markers = oppose
    is_sarcastic, sarcasm_markers = _detect_sarcasm(text)
    if is_sarcastic and support_found and not oppose_found:
        # Sarcastic positive = actually negative
        oppose_found = support_found[:]
        support_found = []

    # Calculate scores
    support_score = len(support_found)
    oppose_score = len(oppose_found)
    neutral_score = len(neutral_found)

    total = support_score + oppose_score + neutral_score

    if total == 0:
        return {"stance": "neutral", "confidence": 0.3, "indicators": []}

    # Determine stance
    if support_score > oppose_score and support_score > neutral_score:
        stance = "support"
        confidence = min(0.95, 0.5 + (support_score / total) * 0.4 + min(support_score * 0.1, 0.3))
        indicators = support_found
    elif oppose_score > support_score and oppose_score > neutral_score:
        stance = "oppose"
        confidence = min(0.95, 0.5 + (oppose_score / total) * 0.4 + min(oppose_score * 0.1, 0.3))
        indicators = oppose_found
    elif support_score == oppose_score and support_score > 0:
        stance = "neutral"
        confidence = 0.4
        indicators = support_found + oppose_found
    else:
        stance = "neutral"
        confidence = min(0.85, 0.4 + (neutral_score / total) * 0.3)
        indicators = neutral_found

    # Boost confidence if sarcasm detected and flipped
    if is_sarcastic and stance == "oppose":
        indicators.extend(sarcasm_markers)

    return {"stance": stance, "confidence": round(confidence, 2), "indicators": indicators}


def detect_stance_batch(texts: List[str], target: Optional[str] = None) -> List[Dict[str, Any]]:
    """Detect stance for multiple texts.

    Args:
        texts: List of texts to analyze.
        target: The topic/entity to detect stance toward.

    Returns:
        list[dict]: List of stance results, one per text.

    Example:
        >>> detect_stance_batch(["Setuju!", "Tak boleh terima ni"])
        [{"stance": "support", ...}, {"stance": "oppose", ...}]
    """
    return [detect_stance(t, target=target) for t in texts]


def compare_stances(text1: str, text2: str) -> Dict[str, Any]:
    """Compare stances of two texts.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        str: "agree" | "disagree" | "unrelated"

    Example:
        >>> compare_stances("Aku sokong", "Betul, setuju")
        "agree"
        >>> compare_stances("Aku sokong", "Tak setuju langsung")
        "disagree"
    """
    result1 = detect_stance(text1)
    result2 = detect_stance(text2)

    s1 = result1["stance"]
    s2 = result2["stance"]

    # Both neutral with low confidence = unrelated
    if s1 == "neutral" and s2 == "neutral":
        if result1["confidence"] < 0.5 and result2["confidence"] < 0.5:
            return "unrelated"
        return "agree"

    # One neutral = unrelated
    if s1 == "neutral" or s2 == "neutral":
        return "unrelated"

    # Same stance = agree
    if s1 == s2:
        return "agree"

    # Different non-neutral stances = disagree
    return "disagree"


def extract_stance_target(text: str) -> Dict[str, Any]:
    """Extract what the text is taking a stance on.

    Attempts to identify the target/topic of the stance expression.

    Args:
        text: Input text.

    Returns:
        str or None: The identified target, or None if unclear.

    Example:
        >>> extract_stance_target("Aku tak setuju dengan kenaikan harga minyak")
        "kenaikan harga minyak"
        >>> extract_stance_target("Bagus la tu")
        None
    """
    if not text or not text.strip():
        return None

    normalized = _normalize_text(text)

    # Pattern: "... dengan/about/on/pasal/tentang <target>"
    patterns = [
        r'(?:setuju|sokong|bantah|oppose|support|agree|disagree)\s+(?:dengan|about|on|pasal|tentang|dgn)\s+(.+?)(?:\.|!|\?|$)',
        r'(?:tak setuju|tak sokong)\s+(?:dengan|about|on|pasal|tentang|dgn)\s+(.+?)(?:\.|!|\?|$)',
        r'(?:isu|topik|perkara|masalah|issue|topic)\s+(.+?)(?:\.|!|\?|$)',
        r'(?:cadangan|proposal|idea|plan|rancangan)\s+(.+?)(?:\.|!|\?|$)',
        r'(?:pasal|tentang|regarding|about|mengenai)\s+(.+?)(?:\.|!|\?|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            target = match.group(1).strip()
            # Clean up trailing words
            target = re.sub(r'\s+(tu|ni|tu kan|ni kan|la|lah)$', '', target)
            if len(target) > 2:
                return target

    # Pattern: stance word at start, rest is about the target
    # e.g., "Bodoh la idea ni" -> "idea"
    start_patterns = [
        r'^(?:bodoh|stupid|gila|nonsense|bagus|mantap|terbaik)\s+(?:la\s+|lah\s+)?(.+?)(?:\s+(?:tu|ni|la|lah))?$',
    ]

    for pattern in start_patterns:
        match = re.search(pattern, normalized)
        if match:
            target = match.group(1).strip()
            target = re.sub(r'\s+(tu|ni|la|lah)$', '', target)
            if len(target) > 2:
                return target

    return None
