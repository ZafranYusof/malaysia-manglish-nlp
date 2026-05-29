"""
malaysian_manglish_nlp.hate_speech - Hate speech detection for Malaysian context.

Detects hate speech in Manglish text with awareness of Malaysian racial,
religious, and cultural context. Classifies by category and severity.

Categories:
    racial      - targeting race/ethnicity (Malay, Chinese, Indian, etc)
    religious   - targeting religion (Islam, Christianity, Buddhism, Hinduism)
    sexist      - gender-based hate
    xenophobic  - targeting foreigners (Bangla, Indon, etc)
    homophobic  - targeting LGBTQ+
    none        - no hate speech detected

Usage:
    from malaysian_manglish_nlp.hate_speech import detect_hate_speech, is_hate_speech

    detect_hate_speech("some text here")
    # {"is_hate": False, "category": "none", "confidence": 0.95, ...}

    is_hate_speech("some text")
    # True/False
"""

from __future__ import annotations

from typing import Any, Dict, List

import re
from collections import defaultdict


# --- Pattern Categories ---
# These define CATEGORIES of hateful patterns without listing actual slurs.
# Patterns use euphemistic references and structural detection.

# Ethnic/racial group references (neutral - not hate speech alone)
ETHNIC_GROUPS = {
    "malay": ["melayu", "malay", "bumiputera", "bumi"],
    "chinese": ["cina", "chinese", "tionghua"],
    "indian": ["india", "indian", "tamil", "keling"],
    "orang asli": ["orang asli", "jakun"],
    "sikh": ["sikh", "singh"],
    "eurasian": ["serani", "eurasian"],
}

RELIGIOUS_GROUPS = {
    "islam": ["islam", "muslim", "muslimah", "ustaz", "ustazah"],
    "christianity": ["kristian", "christian", "catholic", "protestant"],
    "buddhism": ["buddha", "buddhist"],
    "hinduism": ["hindu"],
    "sikhism": ["sikh"],
}

FOREIGN_GROUPS = {
    "bangladeshi": ["bangla", "bangladesh", "bangladeshi"],
    "indonesian": ["indon", "indonesia", "indonesian"],
    "myanmar": ["rohingya", "myanmar", "burma"],
    "nepali": ["nepal", "nepali"],
    "filipino": ["pinoy", "filipina", "filipino"],
    "vietnamese": ["vietnam", "vietnamese"],
    "pakistani": ["paki", "pakistan"],
}

GENDER_TERMS = {
    "women": ["perempuan", "wanita", "woman", "women", "female", "girl"],
    "men": ["lelaki", "man", "men", "male"],
    "feminist": ["feminist", "feminism"],
}

LGBTQ_TERMS = {
    "lgbtq": ["LGBT", "LGBTQ", "gay", "lesbian", "bisexual", "transgender",
              "trans", "queer", "homoseksual", "pondan", "mak nyah",
              "pengkid", "tomboy"],
}

# --- Hate Indicators ---
# These are structural patterns that indicate hate when combined with group references

# Dehumanizing verbs/phrases (when directed at groups)
DEHUMANIZING_PATTERNS = [
    r'\b(balik\s+(negara|kampung|tempat\s+asal))\b',  # "go back to your country"
    r'\b(halau|usir|buang|hapus|bunuh|bantai|tembak)\b',  # violent verbs
    r'\b(binatang|haiwan|anjing|babi|monyet|beruk|kera)\b',  # animal comparisons
    r'\b(sampah|sial|celaka|laknat|hina|kotor|najis|jijik)\b',  # degrading terms
    r'\b(bodoh|bangang|bengap|bebal|dungu|tolol|bahlul)\b',  # intellectual slurs
    r'\b(tak\s+guna|tak\s+berguna|parasit|penyakit|virus)\b',  # dehumanizing labels
    r'\b(mati\s+la|pergi\s+mati|mampos|mampus)\b',  # death wishes
]

# Stereotyping patterns
STEREOTYPING_PATTERNS = [
    r'\b(semua|semue|sume|all)\s+\w+\s+(sama|same|macam\s+tu)\b',  # "all X are the same"
    r'\b(memang\s+la|confirm\s+la|dah\s+la)\s+\w+',  # "of course [group] would..."
    r'\b(dasar|memang)\s+\w+',  # "typical [group]"
]

# Threat patterns
THREAT_PATTERNS = [
    r'\b(nak\s+bunuh|nak\s+bantai|nak\s+tembak|nak\s+bakar)\b',
    r'\b(kena\s+bunuh|kena\s+bantai|kena\s+halau|kena\s+usir)\b',
    r'\b(patut\s+(mati|bunuh|halau|bakar|tembak))\b',
    r'\b(jangan\s+bagi\s+(masuk|duduk|tinggal))\b',
]

# Supremacist patterns
SUPREMACIST_PATTERNS = [
    r'\b(ketuanan|supremacy|superior|dominasi)\b',
    r'\b(tanah\s+(melayu|kita)|negara\s+(kita|kami))\b.*\b(balik|keluar|halau)\b',
]

# --- False Positive Contexts ---
# Contexts where potentially flagged words are NOT hate speech

FOOD_CONTEXT_WORDS = [
    "makan", "masak", "goreng", "rebus", "bakar", "panggang",
    "kuah", "sambal", "kari", "rendang", "recipe", "resipi",
    "menu", "order", "sedap", "lazat", "makanan", "lauk",
    "restoran", "restaurant", "mamak", "warung", "kedai makan",
    "nasi", "mi", "bihun", "sup", "gulai", "daging",
    "satay", "sate", "char", "kuey", "laksa",
]

NEUTRAL_DISCUSSION_WORDS = [
    "sejarah", "history", "budaya", "culture", "tradisi", "tradition",
    "perayaan", "festival", "celebrate", "sambut", "adat", "custom",
    "belajar", "study", "research", "kajian", "discuss", "bincang",
    "article", "berita", "news", "report", "laporan",
    "statistik", "statistic", "data", "survey", "census",
]

# Leetspeak/evasion patterns
LEETSPEAK_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '@': 'a', '$': 's', '!': 'i',
}


def _normalize_leetspeak(text: str) -> str:
    """Convert leetspeak evasion back to normal text."""
    result = text
    for leet, normal in LEETSPEAK_MAP.items():
        result = result.replace(leet, normal)
    return result


def _normalize_text(text: str) -> str:
    """Normalize text for analysis."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _check_food_context(text: str) -> bool:
    """Check if text is in a food context (false positive prevention)."""
    text_lower = text.lower()
    food_count = sum(1 for word in FOOD_CONTEXT_WORDS if word in text_lower)
    word_count = len(text_lower.split())
    # If food words make up significant portion, likely food context
    return food_count >= 2 and food_count / max(word_count, 1) > 0.15


def _check_neutral_context(text: str) -> bool:
    """Check if text is in a neutral/educational context."""
    text_lower = text.lower()
    neutral_count = sum(1 for word in NEUTRAL_DISCUSSION_WORDS if word in text_lower)
    return neutral_count >= 2


def _find_group_references(text: str) -> List[str]:
    """Find all group references in text."""
    text_lower = _normalize_text(text)
    found_groups = []

    all_groups = {
        "racial": ETHNIC_GROUPS,
        "religious": RELIGIOUS_GROUPS,
        "xenophobic": FOREIGN_GROUPS,
        "sexist": GENDER_TERMS,
        "homophobic": LGBTQ_TERMS,
    }

    for category, groups in all_groups.items():
        for group_name, terms in groups.items():
            for term in terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_groups.append({
                        "category": category,
                        "group": group_name,
                        "term": term,
                    })
                    break  # One match per group is enough

    return found_groups


def _check_hate_patterns(text: str) -> Dict[str, Any]:
    """Check for hate speech patterns in text."""
    text_lower = _normalize_text(text)
    # Also check leetspeak-normalized version
    text_deleet = _normalize_leetspeak(text_lower)

    matches = {
        "dehumanizing": [],
        "stereotyping": [],
        "threat": [],
        "supremacist": [],
    }

    for pattern in DEHUMANIZING_PATTERNS:
        for t in [text_lower, text_deleet]:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                matches["dehumanizing"].append(m.group())
                break

    for pattern in STEREOTYPING_PATTERNS:
        for t in [text_lower, text_deleet]:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                matches["stereotyping"].append(m.group())
                break

    for pattern in THREAT_PATTERNS:
        for t in [text_lower, text_deleet]:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                matches["threat"].append(m.group())
                break

    for pattern in SUPREMACIST_PATTERNS:
        for t in [text_lower, text_deleet]:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                matches["supremacist"].append(m.group())
                break

    return matches


def _determine_severity(hate_patterns: Any) -> str:
    """Determine severity based on pattern types found."""
    if hate_patterns["threat"]:
        return "high"
    if hate_patterns["dehumanizing"]:
        dehumanizing_count = len(hate_patterns["dehumanizing"])
        if dehumanizing_count >= 2:
            return "high"
        return "medium"
    if hate_patterns["supremacist"]:
        return "medium"
    if hate_patterns["stereotyping"]:
        return "low"
    return "none"


def _compute_confidence(group_refs: Any, hate_patterns: Any, is_food_ctx: Any, is_neutral_ctx: Any) -> float:
    """Compute confidence score for hate speech detection."""
    if not group_refs:
        return 0.1

    # Base confidence from having group reference + hate pattern
    total_patterns = sum(len(v) for v in hate_patterns.values())

    if total_patterns == 0:
        return 0.15  # Group reference alone is not hate speech

    # More patterns = higher confidence
    confidence = min(0.5 + (total_patterns * 0.15), 0.95)

    # Reduce confidence for food/neutral contexts
    if is_food_ctx:
        confidence *= 0.3
    if is_neutral_ctx:
        confidence *= 0.5

    return round(confidence, 2)


def _generate_explanation(category: Any, severity: Any, hate_patterns: Any, group_refs: Any) -> str:
    """Generate a human-readable explanation."""
    if severity == "none":
        return "No hate speech patterns detected."

    parts = []

    if hate_patterns["threat"]:
        parts.append("Contains threatening language")
    if hate_patterns["dehumanizing"]:
        parts.append("Contains dehumanizing language")
    if hate_patterns["supremacist"]:
        parts.append("Contains supremacist rhetoric")
    if hate_patterns["stereotyping"]:
        parts.append("Contains stereotyping language")

    if group_refs:
        targets = list(set(g["group"] for g in group_refs))
        parts.append(f"targeting: {', '.join(targets)}")

    return ". ".join(parts) + "."


def detect_hate_speech(text: str) -> Dict[str, Any]:
    """
    Detect hate speech in text with Malaysian context awareness.

    Args:
        text (str): Input text to analyze.

    Returns:
        dict: {
            "is_hate": bool,            # Whether hate speech is detected
            "category": str,            # racial|religious|sexist|xenophobic|homophobic|none
            "confidence": float,        # Confidence score 0-1
            "severity": str,            # none|low|medium|high
            "target_group": str|None,   # Targeted group if detected
            "explanation": str,         # Human-readable explanation
        }
    """
    if not text or not text.strip():
        return {
            "is_hate": False,
            "category": "none",
            "confidence": 0.0,
            "severity": "none",
            "target_group": None,
            "explanation": "Empty text.",
        }

    # Check contexts for false positive prevention
    is_food_ctx = _check_food_context(text)
    is_neutral_ctx = _check_neutral_context(text)

    # Find group references
    group_refs = _find_group_references(text)

    # Check hate patterns
    hate_patterns = _check_hate_patterns(text)

    # Determine severity
    severity = _determine_severity(hate_patterns)

    # If no hate patterns found, not hate speech
    total_patterns = sum(len(v) for v in hate_patterns.values())

    if total_patterns == 0 or not group_refs:
        # Special case: threats without explicit group reference
        # can still be hate speech if context implies it
        if hate_patterns["threat"] and not is_food_ctx:
            # Check if dehumanizing terms are present even without group ref
            if hate_patterns["dehumanizing"]:
                confidence = 0.5
                return {
                    "is_hate": True,
                    "category": "none",
                    "confidence": confidence,
                    "severity": severity,
                    "target_group": None,
                    "explanation": "Contains threatening and dehumanizing language.",
                }

        return {
            "is_hate": False,
            "category": "none",
            "confidence": round(1.0 - (total_patterns * 0.1), 2),
            "severity": "none",
            "target_group": None,
            "explanation": "No hate speech patterns detected.",
        }

    # Compute confidence
    confidence = _compute_confidence(group_refs, hate_patterns, is_food_ctx, is_neutral_ctx)

    # Determine primary category from group references
    category_priority = ["racial", "religious", "xenophobic", "sexist", "homophobic"]
    primary_category = "none"
    target_group = None

    for cat in category_priority:
        cat_refs = [g for g in group_refs if g["category"] == cat]
        if cat_refs:
            primary_category = cat
            target_group = cat_refs[0]["group"]
            break

    # Apply food context override
    if is_food_ctx and confidence < 0.5:
        return {
            "is_hate": False,
            "category": "none",
            "confidence": 0.8,
            "severity": "none",
            "target_group": None,
            "explanation": "Food context detected. Terms used in culinary context.",
        }

    # Final determination
    is_hate = confidence >= 0.4 and severity != "none"

    explanation = _generate_explanation(primary_category, severity, hate_patterns, group_refs)

    return {
        "is_hate": is_hate,
        "category": primary_category if is_hate else "none",
        "confidence": confidence,
        "severity": severity if is_hate else "none",
        "target_group": target_group if is_hate else None,
        "explanation": explanation,
    }


def detect_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Detect hate speech in a batch of texts.

    Args:
        texts (list[str]): List of texts to analyze.

    Returns:
        list[dict]: List of detection results.
    """
    if not texts:
        return []

    return [detect_hate_speech(text) for text in texts]


def is_hate_speech(text: str) -> bool:
    """
    Quick boolean check for hate speech.

    Args:
        text (str): Input text to check.

    Returns:
        bool: True if hate speech is detected.
    """
    result = detect_hate_speech(text)
    return result["is_hate"]


def get_severity(text: str) -> Dict[str, Any]:
    """
    Get the severity level of hate speech in text.

    Args:
        text (str): Input text to analyze.

    Returns:
        str: "none" | "low" | "medium" | "high"
    """
    result = detect_hate_speech(text)
    return result["severity"]


def get_target_groups(text: str) -> Dict[str, Any]:
    """
    Get all groups targeted in the text.

    Args:
        text (str): Input text to analyze.

    Returns:
        list[str]: List of targeted group names.
    """
    if not text or not text.strip():
        return []

    group_refs = _find_group_references(text)
    hate_patterns = _check_hate_patterns(text)

    total_patterns = sum(len(v) for v in hate_patterns.values())

    if total_patterns == 0:
        return []

    # Only return groups if hate patterns are present
    return list(set(g["group"] for g in group_refs))


# Module constants
CATEGORIES = ["racial", "religious", "sexist", "xenophobic", "homophobic", "none"]
SEVERITY_LEVELS = ["none", "low", "medium", "high"]
