"""
Discourse analysis and argument mining for Manglish text.

Provides tools for analyzing discourse structure, extracting arguments,
detecting discourse markers, and identifying logical fallacies in
Malaysian Manglish (BM + English code-switched) text.

Zero external dependencies.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional


# ============================================================
# Discourse Markers Database
# ============================================================

DISCOURSE_MARKERS = {
    "causal": [
        "sebab", "because", "sbb", "pasal", "kerana", "that's why",
        "tu la", "tu lah", "sebab tu", "because of", "disebabkan",
        "oleh kerana", "since", "due to", "so that", "supaya",
        "maka", "akibat", "punca", "caused by",
    ],
    "contrast": [
        "tapi", "but", "however", "walau", "walaupun", "padahal",
        "sebaliknya", "nevertheless", "although", "even though",
        "walaubagaimanapun", "yet", "still", "nonetheless",
        "on the other hand", "sebalik", "malah", "tetapi",
    ],
    "addition": [
        "dan", "and", "lagi", "tambahan", "plus", "also", "selain tu",
        "furthermore", "moreover", "besides", "in addition",
        "tambah lagi", "lagi pun", "lagipun", "serta", "juga",
        "bukan tu je", "not only", "on top of that",
    ],
    "temporal": [
        "lepas tu", "then", "sebelum", "after", "dulu", "kemudian",
        "first", "before", "next", "finally", "meanwhile",
        "sementara", "masa tu", "waktu tu", "selepas", "sebelum tu",
        "after that", "before that", "at first", "mula-mula",
        "akhirnya", "last time",
    ],
    "conclusion": [
        "so", "jadi", "kesimpulan", "therefore", "akhirnya",
        "in conclusion", "hence", "thus", "oleh itu", "maka",
        "kesimpulannya", "to sum up", "in short", "pendek kata",
        "ringkasnya", "all in all", "at the end of the day",
        "conclusion dia", "so basically",
    ],
}

# Flatten for quick lookup
_ALL_MARKERS = {}
for marker_type, markers in DISCOURSE_MARKERS.items():
    for marker in markers:
        _ALL_MARKERS[marker.lower()] = marker_type


# ============================================================
# Opinion / Claim Indicators
# ============================================================

OPINION_MARKERS = [
    "i think", "aku rasa", "pada aku", "in my opinion", "imo",
    "i believe", "aku percaya", "memang", "confirm", "obviously",
    "clearly", "definitely", "for sure", "mesti", "patut",
    "sepatutnya", "should", "must", "kena", "wajib",
    "tak patut", "shouldn't", "cannot", "tak boleh",
    "best", "worst", "better", "worse", "paling",
]

EVIDENCE_INDICATORS = [
    "contoh", "example", "for instance", "macam", "like",
    "according to", "data shows", "statistics", "research",
    "study", "kajian", "bukti", "proof", "evidence",
    "based on", "berdasarkan", "percent", "peratus", "%",
    "survey", "report", "laporan",
]

REBUTTAL_INDICATORS = [
    "tapi sebenarnya", "but actually", "however", "on the contrary",
    "that's not true", "tak betul", "salah tu", "wrong",
    "sebenarnya", "actually", "in fact", "the truth is",
    "hakikatnya", "contrary to", "tak setuju", "disagree",
]

# ============================================================
# Fallacy Patterns
# ============================================================

FALLACY_PATTERNS = {
    "ad_hominem": {
        "description": "Attacking the person instead of the argument",
        "patterns": [
            r"kau\s+(?:bodoh|stupid|idiot|bangang|bodo|bengap)",
            r"(?:you|kau|dia|he|she)\s+(?:don'?t|tak|x)\s+(?:know|tau|tahu)",
            r"(?:budak|orang)\s+(?:macam|like)\s+(?:kau|you)",
            r"what\s+(?:do|would)\s+(?:you|kau)\s+know",
            r"(?:kau|you)\s+(?:mana|where)\s+(?:tau|know)",
        ],
    },
    "appeal_to_authority": {
        "description": "Using authority as sole evidence without substance",
        "patterns": [
            r"(?:expert|pakar|professor|doctor|dr)\s+(?:said|cakap|kata)",
            r"(?:government|kerajaan)\s+(?:said|cakap|kata|confirm)",
            r"(?:famous|terkenal)\s+(?:person|orang)\s+(?:said|cakap)",
            r"(?:everyone|semua orang)\s+(?:knows?|tau|tahu)",
            r"(?:they|diorang)\s+(?:all|semua)\s+(?:say|cakap|kata)",
        ],
    },
    "strawman": {
        "description": "Misrepresenting someone's argument to attack it",
        "patterns": [
            r"so\s+(?:you're|kau)\s+saying",
            r"(?:you|kau)\s+(?:mean|maksud)\s+(?:that|tu)",
            r"oh\s+so\s+(?:now|sekarang)",
            r"(?:that's|tu)\s+(?:like|macam)\s+saying",
        ],
    },
    "false_dichotomy": {
        "description": "Presenting only two options when more exist",
        "patterns": [
            r"(?:either|sama ada).*(?:or|atau)",
            r"(?:only|hanya)\s+(?:two|dua)\s+(?:options?|pilihan)",
            r"(?:if not|kalau tak).*(?:then|maka|mesti)",
            r"(?:you|kau)\s+(?:either|sama ada)",
            r"(?:there's|ada)\s+(?:no|takde)\s+(?:other|lain)\s+(?:way|cara)",
        ],
    },
    "slippery_slope": {
        "description": "Assuming one event will lead to extreme consequences",
        "patterns": [
            r"(?:next thing|lepas ni)\s+(?:you know|tau)",
            r"(?:will|akan)\s+(?:lead to|bawa kepada)",
            r"(?:before you know it|tak lama lagi)",
            r"(?:eventually|akhirnya).*(?:everything|semua)",
            r"(?:if|kalau).*(?:then|lepas tu).*(?:then|lepas tu)",
        ],
    },
    "appeal_to_emotion": {
        "description": "Using emotional manipulation instead of logic",
        "patterns": [
            r"(?:think of|fikir pasal)\s+(?:the children|anak-anak|kids)",
            r"(?:how would you feel|macam mana rasa)",
            r"(?:kesian|pity|poor)\s+(?:them|diorang|dia)",
            r"(?:imagine|bayangkan)\s+(?:if|kalau)",
        ],
    },
}


# ============================================================
# Helper Functions
# ============================================================

def _normalize_text(text: str) -> str:
    """Normalize text for processing."""
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on common sentence boundaries
    parts = re.split(r'[.!?]+\s*|\n+', text)
    # Also split on comma + discourse marker patterns
    result = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result if result else [text.strip()]


def _contains_any(text: str, indicators: List[str]) -> bool:
    """Check if text contains any of the given indicators."""
    text_lower = text.lower()
    for indicator in indicators:
        if indicator.lower() in text_lower:
            return True
    return False


def _has_numbers(text: str) -> bool:
    """Check if text contains numbers (potential evidence)."""
    return bool(re.search(r'\d+', text))


def _classify_segment_role(segment: str, prev_segments: List[Dict], all_text: str) -> str:
    """Classify a segment's discourse role."""
    seg_lower = segment.lower().strip()

    # Check for rebuttal (contrast after a claim)
    if _contains_any(seg_lower, REBUTTAL_INDICATORS):
        if any(s.get("role") == "claim" for s in prev_segments):
            return "rebuttal"

    # Check for contrast markers at start
    for marker in DISCOURSE_MARKERS["contrast"]:
        if seg_lower.startswith(marker.lower()):
            if any(s.get("role") == "claim" for s in prev_segments):
                return "rebuttal"

    # Check for conclusion markers
    for marker in DISCOURSE_MARKERS["conclusion"]:
        if seg_lower.startswith(marker.lower()):
            return "conclusion"

    # Check for evidence indicators
    if _contains_any(seg_lower, EVIDENCE_INDICATORS) or _has_numbers(seg_lower):
        if prev_segments:
            return "evidence"

    # Check for causal markers (often introduces evidence)
    for marker in DISCOURSE_MARKERS["causal"]:
        if seg_lower.startswith(marker.lower()):
            return "evidence"

    # Check for opinion/claim markers
    if _contains_any(seg_lower, OPINION_MARKERS):
        return "claim"

    # Default: if it's the first segment or has assertion-like structure
    if not prev_segments:
        return "claim" if len(segment.split()) > 3 else "background"

    # If previous was a claim and this follows, likely evidence or background
    if prev_segments and prev_segments[-1].get("role") == "claim":
        if len(segment.split()) > 5:
            return "evidence"

    return "background"


def _calculate_confidence(segment: str, role: str) -> float:
    """Calculate confidence for a role assignment."""
    seg_lower = segment.lower()
    confidence = 0.5  # base

    if role == "claim":
        for marker in OPINION_MARKERS:
            if marker.lower() in seg_lower:
                confidence += 0.15
                break
        if len(segment.split()) > 5:
            confidence += 0.1

    elif role == "evidence":
        for indicator in EVIDENCE_INDICATORS:
            if indicator.lower() in seg_lower:
                confidence += 0.15
                break
        if _has_numbers(seg_lower):
            confidence += 0.15

    elif role == "rebuttal":
        for indicator in REBUTTAL_INDICATORS:
            if indicator.lower() in seg_lower:
                confidence += 0.2
                break

    elif role == "conclusion":
        for marker in DISCOURSE_MARKERS["conclusion"]:
            if seg_lower.startswith(marker.lower()):
                confidence += 0.25
                break

    elif role == "background":
        confidence = 0.4

    return min(confidence, 1.0)


# ============================================================
# Public API
# ============================================================

def analyze_discourse(text: str) -> Dict:
    """
    Analyze the discourse structure of text.

    Args:
        text: Input text (Manglish/BM/EN)

    Returns:
        dict with:
            - structure: list of segments with roles and confidence
            - coherence_score: float (0-1) indicating logical flow quality
    """
    if not text or not text.strip():
        return {"structure": [], "coherence_score": 0.0}

    text = _normalize_text(text)
    sentences = _split_sentences(text)

    structure = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        role = _classify_segment_role(sentence, structure, text)
        confidence = _calculate_confidence(sentence, role)
        structure.append({
            "segment": sentence.strip(),
            "role": role,
            "confidence": round(confidence, 2),
        })

    coherence = _calculate_coherence(text, structure)

    return {
        "structure": structure,
        "coherence_score": round(coherence, 2),
    }


def _calculate_coherence(text: str, structure: List[Dict]) -> float:
    """Calculate coherence score based on discourse structure."""
    if not structure:
        return 0.0

    score = 0.3  # base score for having any structure

    # Bonus for having discourse markers
    markers_found = detect_discourse_markers(text)
    if markers_found:
        score += min(0.2, len(markers_found) * 0.05)

    # Bonus for logical flow (claim → evidence → conclusion)
    roles = [s["role"] for s in structure]

    # Has both claims and evidence
    if "claim" in roles and "evidence" in roles:
        score += 0.2

    # Has conclusion
    if "conclusion" in roles:
        score += 0.15

    # Penalty for all same role
    unique_roles = set(roles)
    if len(unique_roles) == 1 and len(roles) > 2:
        score -= 0.1

    # Bonus for variety
    if len(unique_roles) >= 3:
        score += 0.1

    # Bonus for proper ordering (claim before evidence)
    claim_idx = next((i for i, r in enumerate(roles) if r == "claim"), -1)
    evidence_idx = next((i for i, r in enumerate(roles) if r == "evidence"), -1)
    if claim_idx >= 0 and evidence_idx > claim_idx:
        score += 0.1

    return max(0.0, min(1.0, score))


def extract_arguments(text: str) -> List[Dict]:
    """
    Extract arguments from text.

    Args:
        text: Input text (Manglish/BM/EN)

    Returns:
        List of argument dicts with:
            - claim: the main claim string
            - evidence: list of supporting evidence strings
            - stance: "for" or "against"
    """
    if not text or not text.strip():
        return []

    text = _normalize_text(text)
    sentences = _split_sentences(text)

    if not sentences:
        return []

    arguments = []
    current_claim = None
    current_evidence = []
    current_stance = "for"

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        seg_lower = sentence.lower()

        # Detect if this is a new claim
        is_claim = _contains_any(seg_lower, OPINION_MARKERS)

        # Detect stance
        negative_indicators = [
            "tak patut", "shouldn't", "tak boleh", "cannot", "wrong",
            "salah", "bad", "buruk", "worst", "terrible", "tak setuju",
            "disagree", "oppose", "bantah", "against",
        ]
        positive_indicators = [
            "patut", "should", "boleh", "can", "good", "bagus",
            "best", "great", "agree", "setuju", "support", "sokong",
        ]

        if is_claim or (not current_claim and len(sentence.split()) > 3):
            # Save previous argument if exists
            if current_claim:
                arguments.append({
                    "claim": current_claim,
                    "evidence": current_evidence,
                    "stance": current_stance,
                })

            current_claim = sentence
            current_evidence = []

            # Determine stance
            if _contains_any(seg_lower, negative_indicators):
                current_stance = "against"
            elif _contains_any(seg_lower, positive_indicators):
                current_stance = "for"
            else:
                current_stance = "for"

        elif current_claim:
            # This is evidence for the current claim
            if _contains_any(seg_lower, EVIDENCE_INDICATORS) or _has_numbers(seg_lower):
                current_evidence.append(sentence)
            elif any(seg_lower.startswith(m.lower()) for m in DISCOURSE_MARKERS["causal"]):
                current_evidence.append(sentence)
            elif len(sentence.split()) > 3:
                current_evidence.append(sentence)

    # Don't forget the last argument
    if current_claim:
        arguments.append({
            "claim": current_claim,
            "evidence": current_evidence,
            "stance": current_stance,
        })

    return arguments


def detect_discourse_markers(text: str) -> List[Dict]:
    """
    Detect discourse markers in text.

    Args:
        text: Input text (Manglish/BM/EN)

    Returns:
        List of dicts with:
            - marker: the detected marker string
            - type: "causal"|"contrast"|"addition"|"temporal"|"conclusion"
            - position: character position in text
    """
    if not text or not text.strip():
        return []

    text = _normalize_text(text)
    text_lower = text.lower()
    results = []
    found_positions = set()  # avoid overlapping detections

    # Sort markers by length (longest first) to prefer longer matches
    all_markers_sorted = sorted(_ALL_MARKERS.keys(), key=len, reverse=True)

    for marker in all_markers_sorted:
        # Use word boundary matching for short markers
        if len(marker) <= 3:
            pattern = r'\b' + re.escape(marker) + r'\b'
        else:
            pattern = re.escape(marker)

        for match in re.finditer(pattern, text_lower):
            pos = match.start()

            # Check for overlap with already found markers
            overlap = False
            for found_start, found_end in found_positions:
                if pos >= found_start and pos < found_end:
                    overlap = True
                    break
                if match.end() > found_start and match.end() <= found_end:
                    overlap = True
                    break

            if not overlap:
                found_positions.add((pos, match.end()))
                results.append({
                    "marker": text[pos:match.end()],
                    "type": _ALL_MARKERS[marker],
                    "position": pos,
                })

    # Sort by position
    results.sort(key=lambda x: x["position"])
    return results


def segment_discourse(text: str) -> List[Dict]:
    """
    Segment text into discourse units with roles.

    Args:
        text: Input text (Manglish/BM/EN)

    Returns:
        List of segment dicts with:
            - segment: text content
            - role: "claim"|"evidence"|"rebuttal"|"conclusion"|"background"
            - confidence: float (0-1)
    """
    if not text or not text.strip():
        return []

    result = analyze_discourse(text)
    return result["structure"]


def argument_strength(text: str) -> float:
    """
    Calculate how well-supported an argument is.

    Args:
        text: Input text (Manglish/BM/EN)

    Returns:
        Float 0-1 indicating argument strength.
        Higher = better supported with evidence and logical structure.
    """
    if not text or not text.strip():
        return 0.0

    text = _normalize_text(text)
    score = 0.0

    # Extract arguments
    args = extract_arguments(text)
    if not args:
        return 0.1

    # Has at least one claim
    score += 0.2

    # Evidence support
    total_evidence = sum(len(a["evidence"]) for a in args)
    if total_evidence > 0:
        score += min(0.3, total_evidence * 0.1)

    # Discourse markers present (shows logical connection)
    markers = detect_discourse_markers(text)
    if markers:
        marker_types = set(m["type"] for m in markers)
        score += min(0.2, len(marker_types) * 0.05)

        # Causal markers are especially good for arguments
        if any(m["type"] == "causal" for m in markers):
            score += 0.1

    # Has conclusion
    discourse = analyze_discourse(text)
    roles = [s["role"] for s in discourse["structure"]]
    if "conclusion" in roles:
        score += 0.1

    # Numbers/data (concrete evidence)
    if _has_numbers(text):
        score += 0.1

    # Penalty for very short text
    word_count = len(text.split())
    if word_count < 10:
        score *= 0.7
    elif word_count < 5:
        score *= 0.4

    # No fallacies is good
    fallacies = detect_fallacies(text)
    if fallacies:
        score -= len(fallacies) * 0.1

    return max(0.0, min(1.0, round(score, 2)))


def detect_fallacies(text: str) -> List[Dict]:
    """
    Detect potential logical fallacies in text.

    Args:
        text: Input text (Manglish/BM/EN)

    Returns:
        List of dicts with:
            - type: fallacy type identifier
            - description: human-readable description
            - evidence: the text that triggered detection
            - confidence: float (0-1)
    """
    if not text or not text.strip():
        return []

    text = _normalize_text(text)
    text_lower = text.lower()
    results = []

    for fallacy_type, fallacy_info in FALLACY_PATTERNS.items():
        for pattern in fallacy_info["patterns"]:
            match = re.search(pattern, text_lower)
            if match:
                # Calculate confidence based on match quality
                matched_text = text[match.start():match.end()]
                confidence = 0.6

                # Higher confidence for longer matches
                if len(matched_text.split()) >= 3:
                    confidence += 0.15
                # Higher confidence if surrounded by argument context
                if _contains_any(text_lower, OPINION_MARKERS):
                    confidence += 0.1

                results.append({
                    "type": fallacy_type,
                    "description": fallacy_info["description"],
                    "evidence": matched_text,
                    "confidence": round(min(confidence, 0.95), 2),
                })
                break  # One detection per fallacy type is enough

    return results
