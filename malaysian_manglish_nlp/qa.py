"""Extractive Question Answering for Manglish text.

Given a context paragraph and a question, extract the answer span from the context.
Handles Manglish question words (sape, ape, bile, mane, nape, camne) and
code-switched BM/EN text.

Zero external dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List

import re
import string

# Pre-compiled regex patterns
_RE_TOKENS = re.compile(r"[\w']+|[^\w\s]")
_RE_WORD_TOKENS = re.compile(r"[\w']+")
_RE_SENTENCE_SPLIT = re.compile(r'[.!?]+\s*|\n+')

# ─── Manglish shortforms for normalization ───────────────────────────────────

SHORTFORMS = {
    # Question words
    "sape": "siapa", "sapa": "siapa", "sp": "siapa",
    "ape": "apa", "ap": "apa", "pe": "apa",
    "bile": "bila", "bl": "bila",
    "mane": "mana", "mn": "mana", "mne": "mana",
    "nape": "kenapa", "knp": "kenapa", "knpe": "kenapa", "apasal": "kenapa",
    "camne": "macam mana", "cmne": "macam mana", "mcm mane": "macam mana",
    "mcm": "macam", "cmne": "macam mana",
    # Common shortforms
    "nk": "nak", "nak": "nak", "x": "tidak", "tak": "tidak", "tk": "tidak",
    "xde": "tiada", "takde": "tiada", "tade": "tiada",
    "dh": "dah", "sdh": "sudah", "dah": "sudah",
    "blh": "boleh", "bole": "boleh", "bleh": "boleh",
    "sbb": "sebab", "sb": "sebab", "psl": "pasal",
    "utk": "untuk", "tuk": "untuk", "tok": "untuk",
    "dgn": "dengan", "ngan": "dengan", "ngn": "dengan",
    "yg": "yang", "y": "yang",
    "ni": "ini", "tu": "itu",
    "kt": "kat", "kat": "di", "dkt": "dekat",
    "org": "orang", "owg": "orang",
    "brp": "berapa", "bpe": "berapa", "bape": "berapa",
    "dlm": "dalam", "lam": "dalam",
    "skrg": "sekarang", "skang": "sekarang",
    "smpi": "sampai", "smpai": "sampai",
    "pmpuan": "perempuan", "ppuan": "perempuan",
    "lki": "lelaki", "llaki": "lelaki",
    "gk": "gak", "je": "sahaja", "ja": "sahaja",
    "lg": "lagi", "lgi": "lagi",
    "dorg": "diorang", "diorg": "diorang", "dorang": "diorang",
    "korg": "korang", "korng": "korang",
    "aku": "saya", "ko": "kau", "kau": "kamu",
}

# Synonyms for matching
SYNONYMS = {
    "siapa": ["who", "siapa", "sape", "sapa"],
    "apa": ["what", "apa", "ape", "apakah"],
    "bila": ["when", "bila", "bile", "bilakah", "kapan"],
    "mana": ["where", "mana", "mane", "di mana"],
    "kenapa": ["why", "kenapa", "nape", "mengapa", "apasal"],
    "macam mana": ["how", "macam mana", "camne", "bagaimana"],
    "besar": ["big", "besar", "large", "huge"],
    "kecil": ["small", "kecil", "little", "tiny"],
    "baik": ["good", "baik", "bagus", "best"],
    "buruk": ["bad", "buruk", "teruk", "worst"],
    "banyak": ["many", "much", "banyak", "ramai"],
    "sedikit": ["few", "little", "sedikit", "sikit"],
    "cepat": ["fast", "quick", "cepat", "laju"],
    "lambat": ["slow", "lambat", "late"],
    "pergi": ["go", "pergi", "pegi", "gi"],
    "datang": ["come", "datang", "dtg"],
    "makan": ["eat", "makan", "mkn"],
    "kerja": ["work", "kerja", "keje", "keja"],
    "rumah": ["house", "home", "rumah", "umah"],
    "sekolah": ["school", "sekolah", "skolah"],
    "universiti": ["university", "universiti", "uni"],
}

# Question type patterns
QUESTION_PATTERNS = {
    "who": [
        r"\bwho\b", r"\bwhom\b", r"\bsiapa\b", r"\bsape\b", r"\bsapa\b",
        r"\bsp\b",
    ],
    "what": [
        r"\bwhat\b", r"\bapa\b", r"\bape\b", r"\bapakah\b",
    ],
    "when": [
        r"\bwhen\b", r"\bbila\b", r"\bbile\b", r"\bbilakah\b", r"\bkapan\b",
    ],
    "where": [
        r"\bwhere\b", r"\bdi mana\b", r"\bkat mana\b",
        r"\bdekat mana\b", r"(?<!macam )\bmana\b(?! nak)", r"\bmane\b(?! nak)",
    ],
    "why": [
        r"\bwhy\b", r"\bkenapa\b", r"\bnape\b", r"\bmengapa\b", r"\bapasal\b",
        r"\bknp\b",
    ],
    "how": [
        r"\bhow\b", r"\bmacam mana\b", r"\bcamne\b", r"\bcmne\b",
        r"\bbagaimana\b", r"\bmcm mane\b",
    ],
    "yes_no": [
        r"^(adakah|is|are|was|were|do|does|did|can|could|will|would|shall|should|have|has|had|boleh|betul|betulkah|ada|sudah|dah)\b",
        r"\bke\??\s*$", r"\btidak\??\s*$", r"\btak\??\s*$",
    ],
}

# Patterns for answer extraction by type
TIME_PATTERNS = [
    r"\b\d{1,2}[:.]\d{2}\s*(am|pm|pagi|petang|malam|tengahari)?\b",
    r"\b\d{1,2}\s*(am|pm|pagi|petang|malam|tengahari)\b",
    r"\bpukul\s+\d{1,2}([:.]\d{2})?\b",
    r"\b(pagi|petang|malam|tengahari|siang)\b",
]

DATE_PATTERNS = [
    r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
    r"\b\d{1,2}\s+(jan|feb|mac|apr|mei|jun|jul|ogos|sep|okt|nov|dis|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4}\b",
    r"\b(jan|feb|mac|apr|mei|jun|jul|ogos|sep|okt|nov|dis|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{2,4}\b",
    r"\b(isnin|selasa|rabu|khamis|jumaat|sabtu|ahad|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(hari ini|semalam|esok|kelmarin|today|yesterday|tomorrow)\b",
    r"\btahun\s+\d{4}\b",
    r"\b\d{4}\b",
]

LOCATION_PATTERNS = [
    r"\b(di|kat|dekat|dkt|kt|dalam|at|in|near)\s+[\w\s]{2,30}",
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalized names
]

PERSON_PATTERNS = [
    r"\b[A-Z][a-z]+(?:\s+(?:bin|binti|b\.|bt\.)\s+[A-Z][a-z]+)?\b",
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
    r"\b(dia|beliau|mereka|he|she|they|him|her|them)\b",
]

REASON_PATTERNS = [
    r"\b(sebab|kerana|because|since|pasal|sbb|psl)\s+.+",
    r"\b(supaya|agar|so that|untuk|in order to)\s+.+",
]

METHOD_PATTERNS = [
    r"\b(dengan|guna|using|by|through|via|pakai)\s+.+",
    r"\b(cara|method|way|kaedah)\s+.+",
]


def _normalize_text(text: str) -> str:
    """Normalize shortforms in text for better matching."""
    tokens = _RE_TOKENS.findall(text.lower())
    result = []
    for token in tokens:
        if token in SHORTFORMS:
            result.append(SHORTFORMS[token])
        else:
            result.append(token)
    return " ".join(result)


def _tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    return _RE_WORD_TOKENS.findall(text.lower())


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = _RE_SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word."""
    word_lower = word.lower()
    synonyms = {word_lower}
    for key, values in SYNONYMS.items():
        if word_lower in values or word_lower == key:
            synonyms.update(values)
            synonyms.add(key)
    return synonyms


def _word_overlap_score(text1: str, text2: str) -> float:
    """Calculate word overlap score between two texts with synonym expansion."""
    words1 = set(_tokenize(text1))
    words2 = set(_tokenize(text2))

    # Remove common stopwords
    stopwords = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "adalah", "untuk",
        "dengan", "pada", "the", "a", "an", "is", "are", "was", "were",
        "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
        "it", "this", "that", "be", "have", "has", "had", "do", "does",
        "did", "will", "would", "can", "could", "shall", "should",
        "aku", "saya", "kau", "kamu", "dia", "mereka", "kita", "kami",
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    }

    # Remove question words from scoring
    question_words = {
        "siapa", "apa", "bila", "mana", "kenapa", "macam", "bagaimana",
        "who", "what", "when", "where", "why", "how", "which",
        "sape", "ape", "bile", "mane", "nape", "camne",
        "adakah", "apakah", "bilakah",
    }

    words1 = words1 - stopwords - question_words
    words2 = words2 - stopwords - question_words

    if not words1 or not words2:
        return 0.0

    # Expand with synonyms
    expanded1 = set()
    for w in words1:
        expanded1.update(_get_synonyms(w))

    expanded2 = set()
    for w in words2:
        expanded2.update(_get_synonyms(w))

    overlap = expanded1 & expanded2
    if not overlap:
        # Try normalized versions
        norm1 = set(_tokenize(_normalize_text(" ".join(words1))))
        norm2 = set(_tokenize(_normalize_text(" ".join(words2))))
        overlap = norm1 & norm2

    denominator = min(len(words1), len(words2))
    if denominator == 0:
        return 0.0

    return len(overlap) / denominator


import math

def _tfidf_score(query: str, sentences: List[str]) -> float:
    """Score sentences against a query using TF-IDF weighting.

    TF = term frequency in sentence (count / total words in sentence)
    IDF = log(total_sentences / sentences_containing_term)

    Args:
        query: The query text.
        sentences: List of candidate sentences.

    Returns:
        list[float]: TF-IDF similarity scores for each sentence.
    """
    if not sentences:
        return []

    stopwords = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "adalah", "untuk",
        "dengan", "pada", "the", "a", "an", "is", "are", "was", "were",
        "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
        "it", "this", "that", "be", "have", "has", "had", "do", "does",
        "did", "will", "would", "can", "could", "shall", "should",
        "aku", "saya", "kau", "kamu", "dia", "mereka", "kita", "kami",
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    }
    question_words = {
        "siapa", "apa", "bila", "mana", "kenapa", "macam", "bagaimana",
        "who", "what", "when", "where", "why", "how", "which",
        "sape", "ape", "bile", "mane", "nape", "camne",
        "adakah", "apakah", "bilakah",
    }

    # Tokenize all sentences
    sent_tokens_list = []
    for s in sentences:
        tokens = [t for t in _tokenize(s) if t not in stopwords and t not in question_words]
        sent_tokens_list.append(tokens)

    # Query tokens
    query_tokens = [t for t in _tokenize(_normalize_text(query)) if t not in stopwords and t not in question_words]
    if not query_tokens:
        return [0.0] * len(sentences)

    # Expand query tokens with synonyms
    expanded_query = set()
    for qt in query_tokens:
        expanded_query.update(_get_synonyms(qt))

    total_sentences = len(sentences)

    # Calculate IDF for each query term
    idf = {}
    for term in expanded_query:
        doc_count = sum(1 for tokens in sent_tokens_list if term in tokens)
        if doc_count > 0:
            idf[term] = math.log(total_sentences / doc_count)
        else:
            idf[term] = 0.0

    # Score each sentence
    scores = []
    for tokens in sent_tokens_list:
        if not tokens:
            scores.append(0.0)
            continue

        score = 0.0
        token_count = len(tokens)
        for term in expanded_query:
            tf = tokens.count(term) / token_count
            score += tf * idf.get(term, 0.0)

        scores.append(score)

    return scores


def classify_question_type(question: str) -> Dict[str, Any]:
    """Classify the type of question.

    Args:
        question: The question text.

    Returns:
        str: One of "who", "what", "when", "where", "why", "how", "yes_no", "other".

    Example:
        >>> classify_question_type("Sape yang buat ni?")
        'who'
        >>> classify_question_type("Bile dia sampai?")
        'when'
    """
    q_lower = question.lower().strip()
    q_normalized = _normalize_text(q_lower)

    # Check each type in priority order (how before where to avoid 'mana' in 'macam mana' matching where)
    for qtype in ["who", "how", "when", "why", "where", "what", "yes_no"]:
        patterns = QUESTION_PATTERNS[qtype]
        for pattern in patterns:
            if re.search(pattern, q_lower, re.IGNORECASE):
                return qtype
            if re.search(pattern, q_normalized, re.IGNORECASE):
                return qtype

    return "other"


def find_relevant_sentence(context: str, question: str) -> Dict[str, Any]:
    """Find the most relevant sentence in context for the given question.

    Uses TF-IDF scoring as primary method, with word overlap as fallback.

    Args:
        context: The context paragraph.
        question: The question to answer.

    Returns:
        str: The most relevant sentence from the context.

    Example:
        >>> ctx = "Ali kerja kat KL. Dia balik rumah pukul 6."
        >>> find_relevant_sentence(ctx, "Bile Ali balik?")
        'Dia balik rumah pukul 6'
    """
    sentences = _sentence_split(context)
    if not sentences:
        return ""

    if len(sentences) == 1:
        return sentences[0]

    # Primary: TF-IDF scoring
    tfidf_scores = _tfidf_score(question, sentences)

    # Find best TF-IDF score
    best_tfidf_idx = 0
    best_tfidf = tfidf_scores[0]
    for i, score in enumerate(tfidf_scores):
        if score > best_tfidf:
            best_tfidf = score
            best_tfidf_idx = i

    # If TF-IDF gives a clear winner (score > 0), use it
    if best_tfidf > 0:
        return sentences[best_tfidf_idx]

    # Fallback: word overlap scoring
    q_normalized = _normalize_text(question)

    best_score = -1
    best_sentence = sentences[0]

    for sentence in sentences:
        score1 = _word_overlap_score(sentence, question)
        score2 = _word_overlap_score(sentence, q_normalized)
        score3 = _word_overlap_score(_normalize_text(sentence), q_normalized)

        score = max(score1, score2, score3)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence


def extract_answer_span(sentence: str, question: str) -> Dict[str, Any]:
    """Extract the specific answer span from a sentence based on question type.

    Args:
        sentence: The sentence containing the answer.
        question: The question being answered.

    Returns:
        str: The extracted answer span.

    Example:
        >>> extract_answer_span("Ali pergi ke Kuala Lumpur", "Where did Ali go?")
        'Kuala Lumpur'
    """
    qtype = classify_question_type(question)
    sentence_lower = sentence.lower()

    if qtype == "when":
        return _extract_when(sentence)
    elif qtype == "where":
        return _extract_where(sentence, question)
    elif qtype == "who":
        return _extract_who(sentence, question)
    elif qtype == "why":
        return _extract_why(sentence)
    elif qtype == "how":
        return _extract_how(sentence)
    elif qtype == "what":
        return _extract_what(sentence, question)
    elif qtype == "yes_no":
        return _extract_yes_no(sentence, question)
    else:
        return sentence.strip()


def _extract_when(sentence: str) -> str:
    """Extract time/date answer from sentence."""
    # Try time patterns first
    for pattern in TIME_PATTERNS:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Try date patterns
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Fallback: look for numbers with context
    match = re.search(r'\b\d+\s*\w+', sentence)
    if match:
        return match.group(0).strip()

    return sentence.strip()


def _extract_where(sentence: str, question: str) -> str:
    """Extract location answer from sentence."""
    # Look for preposition + location
    prep_patterns = [
        r"(?:di|kat|dekat|dkt|kt|dalam|at|in|near|to|ke)\s+([\w\s]+?)(?:\.|,|$|\s+(?:dan|and|tapi|but|untuk|for))",
        r"(?:di|kat|dekat|dkt|kt|dalam|at|in|near|to|ke)\s+([\w\s]+)",
    ]

    for pattern in prep_patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Clean trailing punctuation and common words
            location = re.sub(r'\s*(dan|and|tapi|but|untuk|for|yang|the)\s*$', '', location)
            location = location.strip(string.punctuation + " ")
            if location and len(location) > 1:
                return location

    # Look for capitalized proper nouns (likely place names)
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
    # Filter out words that appear in the question
    q_words = set(_tokenize(question))
    caps = [c for c in caps if c.lower() not in q_words]
    if caps:
        return caps[0]

    return sentence.strip()


def _extract_who(sentence: str, question: str) -> str:
    """Extract person answer from sentence."""
    # Look for proper nouns (capitalized words)
    q_words = set(_tokenize(question))

    # Find capitalized names (possibly with bin/binti)
    name_pattern = r'\b([A-Z][a-z]+(?:\s+(?:bin|binti|b\.|bt\.)\s+[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)*)\b'
    names = re.findall(name_pattern, sentence)
    names = [n for n in names if n.lower() not in q_words and len(n) > 1]

    if names:
        return names[0]

    # Look for pronouns or role words
    role_patterns = [
        r'\b(cikgu|teacher|doctor|doktor|dr\.?\s*\w+|professor|prof\.?\s*\w+)\b',
        r'\b(boss|manager|ketua|pengarah|CEO|CTO)\b',
        r'\b(abang|kakak|adik|mak|ayah|bapa|ibu|anak|cucu)\b',
        r'\b(dia|beliau|mereka|he|she|they)\b',
    ]

    for pattern in role_patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Fallback: first noun-like word not in question
    tokens = sentence.split()
    for token in tokens:
        if token.lower() not in q_words and token[0:1].isupper():
            return token

    return sentence.strip()


def _extract_why(sentence: str) -> str:
    """Extract reason answer from sentence."""
    for pattern in REASON_PATTERNS:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Look for "sebab" or "because" clause
    parts = re.split(r'\b(sebab|kerana|because|since|pasal|sbb)\b', sentence, flags=re.IGNORECASE)
    if len(parts) > 2:
        reason = parts[-1].strip()
        reason = reason.strip(string.punctuation + " ")
        if reason:
            return reason

    return sentence.strip()


def _extract_how(sentence: str) -> str:
    """Extract method/manner answer from sentence."""
    for pattern in METHOD_PATTERNS:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Look for "dengan" or "guna" clause
    parts = re.split(r'\b(dengan|guna|using|by|through|via|pakai|cara)\b', sentence, flags=re.IGNORECASE)
    if len(parts) > 2:
        method = parts[-1].strip()
        method = method.strip(string.punctuation + " ")
        if method:
            connector = parts[-2]
            return (connector + " " + method).strip()

    return sentence.strip()


def _extract_what(sentence: str, question: str) -> str:
    """Extract noun phrase answer from sentence."""
    q_words = set(_tokenize(question))

    # Look for "adalah/ialah/is/are" pattern (definition)
    def_match = re.search(
        r'(?:adalah|ialah|is|are|was|were|merupakan)\s+(.+?)(?:\.|,|$)',
        sentence, re.IGNORECASE
    )
    if def_match:
        answer = def_match.group(1).strip()
        answer = answer.strip(string.punctuation + " ")
        if answer:
            return answer

    # Look for noun phrases not in question
    # Split by common verbs/prepositions and take the complement
    verb_split = re.split(
        r'\b(adalah|ialah|itu|ini|merupakan|called|named|dipanggil)\b',
        sentence, flags=re.IGNORECASE
    )
    if len(verb_split) > 2:
        answer = verb_split[-1].strip()
        answer = answer.strip(string.punctuation + " ")
        if answer and not all(w.lower() in q_words for w in answer.split()):
            return answer

    # Fallback: remove question words from sentence and return remainder
    sentence_tokens = sentence.split()
    answer_tokens = [t for t in sentence_tokens if t.lower() not in q_words]
    if answer_tokens:
        return " ".join(answer_tokens).strip(string.punctuation + " ")

    return sentence.strip()


def _extract_yes_no(sentence: str, question: str) -> str:
    """Determine yes/no answer based on context support."""
    q_normalized = _normalize_text(question)
    s_normalized = _normalize_text(sentence)

    # Check for negation in sentence
    negation_words = {"tidak", "tak", "bukan", "tiada", "belum", "jangan",
                      "not", "no", "never", "neither", "nor", "don't", "doesn't",
                      "didn't", "won't", "wouldn't", "can't", "couldn't", "xde", "x"}

    s_tokens = set(_tokenize(s_normalized))
    has_negation = bool(s_tokens & negation_words)

    # Check if question contains negation
    q_tokens = set(_tokenize(q_normalized))
    q_has_negation = bool(q_tokens & negation_words)

    # Calculate overlap to determine support
    overlap = _word_overlap_score(s_normalized, q_normalized)

    if overlap > 0.3:
        if has_negation and not q_has_negation:
            return "Tidak"
        elif not has_negation:
            return "Ya"
        else:
            return "Ya"
    else:
        return "Tidak pasti"


def _calculate_confidence(context: str, question: str, sentence: str, answer: str, qtype: Any) -> float:
    """Calculate confidence score for an answer."""
    confidence = 0.0

    # Factor 1: Sentence relevance (overlap score)
    overlap = _word_overlap_score(sentence, question)
    norm_overlap = _word_overlap_score(
        _normalize_text(sentence), _normalize_text(question)
    )
    relevance = max(overlap, norm_overlap)
    confidence += relevance * 0.4

    # Factor 2: Answer type match
    type_match = 0.0
    if qtype == "when":
        if re.search(r'\d', answer):
            type_match = 0.8
        elif re.search(r'(pagi|petang|malam|hari|bulan|tahun|morning|evening|night|day|month|year)', answer, re.IGNORECASE):
            type_match = 0.6
    elif qtype == "where":
        if re.search(r'(di|kat|dekat|at|in)\s', sentence, re.IGNORECASE):
            type_match = 0.7
        if answer[0:1].isupper():
            type_match = max(type_match, 0.6)
    elif qtype == "who":
        if answer[0:1].isupper():
            type_match = 0.8
        elif re.search(r'(dia|beliau|he|she)', answer, re.IGNORECASE):
            type_match = 0.5
    elif qtype == "why":
        if re.search(r'(sebab|kerana|because|since|pasal)', answer, re.IGNORECASE):
            type_match = 0.8
        elif re.search(r'(sebab|kerana|because|since|pasal)', sentence, re.IGNORECASE):
            type_match = 0.6
    elif qtype == "how":
        if re.search(r'(dengan|guna|using|by|cara)', answer, re.IGNORECASE):
            type_match = 0.7
    elif qtype == "what":
        if answer and answer != sentence:
            type_match = 0.5
    elif qtype == "yes_no":
        if answer in ("Ya", "Tidak"):
            type_match = 0.7
        else:
            type_match = 0.3

    confidence += type_match * 0.35

    # Factor 3: Answer quality
    if answer and answer != sentence and len(answer) > 1:
        # Shorter, more specific answers are better
        ratio = len(answer) / max(len(sentence), 1)
        if 0.05 < ratio < 0.7:
            confidence += 0.25
        elif ratio <= 0.05:
            confidence += 0.1
        else:
            confidence += 0.1
    elif answer == sentence:
        confidence += 0.05

    return min(confidence, 1.0)


def _find_span_position(context: str, answer: str) -> Dict[str, Any]:
    """Find the start and end position of the answer in the context."""
    # Try exact match first
    idx = context.find(answer)
    if idx >= 0:
        return idx, idx + len(answer)

    # Try case-insensitive
    idx = context.lower().find(answer.lower())
    if idx >= 0:
        return idx, idx + len(answer)

    # Try finding individual words
    answer_words = answer.split()
    if answer_words:
        first_word = answer_words[0]
        idx = context.lower().find(first_word.lower())
        if idx >= 0:
            return idx, idx + len(answer)

    return 0, len(answer)


def answer(context: str, question: str) -> Dict[str, Any]:
    """Answer a question based on the given context.

    Performs extractive QA by finding the most relevant sentence and
    extracting the appropriate answer span based on question type.

    Args:
        context: The context paragraph to search for answers.
        question: The question to answer.

    Returns:
        dict: A dictionary with keys:
            - answer (str): The extracted answer text.
            - confidence (float): Confidence score between 0.0 and 1.0.
            - start (int): Start character position in context.
            - end (int): End character position in context.
            - sentence (str): The sentence containing the answer.

    Example:
        >>> ctx = "Ahmad kerja kat Petronas. Dia start tahun 2020."
        >>> answer(ctx, "Sape kerja kat Petronas?")
        {'answer': 'Ahmad', 'confidence': ..., 'start': 0, 'end': 5, 'sentence': 'Ahmad kerja kat Petronas'}
    """
    if not context or not question:
        return {
            "answer": "",
            "confidence": 0.0,
            "start": 0,
            "end": 0,
            "sentence": "",
        }

    # Classify question type
    qtype = classify_question_type(question)

    # Find most relevant sentence
    sentence = find_relevant_sentence(context, question)

    # Extract answer span
    ans = extract_answer_span(sentence, question)

    # Calculate confidence
    confidence = _calculate_confidence(context, question, sentence, ans, qtype)

    # Find position in original context
    start, end = _find_span_position(context, ans)

    return {
        "answer": ans,
        "confidence": round(confidence, 4),
        "start": start,
        "end": end,
        "sentence": sentence,
    }


def answer_multiple(context: str, questions: str) -> List[Dict[str, Any]]:
    """Answer multiple questions based on the same context.

    Args:
        context: The context paragraph.
        questions: List of question strings.

    Returns:
        list: List of answer dictionaries (same format as answer()).

    Example:
        >>> ctx = "Ali tinggal kat Shah Alam. Dia kerja sebagai engineer."
        >>> answer_multiple(ctx, ["Mane Ali tinggal?", "Ape kerja dia?"])
        [{'answer': 'Shah Alam', ...}, {'answer': 'engineer', ...}]
    """
    return [answer(context, q) for q in questions]
