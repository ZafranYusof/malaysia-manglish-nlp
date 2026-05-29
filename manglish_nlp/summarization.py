"""
manglish_nlp.summarization - Extractive summarization for Manglish text.

Uses TextRank algorithm to extract key sentences from text.
Handles code-switched (BM/EN) text by normalizing shortforms before
scoring but preserving original text in output.

Zero external dependencies - stdlib only.
"""

import re
import math
from collections import Counter


# Common Manglish shortforms for normalization during scoring
_SHORTFORMS = {
    'nk': 'nak', 'tk': 'tak', 'x': 'tak', 'xde': 'takde',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'dlm': 'dalam',
    'mcm': 'macam', 'sbb': 'sebab', 'lg': 'lagi', 'je': 'sahaja',
    'ni': 'ini', 'tu': 'itu', 'kat': 'dekat', 'dah': 'sudah',
    'blh': 'boleh', 'ngan': 'dengan', 'sgt': 'sangat', 'mmg': 'memang',
    'org': 'orang', 'brp': 'berapa', 'cmne': 'macam mana',
    'ade': 'ada', 'xnak': 'tak nak', 'tgh': 'tengah',
    'skrg': 'sekarang', 'smpai': 'sampai', 'blk': 'balik',
    'psl': 'pasal', 'sblm': 'sebelum', 'slps': 'selepas',
    'byk': 'banyak', 'sikit': 'sedikit', 'dkt': 'dekat',
    'kt': 'dekat', 'nnt': 'nanti', 'jgn': 'jangan',
    'klu': 'kalau', 'kalo': 'kalau', 'tp': 'tapi',
}

# Stopwords (combined BM + EN common words)
_STOPWORDS = {
    # BM
    'yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada',
    'adalah', 'dari', 'dalam', 'akan', 'ke', 'tidak', 'ada', 'juga',
    'sudah', 'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami',
    'oleh', 'telah', 'atau', 'tetapi', 'jika', 'maka', 'lah', 'pun',
    'kan', 'la', 'je', 'tu', 'ni', 'eh', 'ah', 'oh',
    # EN
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'it', 'this', 'that', 'i', 'you', 'he', 'she', 'we', 'they',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'and', 'but', 'or', 'so', 'if', 'then', 'than', 'when',
    'what', 'which', 'who', 'how', 'not', 'no', 'just', 'very',
}


def _split_sentences(text):
    """Split text into sentences, handling Manglish patterns."""
    if not text or not text.strip():
        return []

    # Split on sentence-ending punctuation, newlines, or multiple spaces
    # Handle common patterns: periods, !, ?, newlines
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())

    sentences = []
    for part in parts:
        part = part.strip()
        if part and len(part) > 1:
            sentences.append(part)

    # If no splits found, try splitting on commas for very long single sentences
    if len(sentences) <= 1 and len(text) > 200:
        parts = re.split(r'[.!?]\s*', text.strip())
        sentences = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]

    return sentences if sentences else ([text.strip()] if text.strip() else [])


def _normalize_text(text):
    """Normalize shortforms for scoring purposes."""
    words = text.lower().split()
    normalized = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        normalized.append(_SHORTFORMS.get(clean_word, clean_word))
    return ' '.join(normalized)


def _tokenize_words(text):
    """Tokenize text into meaningful words (lowercase, no stopwords)."""
    normalized = _normalize_text(text)
    words = re.findall(r'\b\w+\b', normalized.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _sentence_similarity(sent1, sent2):
    """Calculate similarity between two sentences using word overlap."""
    words1 = set(_tokenize_words(sent1))
    words2 = set(_tokenize_words(sent2))

    if not words1 or not words2:
        return 0.0

    overlap = words1 & words2
    # Normalized overlap (Jaccard-like but with log normalization)
    denominator = math.log(len(words1) + 1) + math.log(len(words2) + 1)
    if denominator == 0:
        return 0.0

    return len(overlap) / denominator


def _build_similarity_matrix(sentences):
    """Build a similarity matrix between all sentence pairs."""
    n = len(sentences)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = _sentence_similarity(sentences[i], sentences[j])

    # Normalize rows
    for i in range(n):
        row_sum = sum(matrix[i])
        if row_sum > 0:
            matrix[i] = [x / row_sum for x in matrix[i]]

    return matrix


def _textrank(sentences, damping=0.85, max_iter=100, tol=1e-6):
    """Run TextRank (PageRank-like) algorithm on sentences."""
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    matrix = _build_similarity_matrix(sentences)

    # Initialize scores uniformly
    scores = [1.0 / n] * n

    for _ in range(max_iter):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = sum(matrix[j][i] * scores[j] for j in range(n) if j != i)
            new_scores[i] = (1 - damping) / n + damping * rank_sum

        # Check convergence
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < tol:
            break

    return scores


def get_sentence_scores(text):
    """
    Score each sentence in the text by importance.

    Args:
        text: Input text (Manglish/BM/EN)

    Returns:
        List of dicts: [{"sentence": str, "score": float, "position": int}]
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Get TextRank scores
    scores = _textrank(sentences)

    # Apply position bias (first and last sentences get a boost)
    n = len(sentences)
    position_boost = [0.0] * n
    if n > 0:
        position_boost[0] = 0.15  # First sentence boost
    if n > 1:
        position_boost[-1] = 0.10  # Last sentence boost

    # Combine scores with position bias
    final_scores = [scores[i] + position_boost[i] for i in range(n)]

    # Normalize to 0-1 range
    max_score = max(final_scores) if final_scores else 1.0
    if max_score > 0:
        final_scores = [s / max_score for s in final_scores]

    return [
        {"sentence": sentences[i], "score": round(final_scores[i], 4), "position": i}
        for i in range(n)
    ]


def summarize_sentences(text, num_sentences=3):
    """
    Extract the most important sentences from text.

    Args:
        text: Input text (Manglish/BM/EN)
        num_sentences: Number of sentences to extract (default 3)

    Returns:
        List of selected sentences in original order
    """
    if not text or not text.strip():
        return []

    scored = get_sentence_scores(text)
    if not scored:
        return []

    # If fewer sentences than requested, return all
    if len(scored) <= num_sentences:
        return [s["sentence"] for s in scored]

    # Sort by score, take top N
    top = sorted(scored, key=lambda x: x["score"], reverse=True)[:num_sentences]

    # Return in original order (by position)
    top_sorted = sorted(top, key=lambda x: x["position"])
    return [s["sentence"] for s in top_sorted]


def summarize(text, num_sentences=3, method='textrank'):
    """
    Summarize text using extractive summarization.

    Args:
        text: Input text (Manglish/BM/EN)
        num_sentences: Number of sentences in summary (default 3)
        method: Summarization method (currently only 'textrank')

    Returns:
        Summary string
    """
    if not text or not text.strip():
        return ""

    if method != 'textrank':
        raise ValueError(f"Unsupported method: {method}. Use 'textrank'.")

    sentences = summarize_sentences(text, num_sentences)
    return ' '.join(sentences)


def extract_key_phrases(text, top_n=10):
    """
    Extract key phrases from text using word co-occurrence.

    Args:
        text: Input text (Manglish/BM/EN)
        top_n: Number of key phrases to return (default 10)

    Returns:
        List of key phrases sorted by score
    """
    if not text or not text.strip():
        return []

    # Normalize for processing
    normalized = _normalize_text(text)
    words = re.findall(r'\b\w+\b', normalized.lower())
    content_words = [w for w in words if w not in _STOPWORDS and len(w) > 1]

    if not content_words:
        return []

    # Build word frequency
    word_freq = Counter(content_words)

    # Extract bigrams and trigrams as phrases
    phrases = Counter()

    # Unigrams (single important words)
    for word, freq in word_freq.items():
        if freq >= 1:
            phrases[word] = freq

    # Bigrams
    for i in range(len(content_words) - 1):
        bigram = f"{content_words[i]} {content_words[i+1]}"
        phrases[bigram] += 2  # Boost multi-word phrases

    # Trigrams
    for i in range(len(content_words) - 2):
        trigram = f"{content_words[i]} {content_words[i+1]} {content_words[i+2]}"
        phrases[trigram] += 3  # Higher boost for longer phrases

    # Score phrases: frequency * length bonus
    scored = {}
    for phrase, freq in phrases.items():
        word_count = len(phrase.split())
        score = freq * (1 + 0.5 * (word_count - 1))
        scored[phrase] = score

    # Sort by score and return top N
    sorted_phrases = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, _ in sorted_phrases[:top_n]]


def summarize_thread(messages, num_points=5):
    """
    Summarize a WhatsApp/chat thread into key points.

    Args:
        messages: List of message strings
        num_points: Number of key points to extract (default 5)

    Returns:
        Summary string with key points
    """
    if not messages:
        return ""

    # Filter empty messages
    messages = [m.strip() for m in messages if m and m.strip()]
    if not messages:
        return ""

    # If very few messages, just return them joined
    if len(messages) <= num_points:
        return '\n'.join(f"• {m}" for m in messages)

    # Group messages into chunks (pseudo-topics based on proximity)
    # Simple approach: sliding window to find topic shifts
    chunks = []
    current_chunk = [messages[0]]

    for i in range(1, len(messages)):
        # Check similarity with current chunk's last message
        sim = _sentence_similarity(messages[i], current_chunk[-1])
        if sim < 0.05 and len(current_chunk) >= 2:
            # Topic shift - start new chunk
            chunks.append(current_chunk)
            current_chunk = [messages[i]]
        else:
            current_chunk.append(messages[i])

    if current_chunk:
        chunks.append(current_chunk)

    # From each chunk, extract the most representative message
    key_points = []
    for chunk in chunks:
        if len(chunk) == 1:
            key_points.append(chunk[0])
        else:
            # Combine chunk into one text and find best sentence
            combined = ' '.join(chunk)
            scores = get_sentence_scores(combined)
            if scores:
                best = max(scores, key=lambda x: x["score"])
                key_points.append(best["sentence"])
            else:
                key_points.append(chunk[0])

    # If we have more points than requested, score and select top ones
    if len(key_points) > num_points:
        # Score by length (longer = more informative) and position
        scored_points = []
        for i, point in enumerate(key_points):
            word_count = len(point.split())
            # Prefer longer, more informative messages
            score = word_count + (0.5 if i == 0 else 0) + (0.3 if i == len(key_points) - 1 else 0)
            scored_points.append((point, score, i))

        scored_points.sort(key=lambda x: x[1], reverse=True)
        top_points = scored_points[:num_points]
        # Restore original order
        top_points.sort(key=lambda x: x[2])
        key_points = [p[0] for p in top_points]

    return '\n'.join(f"• {point}" for point in key_points)
