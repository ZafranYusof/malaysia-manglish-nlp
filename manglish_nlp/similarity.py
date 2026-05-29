"""Text similarity functions for Manglish text.

Supports Jaccard, cosine (bag-of-words), and token overlap similarity.
"""

import re
import math
from collections import Counter
from manglish_nlp.tokenizer import word_tokenize
from manglish_nlp.stemmer import stem_word
from manglish_nlp.normalize import normalize


def jaccard(text1, text2, use_stem=False):
    """Compute Jaccard similarity between two texts.
    
    Parameters:
        text1 (str): First text.
        text2 (str): Second text.
        use_stem (bool): Stem words before comparison (default: False).
    
    Returns:
        float: Similarity score 0-1.
    
    Example:
        >>> jaccard("aku nak makan nasi", "aku nak makan roti")
        0.6
        >>> jaccard("memakan", "makan", use_stem=True)
        1.0
    """
    tokens1 = _get_tokens(text1, use_stem)
    tokens2 = _get_tokens(text2, use_stem)
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return round(intersection / union, 4) if union > 0 else 0.0


def cosine(text1, text2, use_stem=False):
    """Compute cosine similarity between two texts (bag-of-words).
    
    Parameters:
        text1 (str): First text.
        text2 (str): Second text.
        use_stem (bool): Stem words before comparison (default: False).
    
    Returns:
        float: Similarity score 0-1.
    
    Example:
        >>> cosine("makanan sedap sangat", "makanan memang sedap")
        0.667
    """
    tokens1 = _get_tokens(text1, use_stem)
    tokens2 = _get_tokens(text2, use_stem)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # All unique terms
    all_terms = set(counter1.keys()) | set(counter2.keys())
    
    # Dot product
    dot = sum(counter1.get(t, 0) * counter2.get(t, 0) for t in all_terms)
    
    # Magnitudes
    mag1 = math.sqrt(sum(v ** 2 for v in counter1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in counter2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return round(dot / (mag1 * mag2), 4)


def overlap(text1, text2, use_stem=False):
    """Compute overlap coefficient between two texts.
    
    Overlap = |intersection| / min(|set1|, |set2|)
    Good for comparing short text against long text.
    
    Parameters:
        text1 (str): First text.
        text2 (str): Second text.
        use_stem (bool): Stem words before comparison (default: False).
    
    Returns:
        float: Similarity score 0-1.
    
    Example:
        >>> overlap("nak makan", "aku nak makan nasi goreng")
        1.0
    """
    tokens1 = _get_tokens(text1, use_stem)
    tokens2 = _get_tokens(text2, use_stem)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))
    
    return round(intersection / min_size, 4) if min_size > 0 else 0.0


def semantic_similarity(text1, text2):
    """Compute semantic similarity using normalization + stemming + cosine.
    
    Higher-level similarity that accounts for shortforms and word variants.
    
    Parameters:
        text1 (str): First text.
        text2 (str): Second text.
    
    Returns:
        dict: Result with keys:
            - score (float): Combined similarity 0-1
            - jaccard (float): Jaccard score
            - cosine (float): Cosine score
            - overlap (float): Overlap score
            - method (str): 'normalized+stemmed'
    
    Example:
        >>> semantic_similarity("nk mkn nasi", "nak makan nasi")
        {'score': 1.0, ...}
    """
    # Normalize shortforms first
    norm1 = normalize(text1)
    norm2 = normalize(text2)
    
    j = jaccard(norm1, norm2, use_stem=True)
    c = cosine(norm1, norm2, use_stem=True)
    o = overlap(norm1, norm2, use_stem=True)
    
    # Weighted combination
    score = 0.4 * c + 0.35 * j + 0.25 * o
    
    return {
        'score': round(score, 4),
        'jaccard': j,
        'cosine': c,
        'overlap': o,
        'method': 'normalized+stemmed',
    }


def find_most_similar(query, candidates, top_n=5, method='semantic'):
    """Find most similar texts from a list of candidates.
    
    Parameters:
        query (str): Query text.
        candidates (list[str]): List of candidate texts.
        top_n (int): Number of results (default: 5).
        method (str): 'semantic', 'jaccard', 'cosine', or 'overlap'.
    
    Returns:
        list[dict]: Ranked results with 'text', 'score', 'index'.
    
    Example:
        >>> find_most_similar("nk mkn", ["nak makan nasi", "nak pergi", "makan sedap"])
        [{'text': 'nak makan nasi', 'score': 0.9, 'index': 0}, ...]
    """
    results = []
    
    for i, candidate in enumerate(candidates):
        if method == 'semantic':
            sim = semantic_similarity(query, candidate)
            score = sim['score']
        elif method == 'jaccard':
            score = jaccard(query, candidate, use_stem=True)
        elif method == 'cosine':
            score = cosine(query, candidate, use_stem=True)
        elif method == 'overlap':
            score = overlap(query, candidate, use_stem=True)
        else:
            score = jaccard(query, candidate)
        
        results.append({'text': candidate, 'score': score, 'index': i})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def _get_tokens(text, use_stem=False):
    """Tokenize and optionally stem text."""
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha() and len(t) >= 2]
    if use_stem:
        tokens = [stem_word(t) for t in tokens]
    return tokens
