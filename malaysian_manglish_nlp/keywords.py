"""Keyword extraction for Malaysian text.

Supports TF-based, RAKE-inspired, and frequency-based extraction.
"""

from __future__ import annotations

from typing import Dict, List

import re
import math
from collections import Counter
from malaysian_manglish_nlp.utils import get_shortforms, get_particles
from malaysian_manglish_nlp.tokenizer import word_tokenize
from malaysian_manglish_nlp.stemmer import stem_word

# Stop words (BM + EN) to exclude from keywords
_STOP_WORDS = {
    # BM function words
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'ada', 'adalah',
    'untuk', 'dengan', 'pada', 'oleh', 'akan', 'telah', 'sudah',
    'sedang', 'masih', 'juga', 'pun', 'lagi', 'sahaja', 'hanya',
    'tidak', 'tak', 'bukan', 'belum', 'atau', 'tetapi', 'tapi',
    'dalam', 'luar', 'atas', 'bawah', 'antara', 'seperti',
    'saya', 'aku', 'awak', 'kamu', 'dia', 'mereka', 'kami', 'kita',
    'sangat', 'amat', 'terlalu', 'agak', 'kurang', 'lebih',
    'semua', 'setiap', 'banyak', 'sedikit', 'beberapa',
    'kalau', 'jika', 'kerana', 'sebab', 'supaya', 'agar',
    'boleh', 'perlu', 'harus', 'mesti', 'mahu', 'nak',
    'ni', 'tu', 'je', 'jer', 'la', 'lah', 'kan', 'kot', 'weh', 'wei',
    # EN stop words
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'this', 'that', 'these', 'those', 'it', 'its',
    'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us',
    'my', 'your', 'his', 'our', 'their',
    'what', 'which', 'who', 'where', 'when', 'why', 'how',
    'and', 'or', 'but', 'not', 'no', 'if', 'so', 'than', 'then',
    'very', 'just', 'also', 'too', 'only', 'really',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'about', 'into', 'through', 'during', 'before', 'after',
    'some', 'any', 'all', 'each', 'every', 'both', 'few', 'more',
}


def extract_keywords(text: str, top_n: int = 10, method: str = 'frequency') -> List[Dict[str, Any]]:
    """Extract keywords from text.
    
    Parameters:
        text (str): Input text.
        top_n (int): Number of keywords to return (default: 10).
        method (str): Extraction method - 'frequency', 'tfidf', or 'rake'.
    
    Returns:
        list[dict]: Keywords with scores, sorted by relevance.
    
    Example:
        >>> extract_keywords("makanan sedap kat kedai tu, harga murah tapi sedap gila")
        [{'keyword': 'sedap', 'score': 2.0}, {'keyword': 'kedai', 'score': 1.0}, ...]
    """
    if method == 'frequency':
        return _frequency_keywords(text, top_n)
    elif method == 'rake':
        return _rake_keywords(text, top_n)
    elif method == 'tfidf':
        return _tfidf_keywords(text, top_n)
    elif method == 'textrank':
        return _textrank_keywords(text, top_n)
    else:
        return _frequency_keywords(text, top_n)


def _frequency_keywords(text: str, top_n: int) -> List[Dict[str, Any]]:
    """Simple frequency-based keyword extraction."""
    tokens = word_tokenize(text.lower())
    
    # Filter: only alpha words, not stop words, min length 3
    words = [t for t in tokens if t.isalpha() and t not in _STOP_WORDS and len(t) >= 3]
    
    # Stem and group
    stem_map = {}  # stem -> [original words]
    for w in words:
        s = stem_word(w)
        if s not in stem_map:
            stem_map[s] = []
        stem_map[s].append(w)
    
    # Count by stem, but return most common surface form
    results = []
    for s, originals in stem_map.items():
        count = len(originals)
        # Most common surface form
        surface = Counter(originals).most_common(1)[0][0]
        results.append({'keyword': surface, 'score': float(count), 'stem': s})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def _rake_keywords(text: str, top_n: int) -> List[Dict[str, Any]]:
    """RAKE-inspired keyword extraction (phrases)."""
    # Split into phrases by stop words and punctuation
    stop_pattern = '|'.join(re.escape(w) for w in sorted(_STOP_WORDS, key=len, reverse=True))
    phrases = re.split(rf'\b(?:{stop_pattern})\b|[.,!?;:\n]', text.lower())
    phrases = [p.strip() for p in phrases if p.strip()]
    
    # Score words by co-occurrence in phrases
    word_freq = Counter()
    word_degree = Counter()
    
    for phrase in phrases:
        words = [w for w in re.findall(r'[a-zA-Z]+', phrase) if len(w) >= 3 and w not in _STOP_WORDS]
        for w in words:
            word_freq[w] += 1
            word_degree[w] += len(words)
    
    # Word score = degree / frequency
    word_score = {}
    for w in word_freq:
        word_score[w] = word_degree[w] / word_freq[w]
    
    # Phrase score = sum of word scores
    phrase_scores = []
    seen = set()
    for phrase in phrases:
        words = [w for w in re.findall(r'[a-zA-Z]+', phrase) if len(w) >= 3 and w not in _STOP_WORDS]
        if not words:
            continue
        phrase_text = ' '.join(words)
        if phrase_text in seen:
            continue
        seen.add(phrase_text)
        score = sum(word_score.get(w, 0) for w in words)
        phrase_scores.append({'keyword': phrase_text, 'score': round(score, 2)})
    
    phrase_scores.sort(key=lambda x: x['score'], reverse=True)
    return phrase_scores[:top_n]


def _tfidf_keywords(text: str, top_n: int) -> List[Dict[str, Any]]:
    """TF-IDF-inspired extraction (treats sentences as documents)."""
    from malaysian_manglish_nlp.tokenizer import sentence_tokenize
    
    sentences = sentence_tokenize(text)
    if len(sentences) <= 1:
        # Fallback to frequency for single sentence
        return _frequency_keywords(text, top_n)
    
    # Tokenize each sentence
    doc_tokens = []
    for sent in sentences:
        tokens = [t.lower() for t in word_tokenize(sent) 
                  if t.isalpha() and t.lower() not in _STOP_WORDS and len(t) >= 3]
        doc_tokens.append(tokens)
    
    # TF per document
    all_words = set()
    for tokens in doc_tokens:
        all_words.update(tokens)
    
    # IDF
    n_docs = len(doc_tokens)
    idf = {}
    for word in all_words:
        doc_count = sum(1 for tokens in doc_tokens if word in tokens)
        idf[word] = math.log(n_docs / (1 + doc_count)) + 1
    
    # TF-IDF scores (aggregate across all docs)
    scores = Counter()
    for tokens in doc_tokens:
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        for word, count in tf.items():
            scores[word] += (count / max_tf) * idf.get(word, 1)
    
    results = [{'keyword': w, 'score': round(s, 3)} for w, s in scores.most_common(top_n)]
    return results


def _textrank_keywords(text: str, top_n: int, window: int = 4) -> List[Dict[str, Any]]:
    """TextRank keyword extraction using graph-based ranking.
    
    Builds a co-occurrence graph and uses iterative ranking (PageRank-like)
    to find important words.
    """
    tokens = [t.lower() for t in word_tokenize(text)
              if t.isalpha() and t.lower() not in _STOP_WORDS and len(t) >= 3]
    
    if not tokens:
        return []
    
    # Build co-occurrence graph
    vocab = list(set(tokens))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    
    # Adjacency matrix (co-occurrence within window)
    graph = [[0.0] * n for _ in range(n)]
    
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window, len(tokens))):
            if tokens[i] != tokens[j]:
                idx_i = vocab_idx[tokens[i]]
                idx_j = vocab_idx[tokens[j]]
                graph[idx_i][idx_j] += 1.0
                graph[idx_j][idx_i] += 1.0
    
    # PageRank-style iteration
    damping = 0.85
    scores = [1.0 / n] * n
    
    for _ in range(30):  # iterations
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = 0.0
            for j in range(n):
                if graph[j][i] > 0:
                    out_sum = sum(graph[j])
                    if out_sum > 0:
                        rank_sum += graph[j][i] / out_sum * scores[j]
            new_scores[i] = (1 - damping) / n + damping * rank_sum
        scores = new_scores
    
    # Rank words
    word_scores = [(vocab[i], scores[i]) for i in range(n)]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    results = [{'keyword': w, 'score': round(s, 4)} for w, s in word_scores[:top_n]]
    return results
