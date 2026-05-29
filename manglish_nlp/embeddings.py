"""Sentence embeddings for Malaysian text.

Provides lightweight vector representations without external dependencies.
Supports: bag-of-words, TF-IDF, and weighted word vectors.
"""

import math
import re
from collections import Counter
from manglish_nlp.tokenizer import word_tokenize
from manglish_nlp.stemmer import stem_word
from manglish_nlp.normalize import normalize
from manglish_nlp.dictionary import get_stopwords


def sentence_vector(text, method='tfidf', normalize_text=True, use_stem=True):
    """Generate a vector representation of text.
    
    Parameters:
        text (str): Input text.
        method (str): 'bow' (bag-of-words), 'tfidf', or 'weighted'.
        normalize_text (bool): Normalize shortforms first.
        use_stem (bool): Stem words before vectorizing.
    
    Returns:
        dict: Vector representation with 'vector' (word->weight), 'dim', 'method'.
    
    Example:
        >>> sentence_vector("aku nak makan nasi goreng")
        {'vector': {'aku': 0.2, 'nak': 0.2, 'makan': 0.2, ...}, 'dim': 5, ...}
    """
    if normalize_text:
        text = normalize(text)
    
    tokens = _get_tokens(text, use_stem)
    
    if not tokens:
        return {'vector': {}, 'dim': 0, 'method': method, 'tokens': []}
    
    if method == 'bow':
        vector = _bow_vector(tokens)
    elif method == 'tfidf':
        vector = _tfidf_vector(tokens)
    elif method == 'weighted':
        vector = _weighted_vector(tokens)
    else:
        vector = _bow_vector(tokens)
    
    return {
        'vector': vector,
        'dim': len(vector),
        'method': method,
        'tokens': tokens,
    }


def encode_batch(texts, method='tfidf', normalize_text=True, use_stem=True):
    """Encode multiple texts into vectors with shared vocabulary.
    
    Parameters:
        texts (list[str]): Input texts.
        method (str): Vectorization method.
        normalize_text (bool): Normalize shortforms.
        use_stem (bool): Stem words.
    
    Returns:
        dict: Result with 'vectors' (list of dicts), 'vocabulary', 'idf_scores'.
    
    Example:
        >>> encode_batch(["aku nak makan", "dia nak pergi"])
        {'vectors': [...], 'vocabulary': ['aku', 'nak', 'makan', 'dia', 'pergi'], ...}
    """
    # Tokenize all
    all_tokens = []
    for text in texts:
        if normalize_text:
            text = normalize(text)
        tokens = _get_tokens(text, use_stem)
        all_tokens.append(tokens)
    
    # Build vocabulary
    vocab = sorted(set(t for tokens in all_tokens for t in tokens))
    
    # Compute IDF
    n_docs = len(all_tokens)
    idf = {}
    for word in vocab:
        doc_count = sum(1 for tokens in all_tokens if word in tokens)
        idf[word] = math.log(n_docs / (1 + doc_count)) + 1
    
    # Generate vectors
    vectors = []
    for tokens in all_tokens:
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        
        if method == 'tfidf':
            vec = {w: (tf.get(w, 0) / max_tf) * idf.get(w, 1) for w in vocab if tf.get(w, 0) > 0}
        elif method == 'bow':
            vec = {w: tf.get(w, 0) for w in vocab if tf.get(w, 0) > 0}
        else:
            vec = {w: tf.get(w, 0) / max_tf for w in vocab if tf.get(w, 0) > 0}
        
        vectors.append(vec)
    
    return {
        'vectors': vectors,
        'vocabulary': vocab,
        'idf_scores': idf,
        'n_docs': n_docs,
    }


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two sparse vectors.
    
    Parameters:
        vec1 (dict): First vector (word->weight).
        vec2 (dict): Second vector (word->weight).
    
    Returns:
        float: Similarity score 0-1.
    """
    if not vec1 or not vec2:
        return 0.0
    
    all_keys = set(vec1.keys()) | set(vec2.keys())
    
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return round(dot / (mag1 * mag2), 4)


def find_nearest(query, corpus, top_n=5, method='tfidf'):
    """Find most similar texts from corpus using embeddings.
    
    Parameters:
        query (str): Query text.
        corpus (list[str]): List of texts to search.
        top_n (int): Number of results.
        method (str): Vectorization method.
    
    Returns:
        list[dict]: Ranked results with 'text', 'score', 'index'.
    
    Example:
        >>> find_nearest("nk mkn nasi", ["nak makan nasi goreng", "nak pergi kedai", "tidur"])
        [{'text': 'nak makan nasi goreng', 'score': 0.95, 'index': 0}, ...]
    """
    # Encode query + corpus together for shared vocab
    all_texts = [query] + corpus
    encoded = encode_batch(all_texts, method=method)
    
    query_vec = encoded['vectors'][0]
    
    results = []
    for i, vec in enumerate(encoded['vectors'][1:]):
        score = cosine_similarity(query_vec, vec)
        results.append({'text': corpus[i], 'score': score, 'index': i})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def cluster_texts(texts, threshold=0.5, method='tfidf'):
    """Simple clustering of texts by similarity.
    
    Groups texts that are above similarity threshold.
    
    Parameters:
        texts (list[str]): Input texts.
        threshold (float): Minimum similarity to group together.
        method (str): Vectorization method.
    
    Returns:
        list[list[int]]: Clusters (lists of text indices).
    
    Example:
        >>> cluster_texts(["nak makan nasi", "nak makan roti", "nak pergi kedai", "pergi kedai beli barang"])
        [[0, 1], [2, 3]]
    """
    encoded = encode_batch(texts, method=method)
    vectors = encoded['vectors']
    n = len(texts)
    
    # Build similarity matrix
    assigned = [False] * n
    clusters = []
    
    for i in range(n):
        if assigned[i]:
            continue
        
        cluster = [i]
        assigned[i] = True
        
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True
        
        clusters.append(cluster)
    
    return clusters


def _get_tokens(text, use_stem=False):
    """Tokenize and filter text."""
    stopwords = get_stopwords('all')
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha() and len(t) >= 2]
    tokens = [t for t in tokens if t not in stopwords]
    if use_stem:
        tokens = [stem_word(t) for t in tokens]
    return tokens


def _bow_vector(tokens):
    """Bag-of-words vector (normalized frequency)."""
    counts = Counter(tokens)
    total = sum(counts.values())
    return {w: round(c / total, 4) for w, c in counts.items()}


def _tfidf_vector(tokens):
    """Single-document TF-IDF approximation."""
    counts = Counter(tokens)
    max_count = max(counts.values()) if counts else 1
    # Approximate IDF using token rarity (less common = higher weight)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    
    vector = {}
    for word, count in counts.items():
        tf = count / max_count
        # Pseudo-IDF: words appearing once get higher weight
        idf = 1.0 + (1.0 if count == 1 else 0.0)
        vector[word] = round(tf * idf * unique_ratio, 4)
    
    return vector


def _weighted_vector(tokens):
    """Weighted vector (position-aware: earlier words get more weight)."""
    vector = {}
    n = len(tokens)
    
    for i, token in enumerate(tokens):
        # Position weight: earlier = more important
        pos_weight = 1.0 - (i / (n * 2))
        if token in vector:
            vector[token] += pos_weight
        else:
            vector[token] = pos_weight
    
    # Normalize
    max_val = max(vector.values()) if vector else 1
    return {w: round(v / max_val, 4) for w, v in vector.items()}
