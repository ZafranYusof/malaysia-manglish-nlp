"""N-gram based Manglish text generation and autocomplete.

Provides trigram language model with backoff to bigram and unigram for
generating realistic Manglish text and predicting next words.

Zero external dependencies — uses only Python stdlib.

Usage:
    from manglish_nlp import text_generation

    # Generate text
    text = text_generation.generate("aku nak", max_words=20)

    # Autocomplete
    predictions = text_generation.autocomplete("aku nak", top_n=5)

    # Calculate perplexity
    score = text_generation.perplexity("aku nak makan nasi lemak")

    # Style-specific generation
    tweet = text_generation.generate_sentence(style='twitter')
"""

import json
import math
import os
import random
import re

_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources')
_MODEL_PATH = os.path.join(_RESOURCES_DIR, 'ngram_model.json')

# Special tokens
_START = '<s>'
_END = '</s>'

# ============================================================
# Default Model / Corpus
# ============================================================

# Vocabulary pools for synthetic corpus (subset for model building)
_SUBJECTS = [
    'aku', 'kau', 'dia', 'kita', 'kami', 'korang', 'diorang', 'awak',
    'saya', 'bro', 'sis', 'boss', 'member', 'kawan', 'budak', 'orang',
]

_VERBS = [
    'makan', 'pergi', 'beli', 'tengok', 'dengar', 'cakap', 'tulis',
    'baca', 'main', 'kerja', 'tidur', 'bangun', 'lari', 'jalan',
    'masak', 'hantar', 'ambil', 'buat', 'try', 'lepak', 'chill',
    'study', 'drive', 'order', 'cancel', 'download', 'scroll',
    'like', 'share', 'post', 'reply', 'call', 'text', 'grab',
]

_OBJECTS = [
    'nasi lemak', 'roti canai', 'teh tarik', 'milo ais', 'kopi',
    'nasi goreng', 'mee goreng', 'laksa', 'satay', 'rendang',
    'mamak', 'kedai', 'pasar', 'mall', 'office', 'rumah', 'sekolah',
    'phone', 'laptop', 'kereta', 'motor', 'grab', 'shopee', 'tiktok',
]

_ADJECTIVES = [
    'best', 'gila', 'power', 'terror', 'mantap', 'sedap', 'cantik',
    'lawa', 'comel', 'mahal', 'murah', 'besar', 'kecik', 'panas',
    'sejuk', 'baru', 'lama', 'cepat', 'lambat', 'senang', 'susah',
    'bagus', 'teruk', 'okay', 'solid', 'epic', 'cringe', 'fire',
]

_PARTICLES = [
    'la', 'lah', 'wei', 'weh', 'eh', 'kan', 'kot', 'je', 'jer',
    'doh', 'doe', 'bro', 'sis', 'sial', 'gila',
]

_LOCATIONS = [
    'KL', 'Penang', 'JB', 'Ipoh', 'Melaka', 'Kuantan', 'Selangor',
    'Cyberjaya', 'Shah Alam', 'Subang', 'PJ', 'Bangsar', 'KLCC',
    'Cheras', 'Ampang', 'Setapak',
]

_SHORTFORMS = {
    'nak': 'nk', 'macam': 'mcm', 'yang': 'yg', 'sebab': 'sbb',
    'dengan': 'dgn', 'dalam': 'dlm', 'untuk': 'utk', 'sudah': 'dah',
    'tidak': 'tak', 'belum': 'blm', 'boleh': 'blh', 'orang': 'org',
    'sangat': 'sgt', 'betul': 'btl', 'memang': 'mmg', 'tengah': 'tgh',
    'sekarang': 'skrg', 'kenapa': 'knp', 'dekat': 'dkt', 'balik': 'blk',
}

# Style-specific templates
_STYLE_TEMPLATES = {
    'twitter': [
        "{subj} {verb} {obj} {particle}",
        "korang {verb} {obj} tak",
        "sape {verb} {obj} hari ni",
        "{adj} {particle} {obj} tu",
        "baru {verb} {obj} dekat {loc}",
        "confirm {adj} kalau {verb} {obj}",
        "unpopular opinion {obj} overrated {particle}",
        "ratio {subj} {verb} {obj}",
        "pov {subj} {verb} {obj} first time",
        "normalize {verb} {obj} at {loc}",
        "aku literally {verb} {obj} tiga kali dah",
        "obsessed with {obj} lately {particle}",
        "no because why is {obj} so {adj}",
        "hot take {obj} better than {obj2}",
    ],
    'whatsapp': [
        "{subj} {verb} {obj} {particle}",
        "wei {verb} {obj} jom",
        "ok nanti {subj} {verb} {obj}",
        "haha {adj} gila {obj} tu",
        "serious ke {subj} {verb} {obj}",
        "jap aku {verb} dulu",
        "dah {verb} {obj} ke blm",
        "nk {verb} {obj} skrg ke ptg",
        "sape nk {verb} {obj} sama",
        "confirm {subj} {verb} {obj} esok",
        "aku dah penat {verb} {obj} dah",
        "ok noted nanti {verb} {obj}",
        "sorry lambat reply tgh {verb}",
        "last minute tapi jom {verb} {obj}",
        "eh {subj} free tak nk {verb}",
    ],
    'reddit': [
        "guys anyone know where to {verb} {obj} in {loc}",
        "is it just me or {obj} getting more {adj}",
        "PSA {obj} at {loc} is {adj} {particle}",
        "rant {subj} always {verb} {obj} without asking",
        "TIL {obj} in Malaysia is actually {adj}",
        "unpopular opinion {loc} {obj} is overrated",
        "help need recommendation for {obj} near {loc}",
        "anyone else {verb} {obj} everyday or just me",
        "serious question best {obj} in {loc}",
        "just moved to {loc} where to {verb} good {obj}",
        "ELI5 why is {obj} so {adj} in Malaysia",
    ],
    'news': [
        "typical {particle} kerajaan ni",
        "rakyat susah tapi {subj} {verb} {obj}",
        "bila nak turun harga {obj}",
        "dah la {obj} {adj} pastu {verb} lagi",
        "setuju sangat {obj} patut {adj}",
        "ini semua salah {subj} {particle}",
        "harap {obj} jadi lebih {adj}",
        "Malaysia boleh kalau {verb} {obj} betul betul",
        "siapa approve {verb} {obj} ni",
        "rakyat biasa mana mampu {verb} {obj}",
        "bagus la kalau betul {obj} {adj}",
        "tunggu la sampai {obj} naik harga lagi",
    ],
}


def _tokenize(text):
    """Simple tokenizer for text generation."""
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?|[^\s\w]", text.lower())
    return [t for t in tokens if t.strip()]


def _generate_synthetic_corpus(n_sentences=3000, seed=42):
    """Generate synthetic Manglish corpus for model building."""
    rng = random.Random(seed)
    sentences = []

    all_templates = []
    for style, templates in _STYLE_TEMPLATES.items():
        weight = {'twitter': 30, 'whatsapp': 35, 'reddit': 20, 'news': 15}.get(style, 20)
        all_templates.extend([(t, style) for t in templates] * weight)

    for _ in range(n_sentences):
        template, style = rng.choice(all_templates)

        subj = rng.choice(_SUBJECTS)
        verb = rng.choice(_VERBS)
        obj_words = rng.choice(_OBJECTS)
        obj2_words = rng.choice(_OBJECTS)
        adj = rng.choice(_ADJECTIVES)
        particle = rng.choice(_PARTICLES)
        loc = rng.choice(_LOCATIONS)

        sentence = template.format(
            subj=subj, verb=verb, obj=obj_words, obj2=obj2_words,
            adj=adj, particle=particle, loc=loc,
        )

        # Apply shortforms randomly (30% chance)
        if rng.random() < 0.3:
            for full, short in _SHORTFORMS.items():
                if full in sentence and rng.random() < 0.4:
                    sentence = sentence.replace(full, short, 1)

        tokens = _tokenize(sentence)
        if tokens:
            sentences.append(tokens)

    rng.shuffle(sentences)
    return sentences


# ============================================================
# N-gram Model
# ============================================================

def build_ngram_model(texts, n=3):
    """Build an n-gram language model from tokenized texts.

    Parameters:
        texts (list): Either list of strings or list of token lists.
        n (int): N-gram order (default 3 for trigram).

    Returns:
        dict: Model containing unigrams, bigrams, trigrams, and metadata.

    Example:
        >>> model = build_ngram_model(["aku nak makan", "dia nak pergi"])
        >>> 'unigrams' in model
        True
    """
    unigrams = {}
    bigrams = {}
    trigrams = {}
    total_tokens = 0

    for text in texts:
        if isinstance(text, str):
            tokens = _tokenize(text)
        else:
            tokens = [t.lower() for t in text]

        if not tokens:
            continue

        # Add start/end tokens
        padded = [_START, _START] + tokens + [_END]

        # Count unigrams
        for token in tokens:
            unigrams[token] = unigrams.get(token, 0) + 1
            total_tokens += 1

        # Count bigrams
        for i in range(len(padded) - 1):
            key = padded[i] + '|' + padded[i + 1]
            bigrams[key] = bigrams.get(key, 0) + 1

        # Count trigrams
        for i in range(len(padded) - 2):
            key = padded[i] + '|' + padded[i + 1] + '|' + padded[i + 2]
            trigrams[key] = trigrams.get(key, 0) + 1

    # Prune to top N entries to keep model small
    max_unigrams = 2000
    max_bigrams = 3000
    max_trigrams = 5000

    if len(unigrams) > max_unigrams:
        sorted_uni = sorted(unigrams.items(), key=lambda x: x[1], reverse=True)
        unigrams = dict(sorted_uni[:max_unigrams])

    if len(bigrams) > max_bigrams:
        sorted_bi = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
        bigrams = dict(sorted_bi[:max_bigrams])

    if len(trigrams) > max_trigrams:
        sorted_tri = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)
        trigrams = dict(sorted_tri[:max_trigrams])

    model = {
        'n': n,
        'unigrams': unigrams,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'total_tokens': total_tokens,
        'vocab_size': len(unigrams),
    }

    return model


def _save_model(model, path=None):
    """Save model to JSON file."""
    if path is None:
        path = _MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)


def _load_model(path=None):
    """Load model from JSON file."""
    if path is None:
        path = _MODEL_PATH
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Global cached model
_cached_model = None


def load_default_model():
    """Load or build the default n-gram model.

    Loads pre-built model from resources/ if available,
    otherwise builds from synthetic corpus and saves.

    Returns:
        dict: The n-gram model.

    Example:
        >>> model = load_default_model()
        >>> 'unigrams' in model and 'bigrams' in model
        True
    """
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    # Try loading from file
    model = _load_model()
    if model is not None:
        _cached_model = model
        return model

    # Build from synthetic corpus
    corpus = _generate_synthetic_corpus(n_sentences=3000, seed=42)
    model = build_ngram_model(corpus, n=3)

    # Save for future use
    try:
        _save_model(model)
    except (OSError, IOError):
        pass  # Non-critical if save fails

    _cached_model = model
    return model


def _get_model():
    """Get the current model, loading default if needed."""
    global _cached_model
    if _cached_model is None:
        load_default_model()
    return _cached_model


# ============================================================
# Probability / Sampling
# ============================================================

def _get_next_word_probs(context, model, temperature=1.0):
    """Get probability distribution for next word given context.

    Uses trigram with backoff to bigram and unigram.
    Applies Laplace smoothing for unseen n-grams.

    Parameters:
        context (list): Previous tokens (last 2 used for trigram).
        model (dict): The n-gram model.
        temperature (float): Sampling temperature.

    Returns:
        dict: Word -> probability mapping.
    """
    unigrams = model['unigrams']
    bigrams = model['bigrams']
    trigrams = model['trigrams']
    vocab_size = model.get('vocab_size', len(unigrams))
    total_tokens = model.get('total_tokens', sum(unigrams.values()))

    # Smoothing constant
    k = 0.1

    candidates = {}

    # Pad context
    if len(context) == 0:
        context = [_START, _START]
    elif len(context) == 1:
        context = [_START] + context

    w1, w2 = context[-2], context[-1]

    # Trigram probabilities (weight 0.6)
    tri_prefix = w1 + '|' + w2 + '|'
    tri_total = 0
    tri_counts = {}
    for key, count in trigrams.items():
        if key.startswith(tri_prefix):
            word = key[len(tri_prefix):]
            tri_counts[word] = count
            tri_total += count

    # Bigram probabilities (weight 0.3)
    bi_prefix = w2 + '|'
    bi_total = 0
    bi_counts = {}
    for key, count in bigrams.items():
        if key.startswith(bi_prefix):
            word = key[len(bi_prefix):]
            bi_counts[word] = count
            bi_total += count

    # Combine with interpolation weights
    lambda_tri = 0.6 if tri_total > 0 else 0.0
    lambda_bi = 0.3 if bi_total > 0 else 0.0
    lambda_uni = 1.0 - lambda_tri - lambda_bi

    # Ensure lambda_uni is at least some minimum
    if lambda_uni < 0.1:
        lambda_uni = 0.1
        total_lambda = lambda_tri + lambda_bi + lambda_uni
        lambda_tri /= total_lambda
        lambda_bi /= total_lambda
        lambda_uni /= total_lambda

    all_words = set(list(unigrams.keys()) + list(tri_counts.keys()) + list(bi_counts.keys()))

    for word in all_words:
        if word == _START:
            continue

        prob = 0.0

        # Trigram contribution
        if tri_total > 0:
            tri_count = tri_counts.get(word, 0)
            prob += lambda_tri * ((tri_count + k) / (tri_total + k * vocab_size))

        # Bigram contribution
        if bi_total > 0:
            bi_count = bi_counts.get(word, 0)
            prob += lambda_bi * ((bi_count + k) / (bi_total + k * vocab_size))

        # Unigram contribution
        uni_count = unigrams.get(word, 0)
        prob += lambda_uni * ((uni_count + k) / (total_tokens + k * vocab_size))

        if prob > 0:
            candidates[word] = prob

    if not candidates:
        # Fallback: uniform over vocabulary
        for word in list(unigrams.keys())[:50]:
            if word != _START:
                candidates[word] = 1.0 / 50

    # Apply temperature
    if temperature != 1.0 and temperature > 0:
        for word in candidates:
            candidates[word] = candidates[word] ** (1.0 / temperature)

    # Normalize
    total = sum(candidates.values())
    if total > 0:
        for word in candidates:
            candidates[word] /= total

    return candidates


def _sample_word(probs, rng=None):
    """Sample a word from probability distribution."""
    if rng is None:
        rng = random

    if not probs:
        return _END

    words = list(probs.keys())
    weights = [probs[w] for w in words]

    # Weighted random choice
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for word, weight in zip(words, weights):
        cumulative += weight
        if r <= cumulative:
            return word

    return words[-1]


# ============================================================
# Public API
# ============================================================

def generate(prompt='', max_words=50, temperature=1.0):
    """Generate Manglish text continuation from a prompt.

    Parameters:
        prompt (str): Starting text (can be empty for random generation).
        max_words (int): Maximum number of words to generate.
        temperature (float): Controls randomness. <1 = more predictable,
                           >1 = more creative/random.

    Returns:
        str: Generated text including the prompt.

    Example:
        >>> text = generate("aku nak", max_words=10)
        >>> text.startswith("aku nak")
        True
        >>> len(text.split()) <= 12  # prompt + max_words
        True
    """
    model = _get_model()

    if prompt:
        tokens = _tokenize(prompt)
    else:
        tokens = []

    generated = list(tokens)
    context = [_START, _START] if not tokens else tokens[:]

    for _ in range(max_words):
        probs = _get_next_word_probs(context, model, temperature=temperature)
        next_word = _sample_word(probs)

        if next_word == _END:
            break

        generated.append(next_word)
        context.append(next_word)

        # Keep context window manageable
        if len(context) > 10:
            context = context[-5:]

    return ' '.join(generated)


def autocomplete(prefix, top_n=5):
    """Predict next words given a prefix.

    Parameters:
        prefix (str): Input text to complete.
        top_n (int): Number of predictions to return.

    Returns:
        list[str]: Top predicted next words, ordered by probability.

    Example:
        >>> predictions = autocomplete("aku nak")
        >>> isinstance(predictions, list)
        True
        >>> len(predictions) <= 5
        True
    """
    model = _get_model()

    tokens = _tokenize(prefix) if prefix else []
    context = [_START, _START] if not tokens else tokens[:]

    probs = _get_next_word_probs(context, model, temperature=1.0)

    # Remove end token from predictions
    probs.pop(_END, None)

    # Sort by probability
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_words[:top_n]]


def generate_sentence(style='twitter'):
    """Generate a complete sentence in a specific Manglish style.

    Parameters:
        style (str): One of 'twitter', 'whatsapp', 'reddit', 'news'.

    Returns:
        str: Generated sentence in the specified style.

    Example:
        >>> sentence = generate_sentence(style='whatsapp')
        >>> isinstance(sentence, str)
        True
        >>> len(sentence) > 0
        True
    """
    if style not in _STYLE_TEMPLATES:
        style = 'twitter'

    templates = _STYLE_TEMPLATES[style]
    template = random.choice(templates)

    subj = random.choice(_SUBJECTS)
    verb = random.choice(_VERBS)
    obj_words = random.choice(_OBJECTS)
    obj2_words = random.choice(_OBJECTS)
    adj = random.choice(_ADJECTIVES)
    particle = random.choice(_PARTICLES)
    loc = random.choice(_LOCATIONS)

    sentence = template.format(
        subj=subj, verb=verb, obj=obj_words, obj2=obj2_words,
        adj=adj, particle=particle, loc=loc,
    )

    # Apply shortforms for whatsapp style (more informal)
    if style == 'whatsapp' and random.random() < 0.5:
        for full, short in _SHORTFORMS.items():
            if full in sentence and random.random() < 0.4:
                sentence = sentence.replace(full, short, 1)

    return sentence


def perplexity(text):
    """Calculate perplexity of text under the n-gram model.

    Lower perplexity = text is more predictable/expected by the model.
    Higher perplexity = text is more surprising/unusual.

    Parameters:
        text (str): Input text to evaluate.

    Returns:
        float: Perplexity score (lower = more expected).

    Example:
        >>> score = perplexity("aku nak makan nasi lemak")
        >>> isinstance(score, float)
        True
        >>> score > 0
        True
    """
    model = _get_model()
    tokens = _tokenize(text)

    if not tokens:
        return float('inf')

    unigrams = model['unigrams']
    bigrams = model['bigrams']
    trigrams = model['trigrams']
    vocab_size = model.get('vocab_size', len(unigrams))
    total_tokens = model.get('total_tokens', sum(unigrams.values()))

    k = 0.1  # Smoothing
    padded = [_START, _START] + tokens + [_END]
    log_prob_sum = 0.0
    n_tokens = len(tokens) + 1  # +1 for END token

    for i in range(2, len(padded)):
        w1, w2, w3 = padded[i - 2], padded[i - 1], padded[i]

        # Trigram
        tri_key = w1 + '|' + w2 + '|' + w3
        tri_prefix = w1 + '|' + w2 + '|'
        tri_count = trigrams.get(tri_key, 0)
        tri_context = sum(v for key, v in trigrams.items() if key.startswith(tri_prefix))

        # Bigram
        bi_key = w2 + '|' + w3
        bi_prefix = w2 + '|'
        bi_count = bigrams.get(bi_key, 0)
        bi_context = sum(v for key, v in bigrams.items() if key.startswith(bi_prefix))

        # Unigram
        uni_count = unigrams.get(w3, 0)

        # Interpolated probability with smoothing
        lambda_tri = 0.6 if tri_context > 0 else 0.0
        lambda_bi = 0.3 if bi_context > 0 else 0.0
        lambda_uni = 1.0 - lambda_tri - lambda_bi

        prob = 0.0
        if tri_context > 0:
            prob += lambda_tri * ((tri_count + k) / (tri_context + k * vocab_size))
        if bi_context > 0:
            prob += lambda_bi * ((bi_count + k) / (bi_context + k * vocab_size))
        prob += lambda_uni * ((uni_count + k) / (total_tokens + k * vocab_size))

        # Avoid log(0)
        if prob > 0:
            log_prob_sum += math.log2(prob)
        else:
            log_prob_sum += math.log2(1e-10)

    # Perplexity = 2^(-avg_log_prob)
    avg_log_prob = log_prob_sum / n_tokens
    return 2.0 ** (-avg_log_prob)


def reset_model():
    """Reset the cached model (useful for testing).

    Forces the next call to reload or rebuild the model.
    """
    global _cached_model
    _cached_model = None
