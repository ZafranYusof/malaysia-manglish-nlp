"""Word2Vec and FastText embeddings for Malaysian Manglish text.

Trains dense word vectors on synthetic Malaysian social media corpus
covering Twitter, Reddit, WhatsApp, and news comment styles.

Requires gensim>=4.0 (install via: pip install manglish-nlp[embeddings])
"""

import os
import random
import re
import numpy as np

_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources')
_W2V_PATH = os.path.join(_RESOURCES_DIR, 'word2vec.model')
_FT_PATH = os.path.join(_RESOURCES_DIR, 'fasttext.model')


def _check_gensim():
    """Check if gensim is available."""
    try:
        import gensim  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "gensim is required for word embeddings. "
            "Install it with: pip install manglish-nlp[embeddings] "
            "or: pip install gensim>=4.0"
        )


# ============================================================
# Synthetic Corpus Generation
# ============================================================

# Common Manglish vocabulary pools
_SUBJECTS = [
    'aku', 'kau', 'dia', 'kita', 'kami', 'korang', 'diorang', 'awak',
    'saya', 'abang', 'kakak', 'mak', 'ayah', 'bro', 'sis', 'boss',
    'member', 'kawan', 'cikgu', 'doktor', 'budak', 'orang', 'semua',
]

_VERBS = [
    'makan', 'pergi', 'beli', 'tengok', 'dengar', 'cakap', 'tulis',
    'baca', 'main', 'kerja', 'tidur', 'bangun', 'lari', 'jalan',
    'masak', 'basuh', 'cuci', 'hantar', 'ambil', 'buat', 'try',
    'lepak', 'chill', 'study', 'drive', 'park', 'order', 'cancel',
    'download', 'upload', 'scroll', 'swipe', 'like', 'share', 'post',
    'reply', 'dm', 'call', 'text', 'whatsapp', 'google', 'grab',
]

_OBJECTS = [
    'nasi lemak', 'roti canai', 'teh tarik', 'milo ais', 'kopi',
    'nasi goreng', 'mee goreng', 'char kuey teow', 'laksa', 'satay',
    'rendang', 'ayam goreng', 'ikan bakar', 'sup kambing', 'rojak',
    'mamak', 'kedai', 'pasar', 'mall', 'office', 'rumah', 'sekolah',
    'uni', 'hospital', 'klinik', 'masjid', 'surau', 'gym', 'park',
    'phone', 'laptop', 'kereta', 'motor', 'bas', 'lrt', 'mrt',
    'grab', 'foodpanda', 'shopee', 'lazada', 'tiktok', 'instagram',
]

_ADJECTIVES = [
    'best', 'gila', 'power', 'terror', 'mantap', 'sedap', 'cantik',
    'lawa', 'comel', 'handsome', 'kacak', 'hodoh', 'buruk', 'mahal',
    'murah', 'besar', 'kecik', 'panjang', 'pendek', 'tinggi', 'rendah',
    'panas', 'sejuk', 'lembap', 'kering', 'basah', 'baru', 'lama',
    'cepat', 'lambat', 'senang', 'susah', 'bagus', 'teruk', 'okay',
    'solid', 'epic', 'cringe', 'sus', 'slay', 'fire', 'mid', 'peak',
]

_SLANG_PARTICLES = [
    'la', 'lah', 'wei', 'weh', 'eh', 'kan', 'kot', 'je', 'jer',
    'doh', 'doe', 'bro', 'sis', 'bang', 'kak', 'oi', 'yo', 'bruh',
    'gais', 'guys', 'fam', 'bestie', 'sial', 'gila', 'mampos',
]

_SHORTFORMS = {
    'nak': 'nk', 'macam': 'mcm', 'yang': 'yg', 'sebab': 'sbb',
    'dengan': 'dgn', 'dalam': 'dlm', 'untuk': 'utk', 'sudah': 'dah',
    'tidak': 'tak', 'belum': 'blm', 'boleh': 'blh', 'orang': 'org',
    'pergi': 'pgi', 'sangat': 'sgt', 'betul': 'btl', 'memang': 'mmg',
    'tengah': 'tgh', 'sekarang': 'skrg', 'malam': 'mlm', 'pagi': 'pgi',
    'petang': 'ptg', 'berapa': 'brp', 'kenapa': 'knp', 'macam mana': 'mcmne',
    'apa': 'ape', 'siapa': 'sape', 'bila': 'ble', 'mana': 'mne',
    'dekat': 'dkt', 'sampai': 'smpai', 'balik': 'blk', 'kerja': 'kje',
}

_EMOTIONS = [
    'happy', 'sad', 'angry', 'excited', 'bored', 'tired', 'hungry',
    'sleepy', 'stressed', 'relaxed', 'confused', 'surprised', 'scared',
]

_LOCATIONS = [
    'KL', 'Penang', 'JB', 'Ipoh', 'Melaka', 'Kuantan', 'KB',
    'Terengganu', 'Pahang', 'Selangor', 'Putrajaya', 'Cyberjaya',
    'Shah Alam', 'Subang', 'Petaling Jaya', 'Bangsar', 'KLCC',
    'Bukit Bintang', 'Cheras', 'Ampang', 'Setapak', 'Gombak',
]

_TWITTER_TEMPLATES = [
    "{subj} {verb} {obj} {particle}",
    "korang {verb} {obj} tak?",
    "sape {verb} {obj} hari ni?",
    "{adj} {particle} {obj} tu",
    "baru {verb} {obj} dekat {loc}",
    "kenapa {subj} {verb} {obj} {particle}",
    "confirm {adj} kalau {verb} {obj}",
    "thread: kenapa {obj} {adj} sangat",
    "unpopular opinion: {obj} overrated {particle}",
    "ratio + {subj} {verb} {obj}",
    "{subj} really said {verb} {obj} and left",
    "no because why is {obj} so {adj}",
    "the way {subj} {verb} {obj} i cant",
    "pov: {subj} {verb} {obj} first time",
    "{obj} supremacy {particle}",
    "hot take: {obj} better than {obj2}",
    "normalize {verb} {obj} at {loc}",
    "me when {subj} {verb} {obj}: 💀",
    "aku literally {verb} {obj} 3 kali dah",
    "obsessed with {obj} lately {particle}",
]

_REDDIT_TEMPLATES = [
    "Guys, anyone know where to {verb} {obj} in {loc}?",
    "Is it just me or {obj} getting more {adj}?",
    "PSA: {obj} at {loc} is {adj} {particle}",
    "Rant: {subj} always {verb} {obj} without asking",
    "TIL {obj} in Malaysia is actually {adj}",
    "What's the {adj}est {obj} you've tried in {loc}?",
    "Unpopular opinion: {loc} {obj} is overrated",
    "Help - need recommendation for {obj} near {loc}",
    "Anyone else {verb} {obj} everyday or just me?",
    "Story time: {subj} {verb} {obj} and regretted it",
    "ELI5: why is {obj} so {adj} in Malaysia",
    "Serious question - best {obj} in {loc}?",
    "Just moved to {loc}, where to {verb} good {obj}?",
    "Comparison: {obj} vs {obj2} - which one {adj}er?",
    "Monthly thread: what {obj} are you {verb}ing?",
]

_WHATSAPP_TEMPLATES = [
    "{subj} {verb} {obj} {particle}",
    "wei {verb} {obj} jom",
    "ok nanti {subj} {verb} {obj}",
    "haha {adj} gila {obj} tu",
    "serious ke {subj} {verb} {obj}?",
    "jap aku {verb} dulu",
    "dah {verb} {obj} ke blm?",
    "nk {verb} {obj} skrg ke ptg?",
    "sape nk {verb} {obj} sama?",
    "confirm {subj} {verb} {obj} esok",
    "bro {obj} tu {adj} sgt",
    "aku dah penat {verb} {obj} dah",
    "k noted nanti {verb} {obj}",
    "sorry lambat reply tgh {verb}",
    "omw dah {verb} dari {loc}",
    "wait aku {verb} {obj} kejap",
    "hahaha {adj} gila kau ni",
    "ok2 jom {verb} {obj} weekend",
    "eh {subj} free tak? nk {verb}",
    "last minute tapi jom {verb} {obj}",
]

_NEWS_COMMENT_TEMPLATES = [
    "typical {particle} kerajaan ni",
    "rakyat susah tapi {subj} {verb} {obj}",
    "bila nak turun harga {obj}?",
    "dah la {obj} {adj} pastu {verb} lagi",
    "setuju sangat {obj} patut {adj}",
    "ini semua salah {subj} {particle}",
    "harap {obj} jadi lebih {adj}",
    "Malaysia boleh kalau {verb} {obj} betul2",
    "apa jadi dengan {obj} kat {loc}?",
    "dulu {obj} {adj} sekarang dah {adj}",
    "siapa approve {verb} {obj} ni?",
    "rakyat biasa mana mampu {verb} {obj}",
    "bagus la kalau betul {obj} {adj}",
    "jangan percaya sangat {subj} cakap {verb} {obj}",
    "tunggu la sampai {obj} naik harga lagi",
]

_DIALECT_VARIANTS = {
    'aku': ['ambe', 'kawe', 'den', 'aku', 'gue'],
    'kau': ['mu', 'demo', 'kau', 'hang', 'lu'],
    'makan': ['make', 'makan', 'ngap', 'jamu'],
    'pergi': ['pegi', 'pi', 'poi', 'g'],
    'tidak': ['dok', 'tak', 'idok', 'dak', 'x'],
    'cantik': ['molek', 'lawa', 'cun', 'chantek'],
    'besar': ['beso', 'gedabak', 'besar', 'besaq'],
}

_FILLER_SENTENCES = [
    "aku rasa {obj} dekat {loc} paling {adj}",
    "korang pernah try {verb} {obj} tak?",
    "serious talk {obj} ni memang {adj}",
    "kalau nak {verb} {obj} kena pergi {loc}",
    "dah lama tak {verb} {obj} rindu gila",
    "weekend ni plan nak {verb} {obj}",
    "siapa ada recommendation {obj} {adj}?",
    "baru discover {obj} dekat {loc} {adj} gila",
    "everyday aku {verb} {obj} dah jadi routine",
    "honestly {obj} overrated tapi still {verb}",
]


def generate_corpus(n_sentences=2000, seed=42):
    """Generate a synthetic Manglish social media corpus.

    Creates tokenized sentences covering Twitter posts, Reddit comments,
    WhatsApp messages, and news comments with code-switching, slang,
    shortforms, and dialect variations.

    Parameters:
        n_sentences (int): Number of sentences to generate (default 2000).
        seed (int): Random seed for reproducibility.

    Returns:
        list[list[str]]: List of tokenized sentences (list of word lists).

    Example:
        >>> corpus = generate_corpus(100)
        >>> len(corpus)
        100
        >>> isinstance(corpus[0], list)
        True
    """
    random.seed(seed)
    corpus = []

    # Distribution: 30% Twitter, 20% Reddit, 30% WhatsApp, 15% News, 5% Filler
    distributions = [
        (_TWITTER_TEMPLATES, int(n_sentences * 0.30)),
        (_REDDIT_TEMPLATES, int(n_sentences * 0.20)),
        (_WHATSAPP_TEMPLATES, int(n_sentences * 0.30)),
        (_NEWS_COMMENT_TEMPLATES, int(n_sentences * 0.15)),
        (_FILLER_SENTENCES, int(n_sentences * 0.05)),
    ]

    for templates, count in distributions:
        for _ in range(count):
            sentence = _generate_sentence(templates)
            corpus.append(sentence)

    # Fill remaining
    while len(corpus) < n_sentences:
        templates = random.choice([
            _TWITTER_TEMPLATES, _WHATSAPP_TEMPLATES, _FILLER_SENTENCES
        ])
        corpus.append(_generate_sentence(templates))

    random.shuffle(corpus)
    return corpus


def _generate_sentence(templates):
    """Generate a single tokenized sentence from templates."""
    template = random.choice(templates)

    subj = random.choice(_SUBJECTS)
    verb = random.choice(_VERBS)
    obj = random.choice(_OBJECTS)
    obj2 = random.choice(_OBJECTS)
    adj = random.choice(_ADJECTIVES)
    particle = random.choice(_SLANG_PARTICLES)
    loc = random.choice(_LOCATIONS)

    # Apply dialect variants randomly (20% chance)
    if random.random() < 0.2:
        for word, variants in _DIALECT_VARIANTS.items():
            if word == subj:
                subj = random.choice(variants)
            if word == verb:
                verb = random.choice(variants)

    # Apply shortforms randomly (40% chance per word)
    sentence = template.format(
        subj=subj, verb=verb, obj=obj, obj2=obj2,
        adj=adj, particle=particle, loc=loc,
    )

    # Randomly apply shortforms
    if random.random() < 0.4:
        for full, short in _SHORTFORMS.items():
            if full in sentence and random.random() < 0.5:
                sentence = sentence.replace(full, short, 1)

    # Randomly add emoji (15% chance)
    if random.random() < 0.15:
        emojis = ['😂', '💀', '🔥', '😭', '🤣', '👍', '❤️', '😤', '🙄', '💯']
        sentence += ' ' + random.choice(emojis)

    # Randomly elongate words (10% chance)
    if random.random() < 0.1:
        words = sentence.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            w = words[idx]
            if len(w) > 2 and w.isalpha():
                words[idx] = w[:-1] + w[-1] * random.randint(2, 4)
            sentence = ' '.join(words)

    # Tokenize
    tokens = _tokenize_simple(sentence)
    return tokens


def _tokenize_simple(text):
    """Simple tokenizer for corpus generation."""
    # Split on whitespace and punctuation, keep meaningful tokens
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?|[^\s\w]", text.lower())
    # Filter very short noise but keep slang particles
    tokens = [t for t in tokens if len(t) >= 1 and (t.isalnum() or t in '?!.,')]
    return tokens


# ============================================================
# Model Training
# ============================================================

def train_word2vec(corpus=None, vector_size=100, window=5, min_count=2, sg=0):
    """Train a Word2Vec model on Manglish corpus.

    Parameters:
        corpus (list[list[str]]|None): Tokenized sentences. If None, generates corpus.
        vector_size (int): Dimensionality of word vectors (default 100).
        window (int): Context window size (default 5).
        min_count (int): Minimum word frequency (default 2).
        sg (int): 0 for CBOW, 1 for Skip-gram (default 0 = CBOW).

    Returns:
        gensim.models.Word2Vec: Trained model.

    Example:
        >>> model = train_word2vec(vector_size=50, sg=1)
        >>> 'makan' in model.wv
        True
    """
    _check_gensim()
    from gensim.models import Word2Vec

    if corpus is None:
        corpus = generate_corpus()

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=1,
        epochs=5,
        seed=42,
    )

    # Ensure resources directory exists
    os.makedirs(_RESOURCES_DIR, exist_ok=True)
    model.save(_W2V_PATH)

    return model


def train_fasttext(corpus=None, vector_size=100, window=5, min_count=2):
    """Train a FastText model on Manglish corpus.

    FastText handles OOV words via subword information, making it ideal
    for Manglish with its many shortforms and spelling variants.

    Parameters:
        corpus (list[list[str]]|None): Tokenized sentences. If None, generates corpus.
        vector_size (int): Dimensionality of word vectors (default 100).
        window (int): Context window size (default 5).
        min_count (int): Minimum word frequency (default 2).

    Returns:
        gensim.models.FastText: Trained model.

    Example:
        >>> model = train_fasttext(vector_size=50)
        >>> vec = model.wv['makan']
        >>> vec.shape
        (50,)
    """
    _check_gensim()
    from gensim.models import FastText

    if corpus is None:
        corpus = generate_corpus()

    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        epochs=5,
        seed=42,
        min_n=2,  # Minimum subword ngram
        max_n=4,  # Maximum subword ngram (reduced for memory)
    )

    # Ensure resources directory exists
    os.makedirs(_RESOURCES_DIR, exist_ok=True)
    model.save(_FT_PATH)

    return model


def train_all(vector_size=50, window=5, min_count=2, n_sentences=2000):
    """Train both Word2Vec and FastText models on shared corpus.

    Convenience function that generates corpus once and trains both models.

    Parameters:
        vector_size (int): Dimensionality of word vectors (default 50).
        window (int): Context window size.
        min_count (int): Minimum word frequency.
        n_sentences (int): Corpus size (default 2000).

    Returns:
        dict: {'word2vec': Word2Vec model, 'fasttext': FastText model, 'corpus_size': int}

    Example:
        >>> result = train_all(vector_size=50, n_sentences=1000)
        >>> 'word2vec' in result and 'fasttext' in result
        True
    """
    _check_gensim()

    corpus = generate_corpus(n_sentences=n_sentences)

    w2v_model = train_word2vec(corpus=corpus, vector_size=vector_size,
                               window=window, min_count=min_count, sg=0)
    ft_model = train_fasttext(corpus=corpus, vector_size=vector_size,
                              window=window, min_count=min_count)

    return {
        'word2vec': w2v_model,
        'fasttext': ft_model,
        'corpus_size': len(corpus),
    }


# ============================================================
# Model Loading
# ============================================================

def load_word2vec():
    """Load a previously trained Word2Vec model.

    Returns:
        gensim.models.Word2Vec: Loaded model.

    Raises:
        ImportError: If gensim is not installed.
        FileNotFoundError: If no trained model exists.

    Example:
        >>> model = load_word2vec()
        >>> type(model).__name__
        'Word2Vec'
    """
    _check_gensim()
    from gensim.models import Word2Vec

    if not os.path.exists(_W2V_PATH):
        raise FileNotFoundError(
            f"No Word2Vec model found at {_W2V_PATH}. "
            "Train one first with train_word2vec() or train_all()."
        )

    return Word2Vec.load(_W2V_PATH)


def load_fasttext():
    """Load a previously trained FastText model.

    Returns:
        gensim.models.FastText: Loaded model.

    Raises:
        ImportError: If gensim is not installed.
        FileNotFoundError: If no trained model exists.

    Example:
        >>> model = load_fasttext()
        >>> type(model).__name__
        'FastText'
    """
    _check_gensim()
    from gensim.models import FastText

    if not os.path.exists(_FT_PATH):
        raise FileNotFoundError(
            f"No FastText model found at {_FT_PATH}. "
            "Train one first with train_fasttext() or train_all()."
        )

    return FastText.load(_FT_PATH)


# ============================================================
# Query Functions
# ============================================================

def _get_model(model_type='fasttext'):
    """Get the appropriate model by type."""
    if model_type == 'fasttext':
        return load_fasttext()
    elif model_type in ('word2vec', 'w2v'):
        return load_word2vec()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'fasttext' or 'word2vec'.")


def most_similar(word, model_type='fasttext', topn=10):
    """Find words most similar to the given word.

    Parameters:
        word (str): Query word.
        model_type (str): 'fasttext' or 'word2vec' (default 'fasttext').
        topn (int): Number of similar words to return (default 10).

    Returns:
        list[tuple[str, float]]: List of (word, similarity_score) pairs.

    Example:
        >>> results = most_similar('makan', topn=5)
        >>> isinstance(results, list)
        True
        >>> len(results) <= 5
        True
    """
    model = _get_model(model_type)
    try:
        return model.wv.most_similar(word.lower(), topn=topn)
    except KeyError:
        return []


def word_vector(word, model_type='fasttext'):
    """Get the vector representation of a word.

    Parameters:
        word (str): Input word.
        model_type (str): 'fasttext' or 'word2vec' (default 'fasttext').

    Returns:
        numpy.ndarray: Word vector.

    Raises:
        KeyError: If word not in vocabulary (Word2Vec only; FastText handles OOV).

    Example:
        >>> vec = word_vector('makan')
        >>> isinstance(vec, np.ndarray)
        True
        >>> vec.shape[0] > 0
        True
    """
    model = _get_model(model_type)
    return model.wv[word.lower()]


def sentence_vector(text, model_type='fasttext'):
    """Get averaged word vector for a sentence/text.

    Computes the mean of all word vectors in the text.

    Parameters:
        text (str): Input text.
        model_type (str): 'fasttext' or 'word2vec' (default 'fasttext').

    Returns:
        numpy.ndarray: Averaged sentence vector.

    Example:
        >>> vec = sentence_vector('aku nak makan nasi lemak')
        >>> isinstance(vec, np.ndarray)
        True
    """
    model = _get_model(model_type)
    tokens = _tokenize_simple(text)

    vectors = []
    for token in tokens:
        try:
            vectors.append(model.wv[token])
        except KeyError:
            continue

    if not vectors:
        # Return zero vector with model's vector size
        return np.zeros(model.wv.vector_size)

    return np.mean(vectors, axis=0)


def analogy(positive, negative, model_type='fasttext', topn=5):
    """Solve word analogy (e.g., king - man + woman = queen).

    Parameters:
        positive (list[str]): Words that contribute positively.
        negative (list[str]): Words that contribute negatively.
        model_type (str): 'fasttext' or 'word2vec' (default 'fasttext').
        topn (int): Number of results (default 5).

    Returns:
        list[tuple[str, float]]: List of (word, score) pairs.

    Example:
        >>> # makan - nasi + roti = ?
        >>> results = analogy(['makan', 'roti'], ['nasi'], topn=3)
        >>> isinstance(results, list)
        True
    """
    model = _get_model(model_type)
    try:
        return model.wv.most_similar(
            positive=[w.lower() for w in positive],
            negative=[w.lower() for w in negative],
            topn=topn,
        )
    except KeyError as e:
        return []
