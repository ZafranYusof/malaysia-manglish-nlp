"""Dictionary utilities - word validation and lookup.

Inspired by malaya.dictionary (is_malay, is_english).
"""

import re
from manglish_nlp.utils import get_shortforms
from manglish_nlp.stemmer import stem_word

# Lazy loading state
_loaded = False
_BM_WORDS = None
_EN_WORDS = None

# BM affixes that indicate a Malay word
_BM_PREFIXES = ('me', 'ber', 'ter', 'di', 'ke', 'se', 'pe')
_BM_SUFFIXES = ('kan', 'an', 'nya', 'lah', 'kah')


def _ensure_loaded():
    """Load word lists on first access."""
    global _loaded, _BM_WORDS, _EN_WORDS
    if _loaded:
        return
    
    _BM_WORDS = _build_bm_words()
    _EN_WORDS = _build_en_words()
    _loaded = True


def _build_bm_words():
    """Build the BM word set."""
    return {
    # Pronouns
    'saya', 'aku', 'awak', 'kamu', 'anda', 'dia', 'beliau', 'mereka',
    'kami', 'kita', 'engkau', 'hamba',
    # Determiners
    'ini', 'itu', 'si', 'sang', 'para', 'semua', 'setiap', 'beberapa',
    # Prepositions
    'di', 'ke', 'dari', 'pada', 'untuk', 'dengan', 'dalam', 'luar',
    'atas', 'bawah', 'depan', 'belakang', 'antara', 'oleh', 'tanpa',
    'tentang', 'terhadap', 'sejak', 'hingga', 'sampai', 'melalui',
    # Conjunctions
    'dan', 'atau', 'tetapi', 'tapi', 'namun', 'serta', 'mahupun',
    'kerana', 'sebab', 'kalau', 'jika', 'walaupun', 'meskipun',
    'supaya', 'agar', 'sehingga', 'manakala', 'sedangkan',
    # Verbs
    'ada', 'adalah', 'ialah', 'menjadi', 'jadi',
    'pergi', 'datang', 'balik', 'pulang', 'tiba', 'sampai',
    'makan', 'minum', 'tidur', 'bangun', 'duduk', 'berdiri',
    'berjalan', 'berlari', 'berenang', 'terbang', 'melompat',
    'buat', 'kerja', 'belajar', 'mengajar', 'menulis', 'membaca',
    'berkata', 'bercakap', 'mendengar', 'melihat', 'merasa',
    'berfikir', 'mengetahui', 'memahami', 'mengingat', 'melupakan',
    'memberi', 'menerima', 'mengambil', 'meletakkan', 'menghantar',
    'membeli', 'menjual', 'membayar', 'meminjam', 'menyimpan',
    'membuka', 'menutup', 'memulakan', 'mengakhiri', 'meneruskan',
    'suka', 'sayang', 'cinta', 'benci', 'takut', 'malu', 'marah',
    'boleh', 'dapat', 'perlu', 'harus', 'mesti', 'mahu', 'hendak',
    'akan', 'telah', 'sudah', 'sedang', 'masih', 'belum', 'pernah',
    # Adjectives
    'baik', 'buruk', 'besar', 'kecil', 'tinggi', 'rendah',
    'panjang', 'pendek', 'lebar', 'sempit', 'tebal', 'nipis',
    'baru', 'lama', 'muda', 'tua', 'cantik', 'hodoh', 'tampan',
    'pandai', 'bijak', 'bodoh', 'rajin', 'malas', 'tekun',
    'kaya', 'miskin', 'murah', 'mahal', 'percuma',
    'senang', 'susah', 'mudah', 'sukar', 'cepat', 'lambat',
    'dekat', 'jauh', 'dalam', 'cetek', 'panas', 'sejuk',
    'basah', 'kering', 'bersih', 'kotor', 'terang', 'gelap',
    'keras', 'lembut', 'kasar', 'halus', 'licin', 'kasap',
    'sedap', 'manis', 'masam', 'pahit', 'masin', 'pedas',
    'putih', 'hitam', 'merah', 'biru', 'hijau', 'kuning',
    'coklat', 'kelabu', 'ungu', 'jingga', 'emas', 'perak',
    # Nouns
    'orang', 'manusia', 'lelaki', 'perempuan', 'anak', 'budak',
    'ibu', 'bapa', 'adik', 'abang', 'kakak', 'datuk', 'nenek',
    'kawan', 'sahabat', 'musuh', 'jiran', 'guru', 'murid', 'pelajar',
    'doktor', 'polis', 'tentera', 'hakim', 'peguam', 'jurutera',
    'rumah', 'sekolah', 'universiti', 'hospital', 'masjid', 'kedai',
    'pejabat', 'kilang', 'ladang', 'sawah', 'kebun', 'taman',
    'jalan', 'lorong', 'lebuh', 'lebuhraya', 'jambatan', 'terowong',
    'kereta', 'motosikal', 'bas', 'lori', 'kapal', 'bot', 'pesawat',
    'telefon', 'komputer', 'televisyen', 'radio', 'mesin',
    'buku', 'surat', 'kertas', 'pen', 'pensel', 'meja', 'kerusi',
    'makanan', 'minuman', 'nasi', 'roti', 'mi', 'kuih',
    'air', 'susu', 'kopi', 'teh', 'jus',
    'ayam', 'ikan', 'daging', 'udang', 'ketam', 'telur',
    'sayur', 'buah', 'beras', 'gula', 'garam', 'minyak',
    'duit', 'wang', 'ringgit', 'sen', 'harga', 'gaji', 'untung',
    'hari', 'minggu', 'bulan', 'tahun', 'masa', 'waktu', 'jam', 'minit',
    'pagi', 'tengahari', 'petang', 'malam', 'subuh', 'maghrib',
    'negara', 'negeri', 'bandar', 'kampung', 'desa', 'dunia',
    # Adverbs
    'sangat', 'amat', 'terlalu', 'agak', 'kurang', 'lebih', 'paling',
    'selalu', 'sering', 'kadang', 'jarang', 'tidak', 'bukan', 'jangan',
    'juga', 'pun', 'lagi', 'sahaja', 'hanya', 'memang', 'sungguh',
    # Numbers
    'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh',
    'lapan', 'sembilan', 'sepuluh', 'sebelas', 'seratus', 'seribu',
    'pertama', 'kedua', 'ketiga',
    # Question words
    'apa', 'siapa', 'mana', 'bila', 'kenapa', 'mengapa', 'bagaimana', 'berapa',
}


def _build_en_words():
    """Build the EN word set."""
    return {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'any', 'no', 'not', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'also', 'still', 'already', 'always', 'never',
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although',
    'if', 'unless', 'until', 'while', 'since', 'after', 'before',
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about',
    'into', 'through', 'during', 'between', 'against', 'without',
    'go', 'come', 'get', 'make', 'take', 'give', 'say', 'tell', 'think',
    'know', 'see', 'want', 'need', 'like', 'love', 'hate', 'try', 'help',
    'work', 'play', 'run', 'walk', 'eat', 'drink', 'sleep', 'read', 'write',
    'good', 'bad', 'big', 'small', 'new', 'old', 'long', 'short', 'great',
    'little', 'right', 'wrong', 'high', 'low', 'young', 'last', 'next',
    'man', 'woman', 'child', 'people', 'time', 'year', 'day', 'way',
    'world', 'life', 'hand', 'part', 'place', 'case', 'week', 'company',
    'system', 'program', 'question', 'work', 'government', 'number', 'night',
    'point', 'home', 'water', 'room', 'mother', 'area', 'money', 'story',
    'fact', 'month', 'lot', 'right', 'study', 'book', 'eye', 'job', 'word',
    'business', 'issue', 'side', 'kind', 'head', 'house', 'service', 'friend',
    'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'city',
    'community', 'name', 'president', 'team', 'minute', 'idea', 'body',
    'information', 'back', 'parent', 'face', 'others', 'level', 'office',
    'door', 'health', 'person', 'art', 'war', 'history', 'party', 'result',
    'computer', 'phone', 'internet', 'email', 'website', 'software', 'hardware',
    'data', 'file', 'server', 'network', 'database', 'application', 'program',
    'student', 'teacher', 'doctor', 'engineer', 'manager', 'driver',
    'food', 'water', 'coffee', 'tea', 'rice', 'chicken', 'fish', 'bread',
    'car', 'bus', 'train', 'plane', 'road', 'building', 'school', 'hospital',
    'morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow', 'yesterday',
    'happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty', 'sick', 'healthy',
    'beautiful', 'ugly', 'smart', 'stupid', 'fast', 'slow', 'easy', 'hard',
    'cheap', 'expensive', 'free', 'busy', 'ready', 'sorry', 'sure', 'maybe',
}


def is_malay(word):
    """Check if a word is a Malay word.
    
    Uses dictionary lookup + morphological analysis.
    
    Parameters:
        word (str): Word to check.
    
    Returns:
        bool: True if likely a Malay word.
    
    Example:
        >>> is_malay("makan")
        True
        >>> is_malay("computer")
        False
        >>> is_malay("berlari")
        True
    """
    _ensure_loaded()
    lower = word.lower()
    
    # Direct lookup
    if lower in _BM_WORDS:
        return True
    
    # Check shortforms
    if lower in get_shortforms():
        return True
    
    # Check if stemmed form is in dictionary
    root = stem_word(lower)
    if root in _BM_WORDS:
        return True
    
    # Morphological check: has BM affixes
    for prefix in _BM_PREFIXES:
        if lower.startswith(prefix) and len(lower) > len(prefix) + 2:
            return True
    
    for suffix in _BM_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
            return True
    
    return False

def is_english(word):
    """Check if a word is an English word.
    
    Parameters:
        word (str): Word to check.
    
    Returns:
        bool: True if likely an English word.
    
    Example:
        >>> is_english("computer")
        True
        >>> is_english("makan")
        False
    """
    _ensure_loaded()
    lower = word.lower()
    
    if lower in _EN_WORDS:
        return True
    
    # Common EN suffixes
    en_suffixes = ('tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful',
                   'less', 'ous', 'ive', 'ing', 'ated', 'ize', 'ise', 'ally')
    for suffix in en_suffixes:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
            return True
    
    return False

def classify_word(word):
    """Classify a word as BM, EN, both, or unknown.
    
    Parameters:
        word (str): Word to classify.
    
    Returns:
        dict: Result with 'word', 'classification', 'is_malay', 'is_english'.
    
    Example:
        >>> classify_word("makan")
        {'word': 'makan', 'classification': 'bm', 'is_malay': True, 'is_english': False}
        >>> classify_word("hospital")
        {'word': 'hospital', 'classification': 'both', 'is_malay': True, 'is_english': True}
    """
    bm = is_malay(word)
    en = is_english(word)
    
    if bm and en:
        classification = 'both'
    elif bm:
        classification = 'bm'
    elif en:
        classification = 'en'
    else:
        classification = 'unknown'
    
    return {
        'word': word,
        'classification': classification,
        'is_malay': bm,
        'is_english': en,
    }

def get_stopwords(lang='bm'):
    """Get stop words list.
    
    Parameters:
        lang (str): Language - 'bm', 'en', or 'all'.
    
    Returns:
        set: Stop words.
    """
    bm_stops = {
        'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'ada', 'adalah',
        'untuk', 'dengan', 'pada', 'oleh', 'akan', 'telah', 'sudah',
        'sedang', 'masih', 'juga', 'pun', 'lagi', 'sahaja', 'hanya',
        'tidak', 'tak', 'bukan', 'belum', 'atau', 'tetapi', 'tapi',
        'dalam', 'luar', 'atas', 'bawah', 'antara', 'seperti',
        'saya', 'aku', 'awak', 'kamu', 'dia', 'mereka', 'kami', 'kita',
        'sangat', 'amat', 'terlalu', 'agak', 'kurang', 'lebih',
        'semua', 'setiap', 'banyak', 'sedikit', 'beberapa',
        'boleh', 'perlu', 'harus', 'mesti', 'mahu', 'nak',
    }
    
    en_stops = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'my', 'your', 'his', 'its', 'our', 'their',
        'this', 'that', 'these', 'those',
        'and', 'but', 'or', 'not', 'no', 'if', 'so',
        'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of',
        'very', 'just', 'also', 'too', 'only',
    }
    
    if lang == 'bm':
        return bm_stops
    elif lang == 'en':
        return en_stops
    else:
        return bm_stops | en_stops
