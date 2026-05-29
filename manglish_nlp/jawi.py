"""
Jawi (Arabic script for Malay) conversion module.

Provides Rumi (Latin) to Jawi conversion and vice versa,
with support for Malay-specific characters (ca, pa, ga, nga, nya, va)
and common Malay word dictionary for accurate conversion.
"""

import re
from typing import List, Optional, Tuple

# ==============================================================================
# Jawi Character Constants
# ==============================================================================

# Malay-specific Jawi letters (not in standard Arabic)
CA = '\u0686'    # چ ca
PA = '\u067E'    # ڤ pa
NGA = '\u06A0'   # ڠ nga
GA = '\u06A2'    # ݢ ga
NYA = '\u06BD'   # ڽ nya
VA = '\u06CF'    # ۏ va
KAF = '\u06A9'   # ک kaf (Malay style)

# Standard Arabic letters used in Jawi
ALIF = '\u0627'  # ا
BA = '\u0628'    # ب
TA = '\u062A'    # ت
THA = '\u062B'   # ث
JIM = '\u062C'   # ج
HA_HET = '\u062D'  # ح (Arabic ha)
KHA = '\u062E'   # خ
DAL = '\u062F'   # د
ZAL = '\u0630'   # ذ
RA = '\u0631'    # ر
ZAI = '\u0632'   # ز
SIN = '\u0633'   # س
SYIN = '\u0634'  # ش
SAD = '\u0635'   # ص
DAD = '\u0636'   # ض
TA_MARBUTA = '\u0637'  # ط (Arabic ta)
ZA = '\u0638'    # ظ
AIN = '\u0639'   # ع
GHAIN = '\u063A'  # غ
FA = '\u0641'    # ف
QA = '\u0642'    # ق
LAM = '\u0644'   # ل
MIM = '\u0645'   # م
NUN = '\u0646'   # ن
HA = '\u0647'    # ه
WAW = '\u0648'   # و
YA = '\u064A'    # ي
YE = '\u0649'    # ى

# Diacritics
FATHAH = '\u064E'   # a
KASRA = '\u0650'    # i
DAMMA = '\u064F'    # u
SUKUN = '\u0652'    # dead consonant

# ==============================================================================
# Mapping Tables
# ==============================================================================

# Consonant mapping: Rumi -> Jawi
CONSONANT_MAP = {
    'b': BA, 'c': CA, 'd': DAL, 'f': FA, 'g': GA,
    'h': HA, 'j': JIM, 'k': KAF, 'l': LAM, 'm': MIM,
    'n': NUN, 'p': PA, 'q': QA, 'r': RA, 's': SIN,
    't': TA, 'v': VA, 'w': WAW, 'x': KHA + SIN,  # "ks" sound
    'y': YA, 'z': ZAI,
    'sy': SYIN, 'ng': NGA, 'ny': NYA, 'kh': KHA,
    'gh': GHAIN, 'th': THA, 'sh': SYIN, 'dh': ZAL,
    'ph': FA,
}

# Reverse consonant mapping: Jawi -> Rumi
REVERSE_CONSONANT = {
    BA: 'b', TA: 't', THA: 'th', JIM: 'j', HA_HET: 'h',
    KHA: 'kh', DAL: 'd', ZAL: 'dh', RA: 'r', ZAI: 'z',
    SIN: 's', SYIN: 'sy', SAD: 's', DAD: 'd', TA_MARBUTA: 't',
    ZA: 'z', AIN: 'a', GHAIN: 'gh', FA: 'f', QA: 'q',
    KAF: 'k', LAM: 'l', MIM: 'm', NUN: 'n', HA: 'h',
    WAW: 'w', YA: 'y', YE: 'i',
    CA: 'c', PA: 'p', NGA: 'ng', GA: 'g', NYA: 'ny', VA: 'v',
}

# Malay word dictionary: common words with known correct Jawi spellings
# This ensures accuracy for frequently-used words that may not follow
# simple letter-by-letter rules.
MALAY_JAWI_DICT = {
    # Basic verbs
    'makan': f'{MIM}{ALIF}{KAF}{ALIF}{NUN}',
    'minum': f'{MIM}{YE}{NUN}{DAMMA}{MIM}',
    'pergi': f'{PA}{KASRA}{RA}{GA}{YE}',
    'datang': f'{DAL}{ALIF}{TA}{ALIF}{NGA}',
    'buat': f'{BA}{WAW}{ALIF}{TA}',
    'tahu': f'{TA}{ALIF}{HA}{WAW}',
    'mahu': f'{MIM}{ALIF}{HA}{WAW}',
    'nak': f'{NUN}{ALIF}{KAF}',
    'boleh': f'{BA}{DAMMA}{LAM}{YE}{HA}',
    'ada': f'{ALIF}{DAL}{ALIF}',
    'jadi': f'{JIM}{ALIF}{DAL}{YE}',
    'dapat': f'{DAL}{ALIF}{PA}{ALIF}{TA}',
    'nak': f'{NUN}{ALIF}{KAF}',
    'tengok': f'{TA}{FATHAH}{NGA}{DAMMA}{KAF}',
    'cakap': f'{CA}{ALIF}{KAF}{ALIF}{PA}',
    'dengar': f'{DAL}{FATHAH}{NGA}{KASRA}{RA}',
    'tulis': f'{TA}{WAW}{LAM}{YE}{SIN}',
    'baca': f'{BA}{ALIF}{CA}{ALIF}',
    'beli': f'{BA}{FATHAH}{LAM}{YE}',
    'jual': f'{JIM}{WAW}{ALIF}{LAM}',
    'bayar': f'{BA}{ALIF}{YA}{RA}',
    'kerja': f'{KAF}{KASRA}{RA}{JIM}{ALIF}',
    'tidur': f'{TA}{YE}{DAL}{WAW}{RA}',
    'bangun': f'{BA}{ALIF}{NGA}{WAW}{NUN}',
    'duduk': f'{DAL}{WAW}{DAL}{WAW}{KAF}',
    'jalan': f'{JIM}{ALIF}{LAM}{ALIF}{NUN}',
    'main': f'{MIM}{ALIF}{YE}{NUN}',
    'masak': f'{MIM}{ALIF}{SIN}{ALIF}{KAF}',
    'basuh': f'{BA}{ALIF}{SIN}{WAW}{HA}',
    'cuci': f'{CA}{WAW}{CA}{YE}',
    'suka': f'{SIN}{WAW}{KAF}{ALIF}',
    'sakit': f'{SIN}{ALIF}{KAF}{YE}{TA}',
    'sihat': f'{SIN}{YE}{HA}{ALIF}{TA}',
    'tunggu': f'{TA}{WAW}{NGA}{WAW}',
    'hantar': f'{HA}{ALIF}{NUN}{TA}{RA}',
    'ambil': f'{ALIF}{MIM}{BA}{YE}{LAM}',
    'bagi': f'{BA}{ALIF}{GA}{YE}',
    'beri': f'{BA}{KASRA}{RA}{YE}',
    'cari': f'{CA}{ALIF}{RA}{YE}',
    'guna': f'{GA}{WAW}{NUN}{ALIF}',
    'hantar': f'{HA}{ALIF}{NUN}{TA}{RA}',
    'ingat': f'{ALIF}{NGA}{ALIF}{TA}',
    'kata': f'{KAF}{ALIF}{TA}{ALIF}',
    'kenal': f'{KAF}{FATHAH}{NUN}{ALIF}{LAM}',
    'layan': f'{LAM}{ALIF}{YA}{ALIF}{NUN}',
    'masa': f'{MIM}{ALIF}{SIN}{ALIF}',
    'nama': f'{NUN}{ALIF}{MIM}{ALIF}',
    'orang': f'{ALIF}{WAW}{RA}{ALIF}{NGA}',
    'pulang': f'{PA}{WAW}{LAM}{ALIF}{NGA}',
    'rumah': f'{RA}{WAW}{MIM}{ALIF}{HA}',
    'sekolah': f'{SIN}{KAF}{WAW}{LAM}{ALIF}{HA}',
    'tahun': f'{TA}{HA}{WAW}{NUN}',
    'untuk': f'{WAW}{NUN}{TA}{WAW}{KAF}',
    'wang': f'{WAW}{ALIF}{NGA}',
    'banyak': f'{BA}{ALIF}{NYA}{KAF}',
    'besar': f'{BA}{KASRA}{SIN}{ALIF}{RA}',
    'cantik': f'{CA}{ALIF}{NUN}{TA}{YE}{KAF}',
    'cepat': f'{CA}{FATHAH}{PA}{ALIF}{TA}',
    'dalam': f'{DAL}{ALIF}{LAM}{MIM}',
    'dekat': f'{DAL}{KASRA}{KAF}{ALIF}{TA}',
    'dia': f'{DAL}{YE}{ALIF}',
    'hanya': f'{HA}{ALIF}{NYA}{ALIF}',
    'hari': f'{HA}{ALIF}{RA}{YE}',
    'ini': f'{ALIF}{YE}{NUN}{YE}',
    'itu': f'{ALIF}{YE}{TA}{WAW}',
    'juga': f'{JIM}{WAW}{GA}{ALIF}',
    'jauh': f'{JIM}{ALIF}{WAW}{HA}',
    'kecil': f'{KAF}{FATHAH}{CA}{YE}{LAM}',
    'lama': f'{LAM}{ALIF}{MIM}{ALIF}',
    'lari': f'{LAM}{ALIF}{RA}{YE}',
    'malam': f'{MIM}{ALIF}{LAM}{MIM}',
    'mana': f'{MIM}{ALIF}{NUN}{ALIF}',
    'pagi': f'{PA}{ALIF}{GA}{YE}',
    'sama': f'{SIN}{ALIF}{MIM}{ALIF}',
    'sangat': f'{SIN}{ALIF}{NGA}{ALIF}{TA}',
    'saya': f'{SIN}{ALIF}{YA}{ALIF}',
    'semua': f'{SIN}{KASRA}{MIM}{WAW}{ALIF}',
    'sini': f'{SIN}{YE}{NUN}{YE}',
    'situ': f'{SIN}{YE}{TA}{WAW}',
    'sudah': f'{SIN}{WAW}{DAL}{ALIF}{HA}',
    'tapi': f'{TA}{ALIF}{PA}{YE}',
    'tapi': f'{TA}{ALIF}{PA}{YE}',
    'tidak': f'{TA}{YE}{DAL}{ALIF}{KAF}',
    'sudah': f'{SIN}{WAW}{DAL}{ALIF}{HA}',
    'dengan': f'{DAL}{FATHAH}{NGA}{ALIF}{NUN}',
    'dari': f'{DAL}{ALIF}{RA}{YE}',
    'dan': f'{DAL}{ALIF}{NUN}',
    'yang': f'{YA}{ALIF}{NGA}',
    'di': f'{DAL}{YE}',
    'ke': f'{KAF}{FATHAH}',
    'pada': f'{PA}{ALIF}{DAL}{ALIF}',
    'atau': f'{ALIF}{TA}{ALIF}{WAW}',
    'ini': f'{ALIF}{YE}{NUN}{YE}',
    'itu': f'{ALIF}{YE}{TA}{WAW}',
    'sangat': f'{SIN}{ALIF}{NGA}{ALIF}{TA}',
    'jangan': f'{JIM}{ALIF}{NGA}{ALIF}{NUN}',
    'sila': f'{SIN}{YE}{LAM}{ALIF}',
    'tolong': f'{TA}{WAW}{LAM}{DAMMA}{NGA}',
    'terima': f'{TA}{KASRA}{RA}{YE}{MIM}{ALIF}',
    'kasih': f'{KAF}{ALIF}{SIN}{YE}{HA}',
    'sayang': f'{SIN}{ALIF}{YA}{ALIF}{NGA}',
    'cinta': f'{CA}{YE}{NUN}{TA}{ALIF}',
    'baik': f'{BA}{ALIF}{YE}{KAF}',
    'buruk': f'{BA}{WAW}{RA}{WAW}{KAF}',
    'baru': f'{BA}{ALIF}{RA}{WAW}',
    'lama': f'{LAM}{ALIF}{MIM}{ALIF}',
    'tinggi': f'{TA}{YE}{NGA}{GA}{YE}',
    'rendah': f'{RA}{FATHAH}{NUN}{DAL}{ALIF}{HA}',
    'panas': f'{PA}{ALIF}{NUN}{ALIF}{SIN}',
    'sejuk': f'{SIN}{KASRA}{JIM}{WAW}{KAF}',
    'hujan': f'{HA}{WAW}{JIM}{ALIF}{NUN}',
    'angin': f'{ALIF}{NGA}{YE}{NUN}',
    'air': f'{ALIF}{YE}{RA}',
    'api': f'{ALIF}{PA}{YE}',
    'tanah': f'{TA}{ALIF}{NUN}{ALIF}{HA}',
    'langit': f'{LAM}{ALIF}{NGA}{YE}{TA}',
    'bulan': f'{BA}{WAW}{LAM}{ALIF}{NUN}',
    'bintang': f'{BA}{YE}{NUN}{TA}{ALIF}{NGA}',
    'matahari': f'{MIM}{ALIF}{TA}{ALIF}{HA}{ALIF}{RA}{YE}',
    'kereta': f'{KAF}{KASRA}{TA}{ALIF}',
    'bas': f'{BA}{ALIF}{SIN}',
    'kapal': f'{KAF}{ALIF}{PA}{ALIF}{LAM}',
    'jalan': f'{JIM}{ALIF}{LAM}{ALIF}{NUN}',
    'kaki': f'{KAF}{ALIF}{KAF}{YE}',
    'tangan': f'{TA}{ALIF}{NGA}{ALIF}{NUN}',
    'kepala': f'{KAF}{FATHAH}{PA}{ALIF}{LAM}{ALIF}',
    'mata': f'{MIM}{ALIF}{TA}{ALIF}',
    'hidung': f'{HA}{YE}{DAL}{WAW}{NGA}',
    'mulut': f'{MIM}{WAW}{LAM}{WAW}{TA}',
    'telinga': f'{TA}{FATHAH}{LAM}{YE}{NGA}{ALIF}',
    'perut': f'{PA}{KASRA}{RA}{WAW}{TA}',
    'buku': f'{BA}{WAW}{KAF}{WAW}',
    'pensel': f'{PA}{FATHAH}{NUN}{SIN}{FATHAH}{LAM}',
    'pen': f'{PA}{FATHAH}{NUN}',
    'meja': f'{MIM}{FATHAH}{JIM}{ALIF}',
    'kerusi': f'{KAF}{KASRA}{RA}{WAW}{SIN}{YE}',
    'pintu': f'{PA}{YE}{NUN}{TA}{WAW}',
    'tingkap': f'{TA}{YE}{NGA}{KAF}{ALIF}{PA}',
    'bilik': f'{BA}{YE}{LAM}{YE}{KAF}',
    'dapur': f'{DAL}{ALIF}{PA}{WAW}{RA}',
    'tandas': f'{TA}{ALIF}{NUN}{DAL}{ALIF}{SIN}',
    'kedai': f'{KAF}{FATHAH}{DAL}{ALIF}{YE}',
    'pasar': f'{PA}{ALIF}{SIN}{ALIF}{RA}',
    'hospital': f'{HA}{WAW}{SIN}{PA}{YE}{TA}{ALIF}{LAM}',
    'masjid': f'{MIM}{ALIF}{SIN}{JIM}{YE}{DAL}',
    'gereja': f'{GA}{KASRA}{RA}{JIM}{ALIF}',
    'kuil': f'{KAF}{WAW}{YE}{LAM}',
    'candi': f'{CA}{ALIF}{NUN}{DAL}{YE}',
    'istana': f'{ALIF}{YE}{SIN}{TA}{ALIF}{NUN}{ALIF}',
    'negara': f'{NUN}{KASRA}{GA}{ALIF}{RA}{ALIF}',
    'bandar': f'{BA}{ALIF}{NUN}{DAL}{RA}',
    'kampung': f'{KAF}{ALIF}{MIM}{PA}{WAW}{NGA}',
    'pekan': f'{PA}{KASRA}{KAF}{ALIF}{NUN}',
    'pulau': f'{PA}{WAW}{LAM}{ALIF}{WAW}',
    'gunung': f'{GA}{WAW}{NUN}{WAW}{NGA}',
    'sungai': f'{SIN}{WAW}{NGA}{ALIF}{YE}',
    'laut': f'{LAM}{ALIF}{WAW}{TA}',
    'pantai': f'{PA}{ALIF}{NUN}{TA}{ALIF}{YE}',
    # Pronouns
    'aku': f'{ALIF}{KAF}{WAW}',
    'kau': f'{KAF}{ALIF}{WAW}',
    'kamu': f'{KAF}{ALIF}{MIM}{WAW}',
    'kami': f'{KAF}{ALIF}{MIM}{YE}',
    'mereka': f'{MIM}{KASRA}{RA}{KAF}{ALIF}',
    'anda': f'{ALIF}{NUN}{DAL}{ALIF}',
    # Common function words
    'adalah': f'{ALIF}{DAL}{ALIF}{LAM}{ALIF}{HA}',
    'akan': f'{ALIF}{KAF}{ALIF}{NUN}',
    'telah': f'{TA}{FATHAH}{LAM}{ALIF}{HA}',
    'sudah': f'{SIN}{WAW}{DAL}{ALIF}{HA}',
    'belum': f'{BA}{FATHAH}{LAM}{WAW}{MIM}',
    'sedang': f'{SIN}{FATHAH}{DAL}{ALIF}{NGA}',
    'masih': f'{MIM}{ALIF}{SIN}{YE}{HA}',
    'juga': f'{JIM}{WAW}{GA}{ALIF}',
    'pula': f'{PA}{WAW}{LAM}{ALIF}',
    'lagi': f'{LAM}{ALIF}{GA}{YE}',
    'sahaja': f'{SIN}{ALIF}{HA}{ALIF}{JIM}{ALIF}',
    'saja': f'{SIN}{ALIF}{JIM}{ALIF}',
    'kalau': f'{KAF}{ALIF}{LAM}{ALIF}{WAW}',
    'jika': f'{JIM}{YE}{KAF}{ALIF}',
    'kerana': f'{KAF}{KASRA}{RA}{ALIF}{NUN}{ALIF}',
    'supaya': f'{SIN}{WAW}{PA}{ALIF}{YA}{ALIF}',
    'agar': f'{ALIF}{GA}{RA}',
    'biar': f'{BA}{YE}{ALIF}{RA}',
    'biarkan': f'{BA}{YE}{ALIF}{RA}{KAF}{ALIF}{NUN}',
    'tentang': f'{TA}{FATHAH}{NUN}{TA}{ALIF}{NGA}',
    'antara': f'{ALIF}{NUN}{TA}{ALIF}{RA}{ALIF}',
    'sebelum': f'{SIN}{KASRA}{BA}{FATHAH}{LAM}{WAW}{MIM}',
    'selepas': f'{SIN}{KASRA}{LAM}{KASRA}{PA}{ALIF}{SIN}',
    'sejak': f'{SIN}{KASRA}{JIM}{ALIF}{KAF}',
    'semasa': f'{SIN}{KASRA}{MIM}{ALIF}{SIN}{ALIF}',
    'ketika': f'{KAF}{FATHAH}{TA}{YE}{KAF}{ALIF}',
    'apabila': f'{ALIF}{PA}{ALIF}{BA}{YE}{LAM}{ALIF}',
    'walaupun': f'{WAW}{ALIF}{LAM}{ALIF}{PA}{WAW}{NUN}',
    'meskipun': f'{MIM}{FATHAH}{SIN}{KAF}{YE}{PA}{WAW}{NUN}',
    'namun': f'{NUN}{ALIF}{MIM}{WAW}{NUN}',
    'tetapi': f'{TA}{FATHAH}{TA}{ALIF}{PA}{YE}',
}

# Reverse dictionary for Jawi -> Rumi
JAWI_MALAY_DICT = {v: k for k, v in MALAY_JAWI_DICT.items()}

# Common prefixes and suffixes for morphological handling
PREFIXES = ['me', 'mem', 'men', 'meng', 'meny', 'memper', 'ber', 'per', 'pem',
            'pen', 'peng', 'peny', 'pe', 'se', 'ter', 'di', 'ke']
SUFFIXES = ['kan', 'an', 'i', 'lah', 'nya', 'kah']

# All Jawi characters set for detection
JAWI_CHARS = set(
    ALIF + BA + TA + THA + JIM + HA_HET + KHA + DAL + ZAL + RA + ZAI +
    SIN + SYIN + SAD + DAD + TA_MARBUTA + ZA + AIN + GHAIN + FA + QA +
    KAF + LAM + MIM + NUN + HA + WAW + YA + YE +
    CA + PA + NGA + GA + NYA + VA +
    FATHAH + KASRA + DAMMA + SUKUN
)

# ==============================================================================
# Core Functions
# ==============================================================================


def detect_script(text: str) -> str:
    """Detect if text is Rumi (Latin), Jawi (Arabic), or mixed.

    Args:
        text: Input text to analyze.

    Returns:
        'rumi' if Latin script, 'jawi' if Arabic script, 'mixed' if both,
        'unknown' if neither detected.
    """
    if not text or not text.strip():
        return 'unknown'

    has_latin = bool(re.search(r'[a-zA-Z]', text))
    has_jawi = any(ch in JAWI_CHARS for ch in text)

    if has_latin and has_jawi:
        return 'mixed'
    elif has_latin:
        return 'rumi'
    elif has_jawi:
        return 'jawi'
    return 'unknown'


def is_jawi(text: str) -> bool:
    """Quick check if text contains Jawi script.

    Args:
        text: Input text.

    Returns:
        True if text contains Jawi characters.
    """
    return any(ch in JAWI_CHARS for ch in text)


def _lookup_word(word: str) -> Optional[str]:
    """Look up a word in the dictionary (case-insensitive)."""
    return MALAY_JAWI_DICT.get(word.lower())


def _rumi_word_to_jawi(word: str) -> str:
    """Convert a single Rumi word to Jawi.

    First checks dictionary, then falls back to character-by-character
    transliteration.
    """
    # Check dictionary first
    dict_result = _lookup_word(word)
    if dict_result:
        return dict_result

    # Fall back to character-by-character conversion
    result = []
    lower = word.lower()
    i = 0
    length = len(lower)

    while i < length:
        ch = lower[i]

        # Try digraphs first (sy, ng, ny, kh, gh, th, sh, dh, ph)
        if i + 1 < length:
            digraph = lower[i:i+2]
            if digraph in CONSONANT_MAP:
                result.append(CONSONANT_MAP[digraph])
                i += 2
                continue

        # Single consonant
        if ch in CONSONANT_MAP:
            result.append(CONSONANT_MAP[ch])
            i += 1
            continue

        # Vowels
        if ch in 'aeiou':
            if i == 0:
                # Vowel at word start: use alif
                result.append(ALIF)
            else:
                # Vowel in middle/end: use appropriate carrier
                if ch == 'a':
                    result.append(ALIF)
                elif ch == 'e':
                    # Malay 'e' can be pepet or taling
                    # Default to ye for 'e' sound (taling)
                    result.append(YE)
                elif ch == 'i':
                    result.append(YE)
                elif ch == 'o':
                    result.append(WAW)
                elif ch == 'u':
                    result.append(WAW)
            i += 1
            continue

        # Unknown character: keep as-is (number, punctuation)
        result.append(ch)
        i += 1

    return ''.join(result)


def _jawi_word_to_rumi(word: str) -> str:
    """Convert a single Jawi word to Rumi.

    First checks reverse dictionary, then falls back to character-by-character
    transliteration.
    """
    # Check reverse dictionary first
    if word in JAWI_MALAY_DICT:
        return JAWI_MALAY_DICT[word]

    # Fall back to character-by-character conversion
    result = []
    i = 0
    length = len(word)

    while i < length:
        ch = word[i]

        # Skip diacritics (they're implicit in Rumi)
        if ch in (FATHAH, KASRA, DAMMA, SUKUN):
            i += 1
            continue

        if ch in REVERSE_CONSONANT:
            result.append(REVERSE_CONSONANT[ch])
            i += 1
            continue

        # Alif at start often represents initial vowel 'a'
        if ch == ALIF:
            # If previous char was consonant, alif likely represents 'a' vowel
            if result and result[-1] not in 'aeiou':
                result.append('a')
            elif not result:
                result.append('a')
            i += 1
            continue

        # Waw can be consonant 'w' or vowel 'o'/'u'
        if ch == WAW:
            if i + 1 < length and word[i+1] in JAWI_CHARS and word[i+1] != ALIF:
                result.append('w')
            else:
                result.append('w')  # default to consonant
            i += 1
            continue

        # Ya/Ye can be consonant 'y' or vowel 'i'/'e'
        if ch in (YA, YE):
            if ch == YE and i + 1 >= length:
                result.append('i')  # ye at end usually 'i'
            else:
                result.append('y')  # default to consonant
            i += 1
            continue

        # Unknown character: keep as-is
        result.append(ch)
        i += 1

    return ''.join(result)


def to_jawi(text: str) -> str:
    """Convert Rumi (Latin) text to Jawi (Arabic script).

    Handles mixed Malay-English text by only converting recognized Malay words
    and keeping English words as-is.

    Args:
        text: Rumi text to convert.

    Returns:
        Jawi text. English words preserved in Latin script.
    """
    if not text:
        return text

    # Split into tokens preserving whitespace and punctuation
    tokens = re.split(r'(\s+|[^\w\s]+)', text)
    result = []

    for token in tokens:
        if not token:
            continue
        # Whitespace and punctuation pass through
        if re.match(r'^[\s\W]+$', token):
            result.append(token)
            continue
        # Numbers pass through
        if re.match(r'^\d+$', token):
            result.append(token)
            continue
        # Convert word
        result.append(_rumi_word_to_jawi(token))

    return ''.join(result)


def to_rumi(text: str) -> str:
    """Convert Jawi (Arabic script) text to Rumi (Latin).

    Args:
        text: Jawi text to convert.

    Returns:
        Rumi text.
    """
    if not text:
        return text

    # Split by whitespace and punctuation, preserve delimiters
    tokens = re.split(r'(\s+|[^\s\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+)', text)
    result = []

    for token in tokens:
        if not token:
            continue
        # Whitespace and punctuation pass through
        if not is_jawi(token):
            result.append(token)
            continue
        # Convert Jawi word
        result.append(_jawi_word_to_rumi(token))

    return ''.join(result)


def to_jawi_words(text: str) -> List[Tuple[str, str]]:
    """Convert text to Jawi and return word-by-word mapping.

    Args:
        text: Rumi text.

    Returns:
        List of (rumi_word, jawi_word) tuples.
    """
    words = text.split()
    return [(w, _rumi_word_to_jawi(w)) for w in words]


def batch_to_jawi(texts: List[str]) -> List[str]:
    """Convert multiple texts to Jawi.

    Args:
        texts: List of Rumi texts.

    Returns:
        List of Jawi texts.
    """
    return [to_jawi(t) for t in texts]


def batch_to_rumi(texts: List[str]) -> List[str]:
    """Convert multiple Jawi texts to Rumi.

    Args:
        texts: List of Jawi texts.

    Returns:
        List of Rumi texts.
    """
    return [to_rumi(t) for t in texts]
