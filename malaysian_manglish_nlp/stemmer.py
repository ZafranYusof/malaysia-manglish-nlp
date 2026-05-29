"""Malay stemmer - rule-based affix removal."""

from __future__ import annotations

from typing import Any, Dict, List

import re
from malaysian_manglish_nlp.cache import cached

_SUFFIXES = ['kan', 'an', 'i']
_PARTICLES = ['lah', 'kah', 'tah', 'nya', 'pun']

# Known root words for validation (expanded)
_KNOWN_ROOTS = {
    # Common verbs
    'main', 'tulis', 'baca', 'jalan', 'lari', 'makan', 'minum',
    'tidur', 'bangun', 'duduk', 'kerja', 'buat', 'ambil',
    'beri', 'cari', 'dapat', 'guna', 'fikir', 'rasa', 'dengar',
    'lihat', 'kata', 'tanya', 'jawab', 'ajar', 'belajar',
    'sapu', 'ubah', 'isi', 'ikat', 'ukur',
    'hantar', 'masak', 'basuh', 'cuci', 'potong', 'lipat',
    'pukul', 'tarik', 'tolak', 'tekan', 'angkat', 'letak',
    'pilih', 'susun', 'kumpul', 'simpan', 'buang', 'tutup', 'buka',
    'terbang', 'renang', 'panjat', 'lompat', 'loncat',
    'soal', 'ulas', 'bahas', 'usaha', 'niaga', 'jual', 'beli',
    'henti', 'mula', 'akhir', 'habis', 'sampai',
    'tahu', 'kenal', 'ingat', 'lupa', 'faham',
    'suka', 'benci', 'takut', 'berani', 'malu',
    'hidup', 'mati', 'lahir', 'tumbuh', 'layu',
    'naik', 'turun', 'masuk', 'keluar', 'datang', 'pergi',
    'rebah', 'jatuh', 'bangkit',
    'pegang', 'lepas', 'genggam', 'capai', 'raih',
    'cakap', 'bisik', 'jerit', 'panggil', 'sebut',
    'tukar', 'ganti', 'tambah', 'kurang', 'campur',
    'pecah', 'patah', 'retak', 'robek', 'koyak',
    'reka', 'cipta', 'bentuk', 'lukis', 'lakar',
    'lawan', 'serang', 'tahan', 'lindung',
    'pasak', 'kukuh', 'teguh', 'mantap',
    # Additional verbs
    'bayar', 'hutang', 'pinjam', 'sewa', 'upah',
    'goreng', 'rebus', 'panggang', 'kukus', 'tumis', 'bakar',
    'jahit', 'rajut', 'tenun', 'sulam', 'tampal',
    'tanam', 'siram', 'petik', 'cabut', 'tebang', 'cantas',
    'tangkap', 'campak', 'lempar', 'baling', 'lontar',
    'pikul', 'galas', 'jinjing', 'sandang', 'tanggung',
    'tunggu', 'harap', 'percaya', 'yakin', 'ragu',
    'rosak', 'baiki', 'betul', 'elak', 'hindar',
    'nyanyi', 'tari', 'lakon', 'berlakon', 'aksi',
    'tiup', 'hembus', 'sedut', 'hisap', 'hirup',
    'gigit', 'kunyah', 'telan', 'ludah', 'muntah',
    'picit', 'urut', 'gosok', 'garu', 'garuk',
    'tunjuk', 'arah', 'pandu', 'bimbing', 'latih',
    'kahwin', 'cerai', 'nikah', 'tunang', 'lamar',
    'doa', 'solat', 'puasa', 'zakat', 'haji',
    'mandi', 'sikat', 'sisir', 'cukur', 'gunting',
    'parkir', 'pusing', 'belok', 'patah balik', 'undur',
    'tumpah', 'bocor', 'meresap', 'mengalir', 'limpah',
    'kilat', 'kilas', 'imbas', 'semak', 'periksa',
    'cetak', 'salin', 'terjemah', 'tafsir', 'hurai',
    'undi', 'calon', 'lantik', 'turun', 'letak jawatan',
    # Nouns (common roots)
    'rumah', 'sekolah', 'air', 'tanah', 'buah', 'pokok',
    'anak', 'budak', 'orang', 'kawan', 'guru', 'murid',
    'kertas', 'buku', 'surat', 'berita', 'cerita',
    'duit', 'wang', 'harga', 'nilai', 'untung', 'rugi',
    'hujan', 'angin', 'panas', 'sejuk', 'banjir',
    'jalan', 'lorong', 'laluan', 'pintu', 'tingkap',
    'makanan', 'minuman', 'pakaian', 'tempat', 'masa',
    'kerusi', 'meja', 'katil', 'almari', 'rak',
    'kereta', 'motor', 'kapal', 'bot', 'lori',
    # Adjectives (common roots)
    'besar', 'kecil', 'tinggi', 'rendah', 'panjang', 'pendek',
    'baik', 'buruk', 'cantik', 'pandai', 'rajin', 'malas',
    'basah', 'kering', 'panas', 'sejuk', 'terang', 'gelap',
    'keras', 'lembut', 'kasar', 'halus', 'licin',
    'putih', 'hitam', 'merah', 'biru', 'hijau', 'kuning',
    'senang', 'susah', 'cepat', 'lambat', 'dekat', 'jauh',
    'murah', 'mahal', 'baru', 'lama', 'muda', 'tua',
    'gemuk', 'kurus', 'sihat', 'sakit', 'kuat', 'lemah',
    'tebal', 'nipis', 'lebar', 'sempit', 'tajam', 'tumpul',
    'penuh', 'kosong', 'padat', 'longgar', 'ketat',
    'kotor', 'bersih', 'busuk', 'harum', 'wangi',
    'manis', 'masam', 'pahit', 'masin', 'pedas', 'tawar',
    'lurus', 'bengkok', 'bulat', 'rata', 'curam',
}

_STOP_WORDS = {
    'dan', 'atau', 'yang', 'di', 'ke', 'dari', 'ini', 'itu',
    'ada', 'adalah', 'ialah', 'oleh', 'pada', 'untuk', 'akan',
    'telah', 'sudah', 'belum', 'sedang', 'masih', 'juga', 'pun',
    'dengan', 'dalam', 'luar', 'atas', 'bawah', 'antara',
    'mereka', 'kami', 'kita', 'saya', 'aku', 'dia', 'kamu',
}


def stem(text: str) -> Dict[str, Any]:
    """Stem all words in text.
    
    Args:
        text: Input text.
    
    Returns:
        str: Text with all words stemmed.
    
    Example:
        >>> malaysian_manglish_nlp.stem("mereka berlarian di sekolahan")
        'mereka lari di sekolah'
        >>> malaysian_manglish_nlp.stem("memakan menulis pelajaran")
        'makan tulis ajar'
    """
    words = text.split()
    return ' '.join(stem_word(w) for w in words)


@cached(maxsize=2048)
def stem_word(word: str) -> str:
    """Stem a single Malay word by removing affixes.
    
    Args:
        word: Single word to stem.
    
    Returns:
        str: Stemmed word (root form).
    
    Example:
        >>> stem_word("berlari")
        'lari'
        >>> stem_word("memakan")
        'makan'
        >>> stem_word("menulis")
        'tulis'
        >>> stem_word("pelajaran")
        'ajar'
        >>> stem_word("menyapu")
        'sapu'
        >>> stem_word("terbang")
        'terbang'
        >>> stem_word("memasak")
        'masak'
        >>> stem_word("sekolahan")
        'sekolah'
    """
    if len(word) <= 3:
        return word
    
    original = word.lower()
    
    if original in _STOP_WORDS:
        return original
    if original in _KNOWN_ROOTS:
        return original
    
    # Remove particle
    base = original
    for p in _PARTICLES:
        if base.endswith(p) and len(base) > len(p) + 3:
            base = base[:-len(p)]
            break
    
    # Generate ALL possible strippings and pick best
    candidates = set()
    candidates.add(base)
    
    # Try: just suffix removal
    for s in _SUFFIXES:
        if base.endswith(s) and len(base) > len(s) + 3:
            no_suffix = base[:-len(s)]
            candidates.add(no_suffix)
            # Then try prefix removal on the suffix-stripped form
            candidates.update(_all_prefix_candidates(no_suffix))
    
    # Try: just prefix removal
    candidates.update(_all_prefix_candidates(base))
    
    # Try: prefix removal then suffix removal
    for pc in _all_prefix_candidates(base):
        for s in _SUFFIXES:
            if pc.endswith(s) and len(pc) > len(s) + 3:
                candidates.add(pc[:-len(s)])
    
    # Filter valid (min 3 chars)
    valid = {c for c in candidates if len(c) >= 3 and c != original}
    
    if not valid:
        return original
    
    # Priority 1: known root
    known = [c for c in valid if c in _KNOWN_ROOTS]
    if known:
        # If multiple known roots, prefer the one that requires simpler prefix removal
        # i.e. prefer 'masak' (me+masak) over 'pasak' (mem+asak) for 'memasak'
        # Heuristic: prefer candidate that is a direct substring of original
        direct = [c for c in known if c in original]
        if direct:
            return max(direct, key=len)
        # Otherwise prefer longest known root (least aggressive)
        return max(known, key=len)
    
    # Priority 2: shortest valid candidate that's not too aggressive
    # Don't reduce more than 60% of original length
    min_len = max(3, int(len(original) * 0.4))
    reasonable = [c for c in valid if len(c) >= min_len]
    
    if reasonable:
        return min(reasonable, key=len)
    
    return min(valid, key=len)


def _all_prefix_candidates(word: str) -> List[str]:
    """Generate all possible prefix-removal candidates for a word."""
    candidates = set()
    
    # memper-
    if word.startswith('memper') and len(word) > 8:
        candidates.add(word[6:])
    
    # meny- -> s
    if word.startswith('meny') and len(word) > 5:
        candidates.add('s' + word[4:])
    
    # meng-
    if word.startswith('meng') and len(word) > 5:
        rest = word[4:]
        candidates.add(rest)
        if rest[0:1] in 'aeiou':
            candidates.add('k' + rest)
    
    # mem-
    if word.startswith('mem') and len(word) > 4:
        rest = word[3:]
        candidates.add(rest)
        if rest[0:1] in 'bp':
            candidates.add(rest)
        elif rest[0:1] in 'aeiou':
            candidates.add('p' + rest)
    
    # men-
    if word.startswith('men') and len(word) > 4:
        rest = word[3:]
        candidates.add(rest)
        if rest[0:1] in 'aeiou':
            candidates.add('t' + rest)
        if rest[0:1] in 'cdtj':
            candidates.add(rest)
    
    # me- (must check AFTER mem/men/meny/meng)
    if word.startswith('me') and len(word) > 4:
        rest = word[2:]
        # Always try me- strip (memasak -> masak, melawan -> lawan)
        candidates.add(rest)
        if not word.startswith(('mem', 'men', 'meny', 'meng')):
            candidates.add(rest)
    
    # peny- -> s
    if word.startswith('peny') and len(word) > 5:
        candidates.add('s' + word[4:])
    
    # peng-
    if word.startswith('peng') and len(word) > 5:
        rest = word[4:]
        candidates.add(rest)
        if rest[0:1] in 'aeiou':
            candidates.add('k' + rest)
    
    # pem-
    if word.startswith('pem') and len(word) > 4:
        rest = word[3:]
        candidates.add(rest)
        if rest[0:1] in 'aeiou':
            candidates.add('p' + rest)
    
    # pen-
    if word.startswith('pen') and len(word) > 4:
        rest = word[3:]
        candidates.add(rest)
        if rest[0:1] in 'aeiou':
            candidates.add('t' + rest)
    
    # pe- (after pem/pen/peny/peng)
    if word.startswith('pe') and len(word) > 4:
        rest = word[2:]
        if not word.startswith(('pem', 'pen', 'peny', 'peng')):
            candidates.add(rest)
        candidates.add(rest)
        # pe- can also cause consonant doubling: pelajaran -> pe+lajar+an -> ajar
        # Try stripping first consonant after pe- if it matches next char pattern
        if len(rest) > 3 and rest[0] == 'l':
            candidates.add(rest[1:])  # pelajar -> ajar
        if len(rest) > 3 and rest[0] == 'n':
            candidates.add(rest[1:])  # penari -> ari? (less common)
        if len(rest) > 3 and rest[0] == 'r':
            candidates.add(rest[1:])  # perasaan -> asaan?
    
    # Simple prefixes
    for p in ('ber', 'ter', 'per', 'di', 'ke', 'se'):
        if word.startswith(p) and len(word) > len(p) + 2:
            candidates.add(word[len(p):])
    
    # Filter too short
    return {c for c in candidates if len(c) >= 3}


def get_root(word: str) -> Dict[str, Any]:
    """Get the root word with full morphological metadata.
    
    Args:
        word: Input word.
    
    Returns:
        dict: Dictionary with 'original', 'root', 'prefix', 'suffix', 'particle'.
    
    Example:
        >>> get_root("berlarian")
        {'original': 'berlarian', 'root': 'lari', 'prefix': 'ber', 'suffix': 'an', 'particle': ''}
    """
    original = word.lower()
    root = stem_word(word)
    
    prefix = ''
    suffix = ''
    particle = ''
    
    for p in _PARTICLES:
        if original.endswith(p) and not root.endswith(p):
            particle = p
            break
    
    base = original
    if particle:
        base = base[:-len(particle)]
    
    for s in _SUFFIXES:
        if base.endswith(s) and not root.endswith(s):
            suffix = s
            break
    
    # Detect prefix
    for p in ('memper', 'diper', 'meny', 'meng', 'mem', 'men', 'me',
              'peny', 'peng', 'pem', 'pen', 'pe', 'ber', 'ter', 'per', 'di', 'ke', 'se'):
        if original.startswith(p) and not root.startswith(p):
            prefix = p
            break
    
    return {
        'original': original,
        'root': root,
        'prefix': prefix,
        'suffix': suffix,
        'particle': particle,
    }
