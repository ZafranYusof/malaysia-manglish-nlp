"""Spell correction for Malay/Manglish text.

Uses edit distance + BM dictionary for candidate generation,
with frequency-based ranking.
"""

from __future__ import annotations

from typing import Any, Dict, List

import re
from malaysian_manglish_nlp.utils import get_shortforms

# Common BM words (frequency-ordered, most common first)
_BM_DICTIONARY = {
    # Pronouns & determiners
    'saya', 'aku', 'awak', 'kamu', 'dia', 'mereka', 'kami', 'kita',
    'ini', 'itu', 'yang', 'apa', 'siapa', 'mana', 'bila', 'kenapa',
    'engkau', 'beliau', 'baginda', 'hamba', 'patik', 'beta',
    # Verbs (common)
    'ada', 'adalah', 'akan', 'boleh', 'buat', 'pergi', 'datang', 'makan',
    'minum', 'tidur', 'bangun', 'kerja', 'belajar', 'tulis', 'baca',
    'cakap', 'dengar', 'tengok', 'lihat', 'nampak', 'rasa', 'fikir',
    'tahu', 'kenal', 'ingat', 'lupa', 'faham', 'cuba', 'tolong',
    'minta', 'beri', 'ambil', 'letak', 'hantar', 'terima', 'bayar',
    'beli', 'jual', 'guna', 'pakai', 'buka', 'tutup', 'masuk', 'keluar',
    'naik', 'turun', 'jalan', 'lari', 'duduk', 'berdiri',
    'suka', 'sayang', 'benci', 'takut', 'malu', 'marah',
    'sampai', 'balik', 'pulang', 'tiba', 'tinggal', 'pindah',
    'nyanyi', 'menari', 'bermain', 'berlari', 'berenang',
    'masak', 'goreng', 'rebus', 'panggang', 'kukus', 'tumis',
    'basuh', 'cuci', 'sapu', 'gosok', 'lap', 'sidai', 'jemur',
    'jahit', 'lipat', 'potong', 'koyak', 'tampal', 'ikat',
    'tanam', 'siram', 'petik', 'cabut', 'tebang',
    'panjat', 'lompat', 'loncat', 'terjun', 'mendaki',
    'tangkap', 'lepas', 'pegang', 'genggam', 'campak', 'lempar',
    'tolak', 'tarik', 'tekan', 'angkat', 'pikul', 'galas',
    'simpan', 'buang', 'kumpul', 'susun', 'pilih', 'atur',
    'cari', 'jumpa', 'dapat', 'hilang', 'nampak', 'sembunyi',
    'tunggu', 'harap', 'percaya', 'yakin', 'ragu',
    'mula', 'habis', 'selesai', 'siap', 'tamat', 'henti',
    'ubah', 'tukar', 'ganti', 'tambah', 'kurang', 'campur',
    'pecah', 'patah', 'retak', 'robek', 'rosak', 'baiki',
    'terbang', 'mendarat', 'berlayar', 'belayar',
    # Adjectives
    'baik', 'buruk', 'besar', 'kecil', 'tinggi', 'rendah', 'panjang',
    'pendek', 'baru', 'lama', 'muda', 'tua', 'cantik', 'hodoh',
    'pandai', 'bodoh', 'rajin', 'malas', 'kaya', 'miskin',
    'senang', 'susah', 'cepat', 'lambat', 'dekat', 'jauh',
    'panas', 'sejuk', 'basah', 'kering', 'bersih', 'kotor',
    'murah', 'mahal', 'sedap', 'manis', 'masam', 'pahit', 'masin',
    'putih', 'hitam', 'merah', 'biru', 'hijau', 'kuning',
    'gemuk', 'kurus', 'sihat', 'sakit', 'lemah', 'kuat',
    'tebal', 'nipis', 'lebar', 'sempit', 'dalam', 'cetek',
    'keras', 'lembut', 'kasar', 'halus', 'licin', 'kesat',
    'terang', 'gelap', 'cerah', 'mendung', 'redup',
    'tajam', 'tumpul', 'bengkok', 'lurus', 'bulat', 'rata',
    'penuh', 'kosong', 'padat', 'longgar', 'ketat',
    'betul', 'salah', 'benar', 'palsu', 'asli', 'tiruan',
    'bagus', 'teruk', 'hebat', 'biasa', 'luar biasa',
    'gembira', 'sedih', 'marah', 'takut', 'malu', 'bangga',
    'penat', 'segar', 'lapar', 'kenyang', 'dahaga', 'mengantuk',
    # Nouns
    'orang', 'anak', 'budak', 'kawan', 'keluarga', 'rumah', 'sekolah',
    'universiti', 'kedai', 'hospital', 'masjid', 'gereja', 'kuil',
    'jalan', 'kereta', 'motosikal', 'bas', 'kapal', 'telefon',
    'komputer', 'buku', 'kertas', 'pen', 'meja', 'kerusi',
    'makanan', 'minuman', 'nasi', 'roti', 'air', 'susu', 'kopi', 'teh',
    'ayam', 'ikan', 'daging', 'sayur', 'buah', 'telur',
    'duit', 'wang', 'harga', 'gaji', 'kerja', 'pejabat',
    'hari', 'minggu', 'bulan', 'tahun', 'masa', 'waktu',
    'pagi', 'tengahari', 'petang', 'malam', 'semalam', 'esok',
    'negara', 'negeri', 'bandar', 'kampung', 'dunia',
    'pintu', 'tingkap', 'dinding', 'lantai', 'siling', 'tangga',
    'bilik', 'dapur', 'tandas', 'halaman', 'taman', 'padang',
    'sungai', 'laut', 'gunung', 'bukit', 'hutan', 'pantai',
    'langit', 'awan', 'hujan', 'angin', 'matahari', 'bintang', 'bulan',
    'baju', 'seluar', 'kasut', 'topi', 'beg', 'payung',
    'pisau', 'sudu', 'garpu', 'pinggan', 'mangkuk', 'cawan', 'gelas',
    'kucing', 'anjing', 'burung', 'ikan', 'ayam', 'lembu', 'kambing',
    'doktor', 'guru', 'polis', 'tentera', 'peguam', 'jurutera',
    'ibu', 'bapa', 'abang', 'kakak', 'adik', 'datuk', 'nenek',
    'suami', 'isteri', 'anak', 'cucu', 'sepupu', 'saudara',
    'kepala', 'mata', 'hidung', 'mulut', 'telinga', 'tangan', 'kaki',
    'perut', 'dada', 'bahu', 'lutut', 'jari', 'kuku', 'rambut',
    # Function words
    'dan', 'atau', 'tetapi', 'tapi', 'untuk', 'dengan', 'dari',
    'dalam', 'luar', 'atas', 'bawah', 'depan', 'belakang',
    'antara', 'tanpa', 'tentang', 'seperti', 'kerana', 'sebab',
    'kalau', 'jika', 'walaupun', 'supaya', 'agar', 'sehingga',
    'tidak', 'bukan', 'belum', 'sudah', 'sedang', 'masih',
    'sangat', 'amat', 'terlalu', 'agak', 'kurang', 'lebih',
    'semua', 'setiap', 'banyak', 'sedikit', 'beberapa',
    'juga', 'pun', 'lagi', 'sahaja', 'hanya', 'memang',
    'oleh', 'pada', 'kepada', 'daripada', 'terhadap', 'mengenai',
    'sejak', 'hingga', 'sementara', 'sambil', 'ketika', 'apabila',
    'malah', 'bahkan', 'namun', 'walau', 'meskipun', 'sekiranya',
    # Common EN words used in Manglish
    'okay', 'sorry', 'thanks', 'please', 'hello', 'bye',
    'phone', 'laptop', 'wifi', 'online', 'offline',
    'meeting', 'class', 'assignment', 'project', 'exam', 'test',
    'parking', 'shopping', 'dinner', 'lunch', 'breakfast',
    'cancel', 'confirm', 'update', 'download', 'upload',
}

# Combine with shortforms as valid words
_ALL_VALID = _BM_DICTIONARY.copy()


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def _candidates(word: str, max_distance: int = 2) -> List:
    """Generate correction candidates within edit distance."""
    shortforms = set(get_shortforms().keys())
    all_valid = _ALL_VALID | shortforms
    
    results = []
    for valid_word in all_valid:
        dist = _edit_distance(word, valid_word)
        if dist <= max_distance and dist > 0:
            results.append((valid_word, dist))
    
    # Sort by distance, then alphabetically
    results.sort(key=lambda x: (x[0] == word, x[1], x[0]))
    return results


def correct_word(word: str, max_distance: int = 2, top_n: int = 5) -> Dict[str, Any]:
    """Suggest corrections for a single word.
    
    Args:
        word: Potentially misspelled word.
        max_distance: Maximum edit distance for candidates (default: 2).
        top_n: Maximum number of suggestions (default: 5).
    
    Returns:
        dict: Result with keys:
            - original (str): Input word
            - corrected (str): Best correction (or original if valid)
            - is_valid (bool): Whether word is already valid
            - suggestions (list): Top candidates with distances
    
    Example:
        >>> correct_word("mkaan")
        {'original': 'mkaan', 'corrected': 'makan', 'is_valid': False,
         'suggestions': [{'word': 'makan', 'distance': 1}, ...]}
    """
    lower = word.lower()
    shortforms = set(get_shortforms().keys())
    all_valid = _ALL_VALID | shortforms
    
    # Already valid
    if lower in all_valid:
        return {
            'original': word,
            'corrected': word,
            'is_valid': True,
            'suggestions': [],
        }
    
    # Find candidates
    candidates = _candidates(lower, max_distance)
    suggestions = [{'word': w, 'distance': d} for w, d in candidates[:top_n]]
    
    # Best correction: prefer distance 1, then dictionary words over shortforms
    best = word
    if candidates:
        # Prefer real dictionary words over shortforms at same distance
        dict_candidates = [(w, d) for w, d in candidates if w in _BM_DICTIONARY]
        if dict_candidates:
            best = dict_candidates[0][0]
        else:
            best = candidates[0][0]
    
    return {
        'original': word,
        'corrected': best,
        'is_valid': False,
        'suggestions': suggestions,
    }


def correct(text: str, max_distance: int = 1) -> str:
    """Correct spelling in text.
    
    Only corrects words that are not in the dictionary AND have
    a close candidate (distance 1 by default for safety).
    
    Args:
        text: Input text.
        max_distance: Max edit distance (default: 1 for conservative).
    
    Returns:
        dict: Result with keys:
            - corrected (str): Corrected text
            - changes (list): List of corrections made
            - original (str): Original text
    
    Example:
        >>> correct("aku nk pregi mkn")
        {'corrected': 'aku nk pergi makan', 'changes': [...], ...}
    """
    shortforms = set(get_shortforms().keys())
    all_valid = _ALL_VALID | shortforms
    
    words = text.split()
    corrected_words = []
    changes = []
    
    for word in words:
        # Preserve punctuation
        punct = ''
        clean_word = word
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
            clean_word = word[:-1]
        
        lower = clean_word.lower()
        
        # Skip if valid, short, or has numbers
        if lower in all_valid or len(lower) <= 2 or re.search(r'\d', lower):
            corrected_words.append(word)
            continue
        
        # Find best correction
        candidates = _candidates(lower, max_distance)
        
        if candidates:
            # Only auto-correct if distance is 1 (safe)
            best_word, best_dist = candidates[0]
            # Prefer dictionary words
            dict_cands = [(w, d) for w, d in candidates if w in _BM_DICTIONARY and d == 1]
            if dict_cands:
                best_word = dict_cands[0][0]
                best_dist = dict_cands[0][1]
            
            if best_dist <= max_distance:
                corrected_words.append(best_word + punct)
                changes.append({
                    'original': clean_word,
                    'corrected': best_word,
                    'distance': best_dist,
                })
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return {
        'corrected': ' '.join(corrected_words),
        'changes': changes,
        'original': text,
    }
