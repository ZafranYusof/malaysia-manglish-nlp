"""OCR text normalization for Malaysian documents.

Cleans up common OCR errors in Bahasa Melayu and Manglish text.
Handles character confusion, split/merged words, and BM-specific fixes.

Zero external dependencies.
"""

import re
from manglish_nlp.dictionary import is_malay, is_english


# ─── Common OCR character confusion patterns ───────────────────────────────────

# Maps visually similar character sequences to their likely correct form
_CHAR_CONFUSIONS = {
    'rn': 'm',
    'cl': 'd',
    'cl': 'd',
    'vv': 'w',
    'li': 'h',
    'lI': 'h',
    'Il': 'H',
    'IJ': 'U',
    'tl': 'd',
}

# Single character confusions (context-dependent)
_SINGLE_CHAR_CONFUSIONS = {
    'l': '1',  # lowercase L → digit 1
    'I': '1',  # uppercase I → digit 1
    'O': '0',  # uppercase O → digit 0
    'o': '0',  # lowercase o → digit 0 (in numeric context)
    'S': '5',  # S → 5 (in numeric context)
    'B': '8',  # B → 8 (in numeric context)
}

# ─── BM-specific OCR error patterns ───────────────────────────────────────────

# Known BM words where 'rn' should be 'm'
_RN_TO_M_WORDS = {
    'rnakan': 'makan', 'rnakanan': 'makanan', 'rnasak': 'masak',
    'rnasakan': 'masakan', 'rnakan': 'makan', 'rnereka': 'mereka',
    'rnalaysia': 'malaysia', 'rnalaysian': 'malaysian',
    'rnahu': 'mahu', 'rnana': 'mana', 'rnasa': 'masa',
    'rnasih': 'masih', 'rnesti': 'mesti', 'rninta': 'minta',
    'rnuda': 'muda', 'rnurah': 'murah', 'rnulai': 'mulai',
    'rnusim': 'musim', 'rnacarn': 'macam', 'rnernang': 'memang',
    'rnernbuat': 'membuat', 'rnernbeli': 'membeli',
    'rnernbaca': 'membaca', 'rnernbantu': 'membantu',
    'rnengapa': 'mengapa', 'rnasalah': 'masalah',
    'rnasyarakat': 'masyarakat', 'rnajikan': 'majikan',
    'rnaklumat': 'maklumat', 'rnandiri': 'mandiri',
    'rnatahari': 'matahari', 'rnengikut': 'mengikut',
    'dalarn': 'dalam', 'rnalarn': 'malam', 'talarnan': 'talaman',
    'kernudian': 'kemudian', 'kernbali': 'kembali',
    'sernua': 'semua', 'rnernberi': 'memberi',
    'rnernerlukan': 'memerlukan', 'rnernilih': 'memilih',
    'rnerniliki': 'memiliki', 'rnernbawa': 'membawa',
    'rnernbuka': 'membuka', 'rnernbayar': 'membayar',
    'rnernbangun': 'membangun', 'rnernpunyai': 'mempunyai',
    'rnernperolehi': 'memperolehi', 'rnernpelajari': 'mempelajari',
    'rnernperbaiki': 'memperbaiki', 'rnernpertahankan': 'mempertahankan',
    'bernarna': 'bernama', 'ternpat': 'tempat', 'ternpatan': 'tempatan',
    'sernasa': 'semasa', 'kernasukan': 'kemasukan',
    'pernuda': 'pemuda', 'pernudi': 'pemudi',
    'rnajalah': 'majalah', 'rnajlis': 'majlis',
    'rnengernbang': 'mengembang', 'rnengernbara': 'mengembara',
}

# Common BM words that get 'cl' misread as 'd'
_CL_TO_D_WORDS = {
    'clan': 'dan', 'clengan': 'dengan', 'clari': 'dari',
    'clalam': 'dalam', 'clapat': 'dapat', 'clua': 'dua',
    'clulu': 'dulu', 'claerah': 'daerah', 'clia': 'dia',
    'cliri': 'diri', 'clitunjukkan': 'ditunjukkan',
}

# ─── Split word patterns (common BM words that get split by OCR) ──────────────

_SPLIT_WORDS = {
    ('me', 'reka'): 'mereka',
    ('ke', 'rana'): 'kerana',
    ('ke', 'pada'): 'kepada',
    ('da', 'lam'): 'dalam',
    ('de', 'ngan'): 'dengan',
    ('se', 'mua'): 'semua',
    ('ma', 'kan'): 'makan',
    ('ma', 'kanan'): 'makanan',
    ('ber', 'sama'): 'bersama',
    ('ter', 'hadap'): 'terhadap',
    ('mem', 'buat'): 'membuat',
    ('men', 'jadi'): 'menjadi',
    ('meng', 'ikut'): 'mengikut',
    ('per', 'kara'): 'perkara',
    ('ke', 'rajaan'): 'kerajaan',
    ('ma', 'syarakat'): 'masyarakat',
    ('pe', 'kerja'): 'pekerja',
    ('pe', 'lajaran'): 'pelajaran',
    ('pen', 'didikan'): 'pendidikan',
    ('per', 'tama'): 'pertama',
    ('se', 'kolah'): 'sekolah',
    ('te', 'lah'): 'telah',
    ('wa', 'laupun'): 'walaupun',
    ('ba', 'gaimana'): 'bagaimana',
    ('se', 'hingga'): 'sehingga',
    ('ke', 'mudian'): 'kemudian',
    ('ber', 'jalan'): 'berjalan',
    ('ter', 'masuk'): 'termasuk',
    ('se', 'lain'): 'selain',
    ('ma', 'lam'): 'malam',
    ('se', 'orang'): 'seorang',
    ('ber', 'laku'): 'berlaku',
    ('men', 'capai'): 'mencapai',
    ('pe', 'ngurus'): 'pengurus',
    ('ke', 'luarga'): 'keluarga',
    ('ma', 'laysia'): 'malaysia',
}

# ─── Merged word patterns (common BM word boundaries) ─────────────────────────

_MERGE_PREFIXES = [
    'dan', 'atau', 'yang', 'ini', 'itu', 'di', 'ke', 'se',
    'untuk', 'dari', 'pada', 'oleh', 'akan', 'telah', 'sudah',
    'tidak', 'bukan', 'belum', 'juga', 'pun', 'lagi', 'sahaja',
    'dengan', 'dalam', 'antara', 'tetapi', 'namun', 'serta',
]

_MERGE_SUFFIXES = [
    'dan', 'atau', 'yang', 'ini', 'itu', 'dia', 'mereka',
    'untuk', 'dari', 'pada', 'akan', 'telah', 'sudah',
    'tidak', 'bukan', 'belum', 'juga', 'pun', 'lagi',
    'saya', 'kami', 'kita', 'anda', 'awak',
]

# ─── BM word list for validation ──────────────────────────────────────────────

_COMMON_BM_WORDS = {
    'ada', 'adalah', 'agar', 'akan', 'aku', 'alam', 'amat', 'anak',
    'antara', 'apa', 'atau', 'awak', 'bagi', 'bahawa', 'baik',
    'banyak', 'baru', 'bawah', 'beberapa', 'begitu', 'beliau',
    'belum', 'benar', 'besar', 'biasa', 'bila', 'boleh', 'buat',
    'bukan', 'cukup', 'dalam', 'dan', 'dapat', 'dari', 'dengan',
    'dia', 'diri', 'dua', 'dunia', 'hal', 'hampir', 'hanya',
    'hari', 'hendak', 'hidup', 'hingga', 'ia', 'iaitu', 'ini',
    'itu', 'jadi', 'jalan', 'jika', 'juga', 'kalau', 'kami',
    'kata', 'kecil', 'keluar', 'kemudian', 'kepada', 'kerana',
    'kerja', 'kita', 'kurang', 'lagi', 'lain', 'lama', 'lebih',
    'lepas', 'mahu', 'maka', 'makan', 'malam', 'mana', 'masa',
    'masih', 'masuk', 'masyarakat', 'melalui', 'memang', 'membuat',
    'mempunyai', 'mengenai', 'menjadi', 'mereka', 'mesti', 'minta',
    'muda', 'mungkin', 'nama', 'namun', 'negara', 'oleh', 'orang',
    'pada', 'paling', 'pasti', 'pelajar', 'pendidikan', 'penting',
    'pergi', 'perkara', 'perlu', 'pertama', 'pihak', 'pula', 'pun',
    'ramai', 'rumah', 'sahaja', 'sama', 'sampai', 'sangat', 'satu',
    'saya', 'sebab', 'sebelum', 'sedang', 'sedikit', 'sehingga',
    'sejak', 'sekolah', 'selain', 'selalu', 'semua', 'seorang',
    'seperti', 'sesuatu', 'setelah', 'setiap', 'sudah', 'suka',
    'supaya', 'tahun', 'tak', 'tanpa', 'tapi', 'telah', 'tempat',
    'tentang', 'terhadap', 'termasuk', 'tetapi', 'tidak', 'tiga',
    'tinggal', 'tuan', 'untuk', 'waktu', 'walaupun', 'wang', 'yang',
    'malaysia', 'kerajaan', 'rakyat', 'negeri', 'bandar', 'kampung',
    'makanan', 'minuman', 'keluarga', 'sekolah', 'universiti',
    'hospital', 'pejabat', 'kedai', 'pasar', 'masjid', 'gereja',
    'kuil', 'jalan', 'sungai', 'gunung', 'pantai', 'pulau',
    'bagaimana', 'mengapa', 'siapa', 'berapa', 'dimana', 'kemana',
}


def _is_valid_word(word):
    """Check if a word is valid BM or English."""
    w = word.lower().strip()
    if not w:
        return False
    if w in _COMMON_BM_WORDS:
        return True
    try:
        if is_malay(w) or is_english(w):
            return True
    except Exception:
        pass
    return False


def _is_numeric_context(text, pos):
    """Check if position is in a numeric context (surrounded by digits)."""
    before = text[max(0, pos - 2):pos]
    after = text[pos + 1:pos + 3]
    has_digit_before = any(c.isdigit() for c in before)
    has_digit_after = any(c.isdigit() for c in after)
    return has_digit_before or has_digit_after


# ─── Main functions ───────────────────────────────────────────────────────────


def normalize_ocr(text):
    """Normalize OCR output from Malaysian documents.

    Args:
        text: Raw OCR text string.

    Returns:
        dict with keys:
            cleaned (str): Corrected text.
            corrections (list): Each item has 'original', 'corrected', 'type'.
            confidence (float): 0.0-1.0, higher means fewer corrections needed.
    """
    if not text or not text.strip():
        return {"cleaned": text or "", "corrections": [], "confidence": 1.0}

    corrections = []
    result = text

    # Step 1: Fix BM-specific OCR errors (rn→m, cl→d patterns)
    result, bm_corrections = _fix_bm_ocr_patterns(result)
    corrections.extend(bm_corrections)

    # Step 2: Fix split words
    result, split_corrections = _fix_split_words(result)
    corrections.extend(split_corrections)

    # Step 3: Fix merged words
    result, merge_corrections = _fix_merged_words(result)
    corrections.extend(merge_corrections)

    # Step 4: Fix number/letter confusion
    result, num_corrections = _fix_number_letter_confusion(result)
    corrections.extend(num_corrections)

    # Step 5: Fix punctuation spacing
    result, punct_corrections = _fix_punctuation_spacing(result)
    corrections.extend(punct_corrections)

    # Step 6: Fix ligature issues
    result, lig_corrections = _fix_ligatures(result)
    corrections.extend(lig_corrections)

    # Calculate confidence
    words = text.split()
    total_words = max(len(words), 1)
    confidence = max(0.0, 1.0 - (len(corrections) / total_words))

    return {
        "cleaned": result,
        "corrections": corrections,
        "confidence": round(confidence, 4),
    }


def fix_common_errors(text):
    """Quick fix OCR errors without detailed report.

    Args:
        text: Raw OCR text.

    Returns:
        str: Corrected text.
    """
    if not text:
        return text or ""
    result = normalize_ocr(text)
    return result["cleaned"]


def detect_ocr_artifacts(text):
    """Detect suspected OCR errors with positions.

    Args:
        text: Text to analyze.

    Returns:
        list of dicts with 'text', 'position', 'type', 'suggestion'.
    """
    if not text:
        return []

    artifacts = []

    # Check for rn patterns that might be 'm'
    for match in re.finditer(r'\brn\w+', text, re.IGNORECASE):
        word = match.group()
        if word.lower() in _RN_TO_M_WORDS:
            artifacts.append({
                'text': word,
                'position': match.start(),
                'type': 'char_confusion_rn_m',
                'suggestion': _RN_TO_M_WORDS[word.lower()],
            })

    # Check for 'rn' in middle of words
    for match in re.finditer(r'\w+rn\w+', text, re.IGNORECASE):
        word = match.group()
        lower = word.lower()
        if lower in _RN_TO_M_WORDS:
            artifacts.append({
                'text': word,
                'position': match.start(),
                'type': 'char_confusion_rn_m',
                'suggestion': _RN_TO_M_WORDS[lower],
            })
        elif 'rn' in lower:
            # Check if replacing rn with m gives a valid word
            candidate = lower.replace('rn', 'm', 1)
            if _is_valid_word(candidate):
                artifacts.append({
                    'text': word,
                    'position': match.start(),
                    'type': 'char_confusion_rn_m',
                    'suggestion': candidate,
                })

    # Check for number/letter confusion in mixed contexts
    for match in re.finditer(r'[A-Za-z]\d+|\d+[A-Za-z]', text):
        fragment = match.group()
        # Check if it looks like a corrupted number
        if re.match(r'^[OoIlSB]\d+$', fragment) or re.match(r'^\d+[OoIlSB]$', fragment):
            artifacts.append({
                'text': fragment,
                'position': match.start(),
                'type': 'number_letter_confusion',
                'suggestion': _fix_num_letter(fragment),
            })

    # Check for missing spaces after punctuation
    for match in re.finditer(r'[a-zA-Z][.!?][a-zA-Z]', text):
        artifacts.append({
            'text': match.group(),
            'position': match.start(),
            'type': 'missing_space',
            'suggestion': match.group()[0] + match.group()[1] + ' ' + match.group()[2],
        })

    # Check for split words
    words = text.split()
    for i in range(len(words) - 1):
        pair = (words[i].lower(), words[i + 1].lower())
        if pair in _SPLIT_WORDS:
            pos = text.find(words[i] + ' ' + words[i + 1])
            if pos >= 0:
                artifacts.append({
                    'text': words[i] + ' ' + words[i + 1],
                    'position': pos,
                    'type': 'split_word',
                    'suggestion': _SPLIT_WORDS[pair],
                })

    return artifacts


def reconstruct_words(text):
    """Fix split words and merged words.

    Args:
        text: Text with potential word boundary errors.

    Returns:
        str: Text with reconstructed word boundaries.
    """
    if not text:
        return text or ""

    # Fix split words
    result, _ = _fix_split_words(text)
    # Fix merged words
    result, _ = _fix_merged_words(result)
    return result


def fix_malay_ocr(text):
    """BM-specific OCR fixes.

    Focuses on errors common in Malay language OCR:
    - rn→m substitution
    - cl→d substitution
    - Common BM word reconstruction
    - Diacritics handling

    Args:
        text: OCR text from BM documents.

    Returns:
        str: Corrected text.
    """
    if not text:
        return text or ""

    result = text

    # Fix rn→m in known BM words
    result, _ = _fix_bm_ocr_patterns(result)

    # Fix split BM words
    result, _ = _fix_split_words(result)

    # Fix merged BM words
    result, _ = _fix_merged_words(result)

    # Fix diacritics (common in BM formal text)
    result = _fix_diacritics(result)

    return result


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _fix_bm_ocr_patterns(text):
    """Fix BM-specific character confusion patterns."""
    corrections = []
    result = text

    # Fix rn→m patterns (case-insensitive word matching)
    for wrong, right in _RN_TO_M_WORDS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        for match in pattern.finditer(result):
            original = match.group()
            # Preserve case of first letter
            if original[0].isupper():
                corrected = right.capitalize()
            else:
                corrected = right
            corrections.append({
                'original': original,
                'corrected': corrected,
                'type': 'char_confusion_rn_m',
            })
        result = pattern.sub(lambda m: right.capitalize() if m.group()[0].isupper() else right, result)

    # Fix cl→d patterns
    for wrong, right in _CL_TO_D_WORDS.items():
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(result):
            original = match.group()
            if original[0].isupper():
                corrected = right.capitalize()
            else:
                corrected = right
            corrections.append({
                'original': original,
                'corrected': corrected,
                'type': 'char_confusion_cl_d',
            })
        result = pattern.sub(lambda m: right.capitalize() if m.group()[0].isupper() else right, result)

    # Generic rn→m for words not in dictionary
    def _try_rn_fix(match):
        word = match.group()
        lower = word.lower()
        if lower in _RN_TO_M_WORDS:
            return word  # Already handled above
        candidate = lower.replace('rn', 'm', 1)
        if _is_valid_word(candidate) and not _is_valid_word(lower):
            corrections.append({
                'original': word,
                'corrected': candidate if word.islower() else candidate.capitalize(),
                'type': 'char_confusion_rn_m',
            })
            return candidate if word.islower() else candidate.capitalize()
        return word

    result = re.sub(r'\b\w*rn\w*\b', _try_rn_fix, result)

    return result, corrections


def _fix_split_words(text):
    """Fix words that were incorrectly split by OCR."""
    corrections = []
    words = text.split()
    result_words = []
    i = 0

    while i < len(words):
        if i < len(words) - 1:
            # Strip punctuation for matching
            w1 = re.sub(r'[^\w]', '', words[i].lower())
            w2 = re.sub(r'[^\w]', '', words[i + 1].lower())
            pair = (w1, w2)

            if pair in _SPLIT_WORDS:
                merged = _SPLIT_WORDS[pair]
                # Preserve case
                if words[i][0].isupper():
                    merged = merged.capitalize()
                # Preserve trailing punctuation from second word
                trailing = ''
                if words[i + 1] and not words[i + 1][-1].isalnum():
                    trailing = words[i + 1][-1]
                corrections.append({
                    'original': words[i] + ' ' + words[i + 1],
                    'corrected': merged + trailing,
                    'type': 'split_word',
                })
                result_words.append(merged + trailing)
                i += 2
                continue

            # Heuristic: if two short fragments combine into a valid BM word
            if len(w1) <= 3 and len(w2) <= 5:
                candidate = w1 + w2
                if _is_valid_word(candidate) and not _is_valid_word(w1):
                    corrections.append({
                        'original': words[i] + ' ' + words[i + 1],
                        'corrected': candidate,
                        'type': 'split_word',
                    })
                    result_words.append(candidate)
                    i += 2
                    continue

        result_words.append(words[i])
        i += 1

    return ' '.join(result_words), corrections


def _fix_merged_words(text):
    """Fix words that were incorrectly merged by OCR."""
    corrections = []
    words = text.split()
    result_words = []

    for word in words:
        # Skip short words or words with punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        if len(clean_word) <= 5 or _is_valid_word(clean_word):
            result_words.append(word)
            continue

        # Try splitting at known boundaries
        split = _try_split_merged(clean_word.lower())
        if split:
            corrections.append({
                'original': word,
                'corrected': split,
                'type': 'merged_word',
            })
            result_words.append(split)
        else:
            result_words.append(word)

    return ' '.join(result_words), corrections


def _try_split_merged(word):
    """Try to split a merged word into valid components."""
    # Try known prefixes
    for prefix in sorted(_MERGE_PREFIXES, key=len, reverse=True):
        if word.startswith(prefix) and len(word) > len(prefix):
            remainder = word[len(prefix):]
            if _is_valid_word(remainder) and len(remainder) >= 3:
                return prefix + ' ' + remainder

    # Try known suffixes
    for suffix in sorted(_MERGE_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix):
            prefix_part = word[:-len(suffix)]
            if _is_valid_word(prefix_part) and len(prefix_part) >= 3:
                return prefix_part + ' ' + suffix

    return None


def _fix_number_letter_confusion(text):
    """Fix number/letter confusion in numeric contexts."""
    corrections = []

    def _replace_num(match):
        original = match.group()
        fixed = _fix_num_letter(original)
        if fixed != original:
            corrections.append({
                'original': original,
                'corrected': fixed,
                'type': 'number_letter_confusion',
            })
        return fixed

    # Fix patterns like "l23" → "123", "O5" → "05"
    # Only in clearly numeric contexts (surrounded by digits)
    result = re.sub(r'(?<!\w)[IlO]\d{1,}', _replace_num, text)
    result = re.sub(r'\d{1,}[IlOoSB](?!\w)', _replace_num, result)

    return result, corrections


def _fix_num_letter(fragment):
    """Fix a single number/letter confused fragment."""
    result = fragment
    # Leading letter confusions
    replacements = {'I': '1', 'l': '1', 'O': '0', 'o': '0', 'S': '5', 'B': '8'}
    for char, digit in replacements.items():
        # Only replace if surrounded by digits
        result = re.sub(r'(?<=\d)' + re.escape(char) + r'(?=\d)', digit, result)
        result = re.sub(r'^' + re.escape(char) + r'(?=\d)', digit, result)
        result = re.sub(r'(?<=\d)' + re.escape(char) + r'$', digit, result)
    return result


def _fix_punctuation_spacing(text):
    """Fix missing spaces after punctuation."""
    corrections = []

    def _add_space(match):
        original = match.group()
        fixed = original[0] + original[1] + ' ' + original[2]
        corrections.append({
            'original': original,
            'corrected': fixed,
            'type': 'missing_space',
        })
        return fixed

    # Add space after period/exclamation/question mark if followed by letter
    # But not for abbreviations like "Dr." or numbers like "3.5"
    result = re.sub(r'([a-zA-Z])[.!?]([a-zA-Z])', _add_space, text)

    return result, corrections


def _fix_ligatures(text):
    """Fix common ligature OCR issues."""
    corrections = []
    result = text

    # fi ligature often misread
    ligature_fixes = {
        '\ufb01': 'fi',  # ﬁ → fi
        '\ufb02': 'fl',  # ﬂ → fl
        '\ufb03': 'ffi',  # ﬃ → ffi
        '\ufb04': 'ffl',  # ﬄ → ffl
    }

    for lig, replacement in ligature_fixes.items():
        if lig in result:
            corrections.append({
                'original': lig,
                'corrected': replacement,
                'type': 'ligature',
            })
            result = result.replace(lig, replacement)

    return result, corrections


def _fix_diacritics(text):
    """Handle missing or wrong diacritics in BM text."""
    # In standard BM, diacritics are rare except in loanwords
    # Common issue: é in words like 'café' being dropped or garbled
    result = text

    # Fix common diacritic issues in BM loanwords
    diacritic_fixes = {
        'kafe': 'kafe',  # café → kafe (BM spelling)
        'resume': 'resume',  # résumé → resume (BM context)
    }

    for wrong, right in diacritic_fixes.items():
        result = re.sub(r'\b' + re.escape(wrong) + r'\b', right, result, flags=re.IGNORECASE)

    return result
