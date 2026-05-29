"""Text augmentation for Manglish/BM.

Generate variations of text for data augmentation in NLP pipelines.
Inspired by malaya.augmentation.rules.
"""

from __future__ import annotations

from typing import List, Optional

import re
import random

# Vowel replacements (simulate slang/typo)
_VOWEL_REPLACE = {'a': ['o', 'e'], 'i': ['e', 'o'], 'o': ['u', 'a'], 'u': ['o'], 'e': ['a', 'i']}

# Consonant replacements (simulate typo)
_CONSONANT_REPLACE = {
    'b': ['p'], 'd': ['t'], 'f': ['p', 'v'], 'g': ['k', 'h'],
    'j': ['y'], 'k': ['g'], 'n': ['m', 'ng'], 'r': ['l'],
    's': ['z'], 't': ['d'], 'v': ['f'], 'z': ['s'],
}

# Kelantanese vowel shifts
_KELANTAN_VOWEL = {'a': 'o', 'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o'}

# Common BM synonyms (expanded)
_SYNONYMS = {
    'besar': ['gedang', 'agung', 'raksasa', 'luas', 'mega'],
    'kecil': ['cilik', 'mungil', 'mini', 'kerdil', 'comel'],
    'cantik': ['lawa', 'cun', 'jelita', 'molek', 'ayu', 'gorgeous'],
    'pandai': ['bijak', 'cerdik', 'pintar', 'genius', 'smart'],
    'bodoh': ['bangang', 'bebal', 'dungu', 'bengap', 'tolol'],
    'makan': ['jamu', 'santap', 'telan', 'ngap', 'ratah'],
    'pergi': ['pegi', 'gi', 'berangkat', 'bertolak', 'gerak'],
    'balik': ['pulang', 'blk', 'return', 'cabut', 'chow'],
    'cepat': ['laju', 'pantas', 'segera', 'express', 'speed'],
    'lambat': ['lewat', 'perlahan', 'slow', 'lembab', 'lengah'],
    'bagus': ['best', 'power', 'mantap', 'solid', 'padu', 'terbaik', 'hebat'],
    'teruk': ['hampeh', 'hancur', 'parah', 'dahsyat', 'kronik'],
    'marah': ['bengang', 'geram', 'murka', 'naik angin', 'triggered'],
    'gembira': ['happy', 'seronok', 'syok', 'riang', 'ceria', 'excited'],
    'sedih': ['pilu', 'sayu', 'duka', 'down', 'murung'],
    'takut': ['gerun', 'seram', 'cuak', 'gabra', 'nervous'],
    'suka': ['minat', 'gemar', 'enjoy', 'into', 'fancy'],
    'rumah': ['umah', 'rmh', 'kediaman', 'crib', 'home'],
    'kereta': ['keta', 'car', 'kenderaan', 'ride', 'whip'],
    'duit': ['wang', 'ringgit', 'pitih', 'cash', 'money'],
    'muda': ['young', 'remaja', 'junior', 'budak'],
    'tua': ['old', 'senior', 'veteran', 'warga emas'],
    'murah': ['cheap', 'berpatutan', 'affordable', 'jimat'],
    'mahal': ['expensive', 'costly', 'premium', 'pricey'],
    'senang': ['mudah', 'easy', 'simple', 'ringkas'],
    'susah': ['payah', 'hard', 'difficult', 'tough', 'mencabar'],
    'ramai': ['banyak', 'crowded', 'packed', 'penuh sesak'],
    'sikit': ['sedikit', 'few', 'little', 'kurang'],
    'tinggi': ['tall', 'high', 'lofty', 'menjulang'],
    'rendah': ['low', 'short', 'pendek'],
    'panas': ['hot', 'terik', 'menyengat', 'hangat'],
    'sejuk': ['cold', 'cool', 'dingin', 'beku'],
    'kawan': ['member', 'bro', 'geng', 'buddy', 'mate', 'fren'],
    'kerja': ['work', 'job', 'keje', 'hustle', 'grind'],
    'tidur': ['sleep', 'tido', 'rest', 'zzz', 'pengsan'],
    'cantik': ['lawa', 'cun', 'pretty', 'beautiful', 'stunning'],
    'lelaki': ['jantan', 'guy', 'dude', 'bro', 'abang'],
    'perempuan': ['betina', 'girl', 'sis', 'kakak', 'minah'],
    'makanan': ['food', 'lauk', 'hidangan', 'juadah', 'menu'],
    'telefon': ['phone', 'hp', 'handphone', 'fon', 'device'],
    'sekolah': ['school', 'sklh', 'tempat belajar'],
    'universiti': ['uni', 'campus', 'IPT', 'kolej'],
}


def vowel_alternate(word: str, threshold: float = 0.5) -> str:
    """Remove vowels to create SMS-style abbreviation.
    
    Args:
        word: Input word.
        threshold: Probability of removing each vowel (default: 0.5).
    
    Returns:
        str: Word with some vowels removed.
    
    Example:
        >>> vowel_alternate('kampung')
        'kmpng'
        >>> vowel_alternate('sekolah')
        'sklh'
    """
    if len(word) <= 3:
        return word
    
    result = []
    vowels = set('aeiouAEIOU')
    
    for i, char in enumerate(word):
        if char in vowels and i > 0 and i < len(word) - 1:
            if random.random() > threshold:
                result.append(char)
            # else skip (remove vowel)
        else:
            result.append(char)
    
    out = ''.join(result)
    return out if len(out) >= 2 else word


def socialmedia_form(word: str) -> str:
    """Generate social media variations of a word.
    
    Applies common transformations:
    - Vowel removal (makan -> mkn)
    - Repeated last char (best -> besttt)
    - Number substitution (satu -> 1)
    - Elongation (gila -> gilaaaa)
    
    Args:
        word: Input word.
    
    Returns:
        list[str]: Possible social media forms.
    
    Example:
        >>> socialmedia_form('makan')
        ['mkn', 'makannnn', 'MAKAN', 'makan2']
    """
    forms = []
    lower = word.lower()
    
    # Vowel removal
    no_vowel = re.sub(r'[aeiou]', '', lower)
    if len(no_vowel) >= 2 and no_vowel != lower:
        forms.append(no_vowel)
    
    # Elongation (repeat last char)
    if lower[-1:].isalpha():
        forms.append(lower + lower[-1] * 3)
    
    # ALL CAPS
    forms.append(word.upper())
    
    # Add number suffix
    forms.append(lower + '2')
    
    # Repeat last syllable
    if len(lower) >= 4:
        forms.append(lower + lower[-2:])
    
    return forms


def replace_similar_vowels(word: str, threshold: float = 0.5) -> str:
    """Replace vowels with similar vowels to simulate slang.
    
    Args:
        word: Input word.
        threshold: Probability of replacement (default: 0.5).
    
    Returns:
        str: Word with replaced vowels.
    
    Example:
        >>> replace_similar_vowels('makan')  # possible: 'mokan', 'meken'
    """
    result = []
    for char in word:
        lower_char = char.lower()
        if lower_char in _VOWEL_REPLACE and random.random() < threshold:
            replacement = random.choice(_VOWEL_REPLACE[lower_char])
            result.append(replacement if char.islower() else replacement.upper())
        else:
            result.append(char)
    return ''.join(result)


def replace_similar_consonants(word: str, threshold: float = 0.5) -> str:
    """Replace consonants with similar consonants to simulate typo.
    
    Args:
        word: Input word.
        threshold: Probability of replacement (default: 0.5).
    
    Returns:
        str: Word with replaced consonants.
    
    Example:
        >>> replace_similar_consonants('barang')  # possible: 'parang', 'balang'
    """
    result = []
    for char in word:
        lower_char = char.lower()
        if lower_char in _CONSONANT_REPLACE and random.random() < threshold:
            replacement = random.choice(_CONSONANT_REPLACE[lower_char])
            if len(replacement) == 1:
                result.append(replacement if char.islower() else replacement.upper())
            else:
                result.append(replacement)
        else:
            result.append(char)
    return ''.join(result)


def kelantanese_form(word: str) -> str:
    """Convert word to Kelantanese dialect form.
    
    Rules:
    - Final 'a' -> 'o' (makan -> make, barang -> bare)
    - Final 'ang' -> 'e' (barang -> bare)
    - Final 'an' -> 'e' (makan -> make)
    - Final 'ar' -> 'o' (besar -> beso)
    - Double vowel at end (kakak -> kakok)
    
    Args:
        word: Standard BM word.
    
    Returns:
        list[str]: Possible Kelantanese forms.
    
    Example:
        >>> kelantanese_form('makan')
        ['make', 'mako']
        >>> kelantanese_form('barang')
        ['bare']
        >>> kelantanese_form('kakak')
        ['kakok']
    """
    lower = word.lower()
    forms = []
    
    # -ang -> -e
    if lower.endswith('ang'):
        forms.append(lower[:-3] + 'e')
    
    # -an -> -e
    if lower.endswith('an') and not lower.endswith('ang'):
        forms.append(lower[:-2] + 'e')
    
    # -ar -> -o
    if lower.endswith('ar'):
        forms.append(lower[:-2] + 'o')
    
    # -ak -> -ok
    if lower.endswith('ak'):
        forms.append(lower[:-2] + 'ok')
    
    # Final vowel 'a' in last syllable -> 'o'
    if lower.endswith('a'):
        forms.append(lower[:-1] + 'o')
    
    # Penultimate 'a' -> 'o' (kakak -> kakok)
    if len(lower) >= 4:
        for i in range(len(lower) - 2, 0, -1):
            if lower[i] == 'a':
                variant = lower[:i] + 'o' + lower[i+1:]
                if variant not in forms:
                    forms.append(variant)
                break
    
    return forms if forms else [lower]


def synonym(word: str, top_n: int = 3) -> List[str]:
    """Get synonyms for a BM/Manglish word.
    
    Args:
        word: Input word.
        top_n: Max synonyms to return (default: 3).
    
    Returns:
        list[str]: Synonym list (empty if not found).
    
    Example:
        >>> synonym('cantik')
        ['lawa', 'cun', 'jelita']
        >>> synonym('bagus')
        ['best', 'power', 'mantap']
    """
    lower = word.lower()
    if lower in _SYNONYMS:
        return _SYNONYMS[lower][:top_n]
    
    # Reverse lookup
    for key, syns in _SYNONYMS.items():
        if lower in syns:
            result = [key] + [s for s in syns if s != lower]
            return result[:top_n]
    
    return []


def augment(text: str, methods: Optional[str] = None, n: str = 5) -> List[str]:
    """Generate augmented variations of text.
    
    Args:
        text: Input text.
        methods: Methods to use. Default: all.
            Options: 'vowel_remove', 'elongate', 'synonym', 'kelantan', 'typo'
        n: Number of variations to generate (default: 5).
    
    Returns:
        list[str]: Augmented text variations.
    
    Example:
        >>> augment("makanan sedap", n=3)
        ['mknn sedap', 'makanan sedapppp', 'makanan best']
    """
    if methods is None:
        methods = ['vowel_remove', 'elongate', 'synonym', 'typo']
    
    results = set()
    words = text.split()
    
    for _ in range(n * 3):  # Generate extra, deduplicate
        if len(results) >= n:
            break
        
        method = random.choice(methods)
        new_words = words.copy()
        
        if not new_words:
            continue
        
        idx = random.randint(0, len(new_words) - 1)
        target = new_words[idx]
        
        if method == 'vowel_remove':
            new_words[idx] = vowel_alternate(target, threshold=0.7)
        elif method == 'elongate':
            if target[-1:].isalpha():
                new_words[idx] = target + target[-1] * random.randint(2, 4)
            else:
                continue
        elif method == 'synonym':
            syns = synonym(target)
            if syns:
                new_words[idx] = random.choice(syns)
            else:
                continue
        elif method == 'kelantan':
            forms = kelantanese_form(target)
            if forms and forms[0] != target.lower():
                new_words[idx] = random.choice(forms)
            else:
                continue
        elif method == 'typo':
            new_words[idx] = replace_similar_consonants(target, threshold=0.3)
        
        variant = ' '.join(new_words)
        if variant != text:
            results.add(variant)
    
    return list(results)[:n]
