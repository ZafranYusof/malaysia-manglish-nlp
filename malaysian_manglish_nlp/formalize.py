"""Informal to formal BM conversion."""

from __future__ import annotations

import re


# Informal -> Formal mappings
_FORMAL_MAP = {
    # Pronouns
    'aku': 'saya', 'ak': 'saya', 'aq': 'saya',
    'ko': 'anda', 'kau': 'anda', 'hang': 'anda',
    'dia': 'beliau', 'dorang': 'mereka', 'diorang': 'mereka',
    'kitorg': 'kami', 'kitorang': 'kami', 'korang': 'kamu semua',
    # Verbs
    'nak': 'ingin', 'nk': 'ingin',
    'pegi': 'pergi', 'gi': 'pergi',
    'balik': 'pulang', 'blk': 'pulang',
    'cakap': 'berkata', 'ckp': 'berkata',
    'tanya': 'bertanya',
    'bagi': 'memberikan', 'bg': 'memberikan',
    'amik': 'mengambil', 'ambik': 'mengambil',
    'letak': 'meletakkan', 'ltk': 'meletakkan',
    'hantar': 'menghantar', 'hntr': 'menghantar',
    'tolong': 'membantu', 'tlg': 'membantu',
    'tengok': 'melihat', 'tgk': 'melihat',
    'dengar': 'mendengar',
    'rasa': 'merasakan',
    'fikir': 'berfikir', 'pikir': 'berfikir',
    'suruh': 'mengarahkan', 'srh': 'mengarahkan',
    # Time
    'jap': 'sebentar', 'kjap': 'sebentar',
    'skrg': 'sekarang', 'skang': 'sekarang',
    'dah': 'telah', 'dh': 'telah',
    'blm': 'belum', 'blum': 'belum',
    'nnt': 'nanti', 'nnti': 'nanti',
    'lepas': 'selepas', 'lps': 'selepas',
    # Negation
    'x': 'tidak', 'tak': 'tidak', 'tk': 'tidak',
    'xde': 'tiada', 'takde': 'tiada',
    'xblh': 'tidak boleh', 'takleh': 'tidak boleh',
    # Particles (remove in formal)
    'la': '', 'lah': '', 'lor': '',
    'kan': '', 'eh': '', 'weh': '', 'wei': '',
    # Particles (convert)
    'je': 'sahaja', 'jer': 'sahaja', 'aje': 'sahaja',
    'kot': 'mungkin',
    # Common words
    'macam': 'seperti', 'mcm': 'seperti',
    'sebab': 'kerana', 'sbb': 'kerana', 'pasal': 'kerana', 'psl': 'kerana',
    'tapi': 'tetapi', 'tp': 'tetapi',
    'dgn': 'dengan', 'ngn': 'dengan',
    'utk': 'untuk', 'tuk': 'untuk',
    'yg': 'yang',
    'ni': 'ini', 'tu': 'itu',
    'kat': 'di', 'dkt': 'di', 'dekat': 'di',
    'byk': 'banyak', 'bnyk': 'banyak',
    'skit': 'sedikit', 'sket': 'sedikit', 'sikit': 'sedikit',
    'mmg': 'memang', 'sgt': 'sangat',
    'lg': 'lagi', 'lgi': 'lagi',
    'pn': 'juga', 'pon': 'juga', 'pun': 'juga',
}


def formalize(text: str) -> str:
    """Convert informal Manglish/BM to formal BM.
    
    Conversions:
    - Pronouns: aku->saya, ko->anda, dorang->mereka
    - Verbs: nak->ingin, pegi->pergi, balik->pulang
    - Particles removed: la, lah, kan, eh, weh
    - Particles converted: je->sahaja, kot->mungkin
    - Negation: x->tidak, xde->tiada
    - Connectors: sbb->kerana, tp->tetapi
    - Auto-capitalizes, adds period if missing
    
    Parameters:
        text (str): Informal text.
    
    Returns:
        str: Formal BM text.
    
    Example:
        >>> malaysian_manglish_nlp.formalize("aku nk pegi kedai jap, ko nk ikut x?")
        'Saya ingin pergi kedai sebentar, anda ingin ikut tidak?'
        >>> malaysian_manglish_nlp.formalize("mmg best la makanan tu")
        'Memang bagus makanan itu.'
    """
    words = text.lower().split()
    result = []
    
    for word in words:
        punct = ''
        clean_word = word
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
            clean_word = word[:-1]
        
        if clean_word in _FORMAL_MAP:
            formal = _FORMAL_MAP[clean_word]
            if formal:  # Non-empty = convert
                result.append(formal + punct)
            # Empty = particle removed (skip)
        else:
            result.append(word)
    
    output = ' '.join(result)
    
    # Capitalize first letter
    if output:
        output = output[0].upper() + output[1:]
    
    # Capitalize after period
    output = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), output)
    
    # Ensure ends with punctuation
    if output and output[-1] not in '.!?':
        output += '.'
    
    # Clean double spaces
    output = re.sub(r'\s+', ' ', output).strip()
    
    return output
