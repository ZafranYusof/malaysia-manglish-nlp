#!/usr/bin/env python3
"""
Informal Manglish/BM to Formal BM converter.
Usage: python formalize.py "aku nk pegi kedai jap, ko nk ikut x?"
"""

import sys
import re
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(SCRIPT_DIR, '..', 'references', 'shortform-dict.json')

# Informal -> Formal BM mappings (beyond shortforms)
FORMAL_MAP = {
    # Pronouns
    'aku': 'saya',
    'ko': 'anda',
    'kau': 'anda',
    'hang': 'anda',
    'dia': 'beliau',
    'dorang': 'mereka',
    'diorang': 'mereka',
    'kitorg': 'kami',
    'kitorang': 'kami',
    # Verbs
    'nak': 'ingin',
    'nk': 'ingin',
    'pegi': 'pergi',
    'gi': 'pergi',
    'balik': 'pulang',
    'cakap': 'berkata',
    'tanya': 'bertanya',
    'bagi': 'memberikan',
    'amik': 'mengambil',
    'ambik': 'mengambil',
    'letak': 'meletakkan',
    'hantar': 'menghantar',
    'tolong': 'membantu',
    'tengok': 'melihat',
    'tgk': 'melihat',
    'dengar': 'mendengar',
    'rasa': 'merasakan',
    'fikir': 'berfikir',
    'pikir': 'berfikir',
    # Time
    'jap': 'sebentar',
    'skrg': 'sekarang',
    'skang': 'sekarang',
    'dah': 'telah',
    'blm': 'belum',
    'nnt': 'nanti',
    'lepas': 'selepas',
    # Negation
    'x': 'tidak',
    'tak': 'tidak',
    'xde': 'tiada',
    'takde': 'tiada',
    'xblh': 'tidak boleh',
    'takleh': 'tidak boleh',
    # Particles (remove in formal)
    'la': '',
    'lah': '',
    'je': 'sahaja',
    'jer': 'sahaja',
    'kot': 'mungkin',
    'kan': '',
    'eh': '',
    'weh': '',
    'wei': '',
    # Common words
    'macam': 'seperti',
    'mcm': 'seperti',
    'sebab': 'kerana',
    'sbb': 'kerana',
    'pasal': 'kerana',
    'tapi': 'tetapi',
    'tp': 'tetapi',
    'dgn': 'dengan',
    'utk': 'untuk',
    'yg': 'yang',
    'ni': 'ini',
    'tu': 'itu',
    'kat': 'di',
    'dkt': 'di',
    'byk': 'banyak',
    'skit': 'sedikit',
    'sket': 'sedikit',
    'sikit': 'sedikit',
    'lagi': 'lagi',
    'pun': 'juga',
    'mmg': 'memang',
    'sgt': 'sangat',
}


def formalize(text):
    """Convert informal Manglish/BM to formal BM."""
    words = text.lower().split()
    result = []
    
    for word in words:
        # Strip punctuation
        punct = ''
        clean_word = word
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
            clean_word = word[:-1]
        
        # Look up formal form
        if clean_word in FORMAL_MAP:
            formal = FORMAL_MAP[clean_word]
            if formal:  # Skip empty (particles to remove)
                result.append(formal + punct)
            # If empty, particle is dropped
        else:
            result.append(word)
    
    # Post-processing
    output = ' '.join(result)
    
    # Capitalize first letter
    if output:
        output = output[0].upper() + output[1:]
    
    # Capitalize after period
    output = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), output)
    
    # Ensure ends with period if no punctuation
    if output and output[-1] not in '.!?':
        output += '.'
    
    # Clean double spaces
    output = re.sub(r'\s+', ' ', output).strip()
    
    return output


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python formalize.py <text>")
        print('Example: python formalize.py "aku nk pegi kedai jap, ko nk ikut x?"')
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    print(formalize(text))
