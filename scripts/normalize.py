#!/usr/bin/env python3
"""
Manglish text normalizer - expands shortforms/slang to standard BM/EN.
Usage: python normalize.py "nk tnya brapa sem utk grad"
"""

import json
import sys
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(SCRIPT_DIR, '..', 'references', 'shortform-dict.json')

def load_dict():
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['shortforms']

def normalize(text, shortforms=None):
    """Normalize Manglish shortforms to standard form."""
    if shortforms is None:
        shortforms = load_dict()
    
    # Split preserving punctuation
    tokens = re.findall(r"[\w']+|[^\w\s]", text.lower())
    result = []
    
    for token in tokens:
        # Check exact match
        if token in shortforms:
            result.append(shortforms[token])
        # Check without trailing numbers (e.g., "nk2" -> "nak")
        elif re.sub(r'\d+$', '', token) in shortforms:
            result.append(shortforms[re.sub(r'\d+$', '', token)])
        else:
            result.append(token)
    
    # Rejoin with proper spacing (no space before punctuation)
    output = ''
    for i, token in enumerate(result):
        if i == 0:
            output = token
        elif token in '.,!?;:':
            output += token
        else:
            output += ' ' + token
    
    return output

def normalize_preserve_case(text, shortforms=None):
    """Normalize but try to preserve original casing."""
    if shortforms is None:
        shortforms = load_dict()
    
    words = text.split()
    result = []
    
    for word in words:
        lower = word.lower().strip('.,!?;:')
        punct = ''
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
        
        if lower in shortforms:
            expanded = shortforms[lower]
            # Preserve capitalization
            if word[0].isupper():
                expanded = expanded.capitalize()
            result.append(expanded + punct)
        else:
            result.append(word)
    
    return ' '.join(result)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python normalize.py <text>")
        print("Example: python normalize.py \"nk tnya brapa sem utk grad\"")
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    print(normalize(text))
