"""Text normalization - expand shortforms/slang to standard form."""

from __future__ import annotations

import re
from malaysian_manglish_nlp.utils import get_shortforms
from malaysian_manglish_nlp.cache import cached

# Pre-compiled regex patterns
_RE_TOKENS = re.compile(r"[\w']+|[^\w\s]")
_RE_TRAILING_DIGITS = re.compile(r'\d+$')


@cached(maxsize=1024)
def normalize(text: str, preserve_case: bool = False) -> str:
    """Normalize Manglish shortforms to standard BM/EN.
    
    Parameters:
        text (str): Input text containing shortforms.
        preserve_case (bool): If True, attempt to preserve original casing.
    
    Returns:
        str: Text with shortforms expanded.
    
    Example:
        >>> malaysian_manglish_nlp.normalize("nk tnya brapa sem utk grad")
        'nak tanya berapa semester untuk grad'
        >>> malaysian_manglish_nlp.normalize("Nk pergi", preserve_case=True)
        'Nak pergi'
    """
    if preserve_case:
        return normalize_preserve_case(text)
    
    shortforms = get_shortforms()
    tokens = _RE_TOKENS.findall(text.lower())
    result = []
    
    for token in tokens:
        if token in shortforms:
            result.append(shortforms[token])
        elif _RE_TRAILING_DIGITS.sub('', token) in shortforms:
            result.append(shortforms[_RE_TRAILING_DIGITS.sub('', token)])
        else:
            result.append(token)
    
    output = ''
    for i, token in enumerate(result):
        if i == 0:
            output = token
        elif token in '.,!?;:':
            output += token
        else:
            output += ' ' + token
    
    return output


def normalize_preserve_case(text: str) -> str:
    """Normalize shortforms while preserving original casing.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        str: Normalized text with case preserved.
    
    Example:
        >>> normalize_preserve_case("Nk Pergi Kedai")
        'Nak Pergi Kedai'
    """
    shortforms = get_shortforms()
    words = text.split()
    result = []
    
    for word in words:
        lower = word.lower().strip('.,!?;:')
        punct = ''
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
        
        if lower in shortforms:
            expanded = shortforms[lower]
            if word[0].isupper():
                expanded = expanded.capitalize()
            result.append(expanded + punct)
        else:
            result.append(word)
    
    return ' '.join(result)
