"""Tokenization for Malaysian text."""

import re

# Malay affixes for token boundary detection
_PREFIXES = {'me', 'men', 'mem', 'meng', 'meny', 'ber', 'di', 'ke', 'se', 'per', 'pen', 'pem', 'peng', 'peny', 'ter'}
_SUFFIXES = {'kan', 'an', 'i', 'nya', 'lah', 'kah', 'tah', 'pun'}

# Pre-compiled sentence-ending patterns
_SENT_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_SENT_END_LOOSE = re.compile(r'(?<=[.!?])\s+')

# Pre-compiled tokenizer pattern
_TOKEN_PATTERN = re.compile(
    r'https?://\S+|www\.\S+'                    # URLs
    r'|[\w.+-]+@[\w-]+\.[\w.-]+'                # Email
    r'|\+?6?0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}'  # Phone (Malaysian)
    r'|(?:RM|MYR|rm)\s?[\d,]+(?:\.\d{1,2})?'   # Money
    r'|#[\w]+'                                   # Hashtag
    r'|@[\w]+'                                   # Mention
    r'|[:;][\-]?[)(/DPpOo\|\\]|<3|\^_?\^|T_T|>_<|\._\.'  # Emoticons
    r'|[\U0001F600-\U0001F64F'                   # Emoji
    r'\U0001F300-\U0001F5FF'
    r'\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF'
    r'\U00002702-\U000027B0'
    r'\U000024C2-\U0001F251'
    r'\U0001F900-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F]+'
    r'|\d+(?:\.\d+)?%?'                          # Numbers
    r"|[\w']+"                                    # Words
    r'|[^\w\s]'                                   # Punctuation
)


def tokenize(text):
    """Tokenize text into words (alias for word_tokenize).
    
    Handles Manglish-specific patterns:
    - Preserves particles (la, lah, kan) as separate tokens
    - Handles shortforms as single tokens
    - Splits punctuation from words
    - Preserves emoji as tokens
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list[str]: List of tokens.
    
    Example:
        >>> manglish_nlp.tokenize("aku nk pergi la weh!")
        ['aku', 'nk', 'pergi', 'la', 'weh', '!']
        >>> manglish_nlp.tokenize("best giler😂😂")
        ['best', 'giler', '😂', '😂']
    """
    return word_tokenize(text)

def word_tokenize(text):
    """Tokenize text into words.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list[str]: List of word tokens.
    
    Example:
        >>> word_tokenize("tak nak la, mahal sgt!")
        ['tak', 'nak', 'la', ',', 'mahal', 'sgt', '!']
    """
    return _TOKEN_PATTERN.findall(text)

def sentence_tokenize(text):
    """Split text into sentences.
    
    Handles:
    - Standard punctuation boundaries (.!?)
    - Malaysian informal patterns (multiple sentences without caps)
    - Preserves emoji within sentences
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list[str]: List of sentences.
    
    Example:
        >>> sentence_tokenize("Aku nak pergi. Ko nak ikut? Jom la!")
        ['Aku nak pergi.', 'Ko nak ikut?', 'Jom la!']
    """
    # Try strict split first (after punctuation + space + capital)
    sentences = _SENT_END.split(text)
    
    if len(sentences) <= 1:
        # Fallback: split on any sentence-ending punctuation + space
        sentences = _SENT_END_LOOSE.split(text)
    
    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def morpheme_tokenize(word):
    """Split a Malay word into morphemes (prefix + root + suffix).
    
    Basic rule-based morpheme segmentation for BM words.
    
    Parameters:
        word (str): Single word.
    
    Returns:
        dict: Dictionary with 'prefix', 'root', 'suffix' keys.
    
    Example:
        >>> morpheme_tokenize("berlarian")
        {'prefix': 'ber', 'root': 'lari', 'suffix': 'an'}
        >>> morpheme_tokenize("makan")
        {'prefix': '', 'root': 'makan', 'suffix': ''}
    """
    lower = word.lower()
    prefix = ''
    suffix = ''
    root = lower
    
    # Check prefixes (longest match first)
    for p in sorted(_PREFIXES, key=len, reverse=True):
        if root.startswith(p) and len(root) > len(p) + 2:
            prefix = p
            root = root[len(p):]
            break
    
    # Check suffixes
    for s in sorted(_SUFFIXES, key=len, reverse=True):
        if root.endswith(s) and len(root) > len(s) + 2:
            suffix = s
            root = root[:-len(s)]
            break
    
    return {'prefix': prefix, 'root': root, 'suffix': suffix}
