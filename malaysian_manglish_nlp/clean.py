"""Text cleaning for noisy Manglish text."""

from __future__ import annotations

from typing import Any

import re

# Pre-compiled regex patterns for performance
_RE_REPEATED_CHARS = re.compile(r'(.)\1{2,}')
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)
_RE_WHITESPACE = re.compile(r'\s+')
_RE_EXCESSIVE_PUNCT = re.compile(r'([!?.]){2,}')
_RE_DOUBLE_FIRST_LETTER = re.compile(r'\b([a-zA-Z])\1([a-zA-Z]{3,})\b')
_RE_HAHA = re.compile(r'\bhaha(ha)+\b', re.IGNORECASE)
_RE_WKWK = re.compile(r'\bwkwk(wk)+\b', re.IGNORECASE)
_RE_LOL = re.compile(r'\blol(ol)+\b', re.IGNORECASE)
_RE_URL = re.compile(r'https?://\S+')
_RE_MENTION_HASHTAG = re.compile(r'[@#]\w+')


def clean(text: str) -> str:
    """Clean noisy Manglish text (alias for clean_text).
    
    Parameters:
        text (str): Noisy input text.
    
    Returns:
        str: Cleaned text.
    
    Example:
        >>> malaysian_manglish_nlp.clean("besttttttt gilerrrr 😂😂😂😂")
        'best giler 😂'
    """
    return clean_text(text)

def clean_text(text: str) -> str:
    """Clean noisy text while preserving meaning.
    
    Operations:
    1. Reduce repeated characters (3+ to 1)
    2. Reduce repeated emoji (keep max 2 unique)
    3. Normalize whitespace
    4. Reduce excessive punctuation
    5. Fix double-first-letter typos
    6. Normalize laugh patterns
    
    Parameters:
        text (str): Input text.
    
    Returns:
        str: Cleaned text.
    """
    result = text
    
    # Reduce repeated chars
    result = _RE_REPEATED_CHARS.sub(r'\1', result)
    
    # Reduce repeated emoji
    def reduce_emoji(match: Any) -> str:
        """Reduce repeated emoji characters.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        emojis = match.group(0)
        seen = []
        for char in emojis:
            if char not in seen and len(seen) < 2:
                seen.append(char)
        return ''.join(seen)
    
    result = _RE_EMOJI.sub(reduce_emoji, result)
    
    # Normalize whitespace
    result = _RE_WHITESPACE.sub(' ', result).strip()
    
    # Reduce excessive punctuation
    result = _RE_EXCESSIVE_PUNCT.sub(r'\1', result)
    
    # Fix double first letter
    result = _RE_DOUBLE_FIRST_LETTER.sub(r'\1\2', result)
    
    # Normalize laughs
    result = _RE_HAHA.sub('haha', result)
    result = _RE_WKWK.sub('wkwk', result)
    result = _RE_LOL.sub('lol', result)
    
    return result

def clean_for_nlp(text: str) -> str:
    """Aggressive cleaning for NLP/ML pipelines.
    
    Additional operations beyond clean_text:
    - Remove all emoji
    - Remove URLs
    - Remove mentions (@) and hashtags (#)
    - Lowercase
    
    Parameters:
        text (str): Input text.
    
    Returns:
        str: Aggressively cleaned text.
    
    Example:
        >>> clean_for_nlp("Check https://t.co/abc @user #trending besttt 😂")
        'check best'
    """
    result = clean_text(text)
    
    # Remove emoji
    result = _RE_EMOJI.sub('', result)
    
    # Remove URLs
    result = _RE_URL.sub('', result)
    
    # Remove mentions/hashtags
    result = _RE_MENTION_HASHTAG.sub('', result)
    
    # Normalize whitespace
    result = _RE_WHITESPACE.sub(' ', result).strip()
    
    return result
