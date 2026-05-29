#!/usr/bin/env python3
"""
Manglish text cleaner.
Removes noise: repeated chars, excessive emoji, normalize spacing.
Usage: python clean.py "besttttttt gilerrrr ahhhh 😂😂😂😂"
"""

import sys
import re


def clean_text(text):
    """Clean noisy Manglish text."""
    result = text
    
    # 1. Reduce repeated characters (3+ -> 1)
    # "besttttt" -> "best", "gilerrrr" -> "giler"
    result = re.sub(r'(.)\1{2,}', r'\1', result)
    
    # 2. Reduce repeated emoji (3+ -> 1)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    
    # Reduce consecutive same emoji to max 1
    def reduce_emoji(match):
        emojis = match.group(0)
        # Keep unique emojis, max 2 total
        seen = []
        for char in emojis:
            if char not in seen and len(seen) < 2:
                seen.append(char)
        return ''.join(seen)
    
    result = emoji_pattern.sub(reduce_emoji, result)
    
    # 3. Normalize whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    
    # 4. Remove excessive punctuation (??? -> ?, !!! -> !)
    result = re.sub(r'([!?.]){2,}', r'\1', result)
    
    # 5. Fix common typo patterns
    # Double first letter: "mmakan" -> "makan"
    result = re.sub(r'\b([a-zA-Z])\1([a-zA-Z]{3,})\b', r'\1\2', result)
    
    # 6. Normalize common variations
    result = re.sub(r'\bhaha(ha)+\b', 'haha', result, flags=re.IGNORECASE)
    result = re.sub(r'\bwkwk(wk)+\b', 'wkwk', result, flags=re.IGNORECASE)
    result = re.sub(r'\blol(ol)+\b', 'lol', result, flags=re.IGNORECASE)
    
    return result


def clean_for_nlp(text):
    """Aggressive cleaning for NLP pipeline."""
    result = clean_text(text)
    
    # Remove all emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    result = emoji_pattern.sub('', result)
    
    # Remove URLs
    result = re.sub(r'https?://\S+', '', result)
    
    # Remove mentions/hashtags
    result = re.sub(r'[@#]\w+', '', result)
    
    # Normalize whitespace again
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean.py <text>")
        print("       python clean.py --nlp <text>  (aggressive mode)")
        sys.exit(1)
    
    if sys.argv[1] == '--nlp':
        text = ' '.join(sys.argv[2:])
        print(clean_for_nlp(text))
    else:
        text = ' '.join(sys.argv[1:])
        print(clean_text(text))
