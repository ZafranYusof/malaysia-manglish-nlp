#!/usr/bin/env python3
"""
Manglish sentiment analyzer.
Handles Malaysian expressions, intensifiers, negators, and slang.
Usage: python sentiment.py "gila best makanan dia"
"""

import sys
import re
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(SCRIPT_DIR, '..', 'references', 'shortform-dict.json')


def load_lexicon():
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_sentiment(text):
    """Analyze sentiment of Manglish text."""
    data = load_lexicon()
    positive_words = set(data.get('slang_positive', {}).keys())
    negative_words = set(data.get('slang_negative', {}).keys())
    intensifiers = data.get('intensifiers', {})
    negators = set(data.get('negators', []))
    
    # Additional sentiment words
    extra_positive = {
        'suka', 'sayang', 'happy', 'gembira', 'seronok', 'enjoy',
        'nice', 'good', 'great', 'awesome', 'amazing', 'love',
        'cantik', 'comel', 'sweet', 'perfect', 'wow', 'yay',
        'thanks', 'terima kasih', 'tq', 'terbaik', 'hebat',
    }
    extra_negative = {
        'benci', 'marah', 'sedih', 'kecewa', 'sakit', 'penat',
        'bad', 'terrible', 'horrible', 'hate', 'angry', 'sad',
        'susah', 'payah', 'teruk', 'buruk', 'jahat', 'bodoh',
        'menyesal', 'rugi', 'waste', 'stupid', 'ugly',
    }
    
    positive_words.update(extra_positive)
    negative_words.update(extra_negative)
    
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    
    score = 0.0
    pos_found = []
    neg_found = []
    multiplier = 1.0
    negate_next = False
    
    for i, word in enumerate(words):
        # Check negator
        if word in negators:
            negate_next = True
            continue
        
        # Check intensifier
        if word in intensifiers:
            multiplier = intensifiers[word]
            continue
        
        # Score word
        if word in positive_words:
            val = 1.0 * multiplier
            if negate_next:
                val = -val
                neg_found.append(f"NOT({word})")
            else:
                pos_found.append(word)
            score += val
        elif word in negative_words:
            val = -1.0 * multiplier
            if negate_next:
                val = -val
                pos_found.append(f"NOT({word})")
            else:
                neg_found.append(word)
            score += val
        
        # Reset
        multiplier = 1.0
        negate_next = False
    
    # Normalize score to -1 to 1 range
    word_count = max(len(pos_found) + len(neg_found), 1)
    normalized = max(-1.0, min(1.0, score / word_count))
    
    # Classify
    if normalized > 0.2:
        sentiment = "positive"
    elif normalized < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Context explanation
    context_parts = []
    if pos_found:
        context_parts.append(f"positive: {', '.join(pos_found[:3])}")
    if neg_found:
        context_parts.append(f"negative: {', '.join(neg_found[:3])}")
    if any(w in intensifiers for w in words):
        context_parts.append("intensified")
    
    return {
        "sentiment": sentiment,
        "score": round(normalized, 3),
        "raw_score": round(score, 3),
        "positive_words": pos_found[:5],
        "negative_words": neg_found[:5],
        "context": "; ".join(context_parts) if context_parts else "neutral/no sentiment words"
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sentiment.py <text>")
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    result = analyze_sentiment(text)
    print(json.dumps(result, indent=2))
