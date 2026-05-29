"""Utility functions and resource loading."""

import json
import os

_RESOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
_CACHE = {}


def load_dictionary():
    """Load the main shortform/slang dictionary.
    
    Returns:
        dict: Full dictionary with shortforms, slang_positive, slang_negative,
              intensifiers, negators, particles, emoji_sentiment.
    
    Example:
        >>> data = manglish_nlp.load_dictionary()
        >>> data['shortforms']['nk']
        'nak'
    """
    if 'dict' not in _CACHE:
        path = os.path.join(_RESOURCE_DIR, 'dictionary.json')
        with open(path, 'r', encoding='utf-8') as f:
            _CACHE['dict'] = json.load(f)
    return _CACHE['dict']


def get_shortforms():
    """Get shortform mapping dict."""
    return load_dictionary()['shortforms']


def get_positive_words():
    """Get positive slang words."""
    return load_dictionary()['slang_positive']


def get_negative_words():
    """Get negative slang words."""
    return load_dictionary()['slang_negative']


def get_intensifiers():
    """Get intensifier words with multipliers."""
    return load_dictionary()['intensifiers']


def get_negators():
    """Get negator words."""
    return load_dictionary()['negators']


def get_particles():
    """Get Manglish particles with metadata."""
    return load_dictionary().get('particles', {})


def available_tasks():
    """List all available NLP tasks.
    
    Returns:
        list: Available task names.
    
    Example:
        >>> manglish_nlp.available_tasks()
        ['normalize', 'detect_language', 'sentiment', 'clean', 'formalize',
         'tokenize', 'stem', 'segment', 'pos_tag', 'ner_tag']
    """
    return [
        'normalize', 'detect_language', 'sentiment', 'clean', 'formalize',
        'tokenize', 'stem', 'segment', 'pos_tag', 'ner_tag'
    ]
