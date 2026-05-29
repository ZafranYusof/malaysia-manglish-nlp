"""Fast tokenizer with C extension, fallback to pure Python.

Provides the same API as ``manglish_nlp.tokenizer`` but uses a compiled C
extension for 5-10x speedup on tokenization, sentence splitting, and
normalization.  Falls back transparently to the pure-Python implementation
when the C extension is unavailable (e.g. on platforms where compilation
fails or is skipped).

Usage::

    from manglish_nlp.tokenizer_fast import tokenize, split_sentences, normalize

    tokens = tokenize("aku nk pergi la weh!")
    sents  = split_sentences("Aku pergi. Ko ikut? Jom!")
    norm   = normalize("  Aku   NAK   pergi  ")

All functions are drop-in compatible with the originals in ``tokenizer.py``.
"""
from __future__ import annotations

import re
import warnings
from typing import , Dict, List

# ---------------------------------------------------------------------------
# Try importing the C extension
# ---------------------------------------------------------------------------

try:
    from manglish_nlp._tokenizer_fast import (
        fast_tokenize as _c_tokenize,
        fast_split_sentences as _c_split_sentences,
        fast_normalize as _c_normalize,
    )
    HAS_C_EXT: bool = True
except ImportError:
    HAS_C_EXT = False

# ---------------------------------------------------------------------------
# Pure-Python fallback (mirrors tokenizer.py logic)
# ---------------------------------------------------------------------------

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

_SENT_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_SENT_END_LOOSE = re.compile(r'(?<=[.!?])\s+')


def _py_tokenize(text: str) -> List[str]:
    """Pure-Python word tokenization fallback."""
    return _TOKEN_PATTERN.findall(text)


def _py_split_sentences(text: str) -> List[str]:
    """Pure-Python sentence splitting fallback."""
    sentences = _SENT_END.split(text)
    if len(sentences) <= 1:
        sentences = _SENT_END_LOOSE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _py_normalize(text: str) -> str:
    """Pure-Python normalization fallback."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenize *text* into a list of word/punctuation/emoji tokens.

    Uses the C extension when available for ~10x speedup; otherwise falls
    back to the regex-based pure-Python tokenizer.

    Args:
        text: Input text to tokenize.

    Returns:
        List of token strings.

    Example::

        >>> tokenize("aku nk pergi la weh!")
        ['aku', 'nk', 'pergi', 'la', 'weh', '!']
    """
    if not text:
        return []
    if HAS_C_EXT:
        return _c_tokenize(text)
    return _py_tokenize(text)


def split_sentences(text: str) -> List[str]:
    """Split *text* into a list of sentence strings.

    Args:
        text: Input text containing one or more sentences.

    Returns:
        List of sentence strings (stripped of surrounding whitespace).

    Example::

        >>> split_sentences("Aku pergi. Ko ikut? Jom!")
        ['Aku pergi.', 'Ko ikut?', 'Jom!']
    """
    if not text:
        return []
    if HAS_C_EXT:
        return _c_split_sentences(text)
    return _py_split_sentences(text)


def normalize(text: str) -> str:
    """Normalize *text*: lowercase, collapse whitespace, strip edges.

    Args:
        text: Input text to normalize.

    Returns:
        Normalized string.

    Example::

        >>> normalize("  Aku   NAK   pergi  ")
        'aku nak pergi'
    """
    if not text:
        return ''
    if HAS_C_EXT:
        return _c_normalize(text)
    return _py_normalize(text)


def is_c_extension_available() -> bool:
    """Return ``True`` if the C extension is loaded and active.

    Useful for tests and benchmarking to confirm which backend is in use.
    """
    return HAS_C_EXT


def benchmark(text: str, iterations: int = 1000) -> dict:
    """Benchmark C extension vs pure-Python tokenization.

    Args:
        text: Sample text to tokenize repeatedly.
        iterations: Number of iterations per backend.

    Returns:
        Dict with keys ``c_ext``, ``python``, ``speedup``, and ``available``.
        Times are in seconds.  If C extension is unavailable, ``c_ext`` and
        ``speedup`` are ``None``.
    """
    import time

    results: dict = {'available': HAS_C_EXT, 'iterations': iterations}

    # Pure Python
    t0 = time.perf_counter()
    for _ in range(iterations):
        _py_tokenize(text)
    results['python'] = time.perf_counter() - t0

    if HAS_C_EXT:
        t0 = time.perf_counter()
        for _ in range(iterations):
            _c_tokenize(text)
        results['c_ext'] = time.perf_counter() - t0
        results['speedup'] = results['python'] / max(results['c_ext'], 1e-9)
    else:
        results['c_ext'] = None
        results['speedup'] = None

    return results


# Alias for backward compat
word_tokenize = tokenize
sentence_tokenize = split_sentences
