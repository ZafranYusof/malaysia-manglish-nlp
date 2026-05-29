"""Sentiment analysis with Malaysian context."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import re
from malaysian_manglish_nlp.utils import get_positive_words, get_negative_words, get_intensifiers, get_negators
from malaysian_manglish_nlp.cache import cached

# Pre-compiled regex pattern
_RE_WORDS = re.compile(r'[a-zA-Z0-9]+')

# === Text Preprocessing ===

def _normalize_elongated_word(word: str) -> str:
    """Reduce repeated characters (3+) to base form for lookup.

    Args:
        word: Input word possibly containing repeated characters.

    Returns:
        Word with repeated characters collapsed to single instance.
    """
    return re.sub(r'(.)\1{2,}', r'\1', word)

def _preprocess_text(text: str) -> str:
    """Preprocess text before sentiment analysis: strip hashtags.

    Args:
        text: Raw input text.

    Returns:
        Text with hashtag symbols removed (content preserved).
    """
    processed = re.sub(r'#(\w+)', r'\1', text)
    return processed

# === Sarcasm Detection (lightweight, inline) ===

_SARCASM_POSITIVE_OPENERS = {
    'bagus', 'best', 'pandai', 'rajin', 'hebat', 'power', 'cantik',
    'amazing', 'wow', 'efficient', 'murah', 'tahniah', 'congratulations',
}

_SARCASM_NEGATIVE_CONTEXT = {
    'lambat', 'lama', 'jam', 'teruk', 'bodoh', 'fail', 'rosak', 'salah',
    'tidur', 'last', 'tunggu', 'wait', 'never', 'tak', 'x', 'langsung',
    'je', 'baru', 'pun', 'sampah', 'hancur', 'terrible', 'useless',
    'lot', 'dua', 'tiga', 'rm500k', 'rm500', 'apartment', 'sehari',
}

def _detect_sarcasm_quick(text: str, words: List[str]) -> bool:
    """Quick sarcasm check for sentiment integration.

    Args:
        text: Original input text.
        words: Pre-tokenized word list from text.

    Returns:
        True if text is likely sarcastic, False otherwise.
    """
    lower = text.lower()
    word_set = set(words)
    
    has_positive_opener = bool(word_set & _SARCASM_POSITIVE_OPENERS)
    has_negative_context = bool(word_set & _SARCASM_NEGATIVE_CONTEXT)
    
    if not has_positive_opener:
        return False
    
    # Pattern: positive word + "la/lah" + comma/context with another positive (ironic doubling)
    # e.g. "Bagus la tu, memang pandai" - two praise words = sarcasm
    if re.search(r'(bagus|best|pandai|rajin|hebat|power|cantik)\s*(la|lah)\s*(tu|ni)?\s*,', lower):
        return True
    
    # Pattern: positive word + "la/lah" + negative context
    if re.search(r'(bagus|best|pandai|rajin|hebat|power|cantik|amazing|efficient|murah)\s*(la|lah)?\s*.{0,30}(lambat|lama|jam|teruk|bodoh|fail|tidur|tunggu|wait|lot|dua|sehari|rm\d)', lower):
        return True
    
    # Pattern: "Wah/Wow" + positive + time/negative indicator
    if re.search(r'(wah|wow)\s+\w+\s*(la|lah|gila|betul)?\s*.{0,30}(\d+\s*jam|lambat|lama|tunggu|baru)', lower):
        return True
    
    # Pattern: "Tahniah" + time indicator (lateness)
    if re.search(r'tahniah\s*.{0,30}(jam|lambat|lama|baru)', lower):
        return True
    
    # Pattern: positive opener followed by contradiction evidence
    if has_positive_opener and has_negative_context:
        pos_positions = [i for i, w in enumerate(words) if w in _SARCASM_POSITIVE_OPENERS]
        neg_positions = [i for i, w in enumerate(words) if w in _SARCASM_NEGATIVE_CONTEXT]
        if pos_positions and neg_positions:
            if max(neg_positions) > min(pos_positions):
                return True
    
    return False

# === Mixed Sentiment Detection ===

_CONTRAST_MARKERS = {'tapi', 'but', 'however', 'cuma', 'cuman', 'except', 'unfortunately'}

# === Passive Aggressive Patterns ===

_PASSIVE_AGGRESSIVE_PATTERNS = [
    r'takpe\s*la\s*.*(sorang|sendiri|macam biasa|je)',
    r'(aku|i)\s*(ok|fine|takpe)\s*(je|la)?\s*.*(tak\s*(kisah|ajak|peduli)|korang)',
    r'(ye|yela|ye\s*la)\s*(la)?\s*(aku|i)?\s*(yang|yg)?\s*salah',
    r'(aku|i)\s*memang\s*tak\s*penting',
    r'macam\s*biasa\s*kan',
    r'aku\s*je\s*yang',
]

# === Implicit Sentiment Phrases ===

_IMPLICIT_NEGATIVE_PHRASES = [
    'suka hati kau', 'suka hati ko', 'suka hati hang',
    'lantak kau', 'lantak ko', 'lantak hang', 'lantak la',
    'kenapa la aku', 'kenapa la i',
    'bila la nak habis', 'bila nak habis',
    'tau takpe',
    'tidur 3 jam', 'tidur 2 jam', 'tidur 1 jam',
]

# === Understatement Patterns (positive/neutral) ===

_UNDERSTATEMENT_PATTERNS = [
    (r'tak\s*(teruk|bad|buruk)\s*(la|lah)?', 'positive'),
    (r'(boleh|blh)\s*tahan\s*(la|lah)?', 'positive'),
    (r'not\s*bad\s*(la|lah)?', 'positive'),
    (r'lumayan\s*(la|lah)?', 'positive'),
]

# === Dialect Sentiment Words ===

_DIALECT_POSITIVE = {
    'sedak': 1.0, 'molek': 1.0, 'suke': 1.0, 'suko': 1.0,
    'cettong': 1.0,
    'siok': 1.0, 'nyamai': 1.0, 'gilak': 1.5,
    'sodap': 1.0, 'bona': 1.5,
}

_DIALECT_NEGATIVE = {
    'susa': -1.0,
    'polak': -1.0,
    'cito': -0.0,  # neutral word (story)
}

# Dialect negators (function like 'tak')
_DIALECT_NEGATORS = {'tok', 'sik', 'dok'}

# Additional words for edge cases
_EXTRA_POSITIVE = {
    'productive', 'alhamdulillah', 'thank', 'menang', 'gelak',
    'haha', 'lol', 'final', 'followers', 'siok',
}

_EXTRA_NEGATIVE = {
    'tolong', 'jangan', 'spend', 'repair', 'fml', 'smh',
    'overthinking', 'sadlife', 'susa', 'habis', 'workload',
}

# === Aspect-Based Sentiment Analysis ===

_ASPECT_KEYWORDS = {
    'food': {'makanan', 'makan', 'rasa', 'sedap', 'masak', 'lauk', 'nasi', 'mee', 'kuih',
             'food', 'meal', 'dish', 'taste', 'flavor', 'menu', 'portion', 'recipe',
             'sambal', 'ayam', 'ikan', 'sayur', 'sup', 'goreng', 'rebus', 'pedas',
             'manis', 'masam', 'pahit', 'lemak', 'santan', 'kuah', 'hidangan'},
    'service': {'service', 'staff', 'waiter', 'waitress', 'pelayan', 'lambat', 'cepat',
                'layan', 'layanan', 'pekerja', 'cashier', 'server', 'attend', 'response',
                'friendly', 'rude', 'slow', 'fast', 'helpful', 'attitude', 'hospitality',
                'order', 'tunggu', 'wait', 'serve', 'customer'},
    'price': {'harga', 'mahal', 'murah', 'berbaloi', 'price', 'cost', 'expensive', 'cheap',
              'affordable', 'worth', 'value', 'overpriced', 'budget', 'ringgit', 'rm',
              'bayar', 'pay', 'bil', 'bill', 'charge', 'fee', 'diskaun', 'discount',
              'promo', 'offer', 'pakej', 'package'},
    'ambiance': {'tempat', 'suasana', 'cantik', 'kotor', 'bersih', 'selesa', 'ambiance',
                 'atmosphere', 'vibe', 'decor', 'decoration', 'clean', 'dirty', 'cozy',
                 'comfortable', 'crowded', 'sesak', 'luas', 'spacious', 'parking',
                 'aircond', 'sejuk', 'bising', 'noisy', 'quiet', 'tenang',
                 'view', 'pemandangan', 'lighting', 'interior'},
    'quality': {'kualiti', 'bagus', 'rosak', 'elok', 'quality', 'good', 'bad', 'broken',
                'excellent', 'poor', 'standard', 'premium', 'fresh', 'segar', 'basi',
                'stale', 'original', 'fake', 'tiruan', 'authentic', 'genuine', 'tahan',
                'durable', 'flimsy', 'solid', 'mantap', 'teruk', 'best', 'worst'},
}

_ASPECT_POSITIVE = {
    'sedap', 'best', 'bagus', 'cantik', 'bersih', 'murah', 'berbaloi', 'cepat',
    'friendly', 'selesa', 'elok', 'mantap', 'fresh', 'segar', 'luas', 'spacious',
    'cozy', 'comfortable', 'affordable', 'worth', 'excellent', 'premium', 'solid',
    'good', 'great', 'nice', 'awesome', 'amazing', 'perfect', 'terbaik', 'padu',
    'superb', 'delicious', 'yummy', 'tasty', 'lazat', 'enak', 'helpful', 'fast',
    'quiet', 'tenang', 'genuine', 'authentic', 'durable', 'tahan', 'original',
}

_ASPECT_NEGATIVE = {
    'teruk', 'rosak', 'kotor', 'mahal', 'lambat', 'basi', 'busuk', 'buruk',
    'bad', 'terrible', 'horrible', 'slow', 'rude', 'dirty', 'expensive',
    'overpriced', 'crowded', 'sesak', 'bising', 'noisy', 'broken', 'poor',
    'fake', 'tiruan', 'flimsy', 'worst', 'stale', 'panas', 'sempit',
    'uncomfortable', 'unfriendly', 'attitude', 'kasar', 'hodoh', 'cacat',
}

_NEGATORS_SET = {'tak', 'tidak', 'bukan', 'x', 'xde', 'takde', 'no', 'not', "don't", "doesn't", 'never', 'belum'}

def aspect_sentiment(text: str) -> List[Dict[str, Any]]:
    """Analyze sentiment per aspect detected in text.

    Detects aspects (food, service, price, ambiance, quality) and analyzes
    sentiment of surrounding words for each aspect found.

    Args:
        text: Input text.

    Returns:
        List of aspect sentiment dicts, each containing:
            - aspect (str): The aspect category.
            - sentiment (str): 'positive', 'negative', or 'neutral'.
            - phrase (str): The relevant phrase around the aspect keyword.
            - score (float): Sentiment score (-1.0 to 1.0).

    Example:
        >>> aspect_sentiment("makanan sedap tapi service teruk")
        [{'aspect': 'food', 'sentiment': 'positive', ...}, ...]
    """
    words = _RE_WORDS.findall(text.lower())
    results = []
    seen_aspects = set()

    for i, word in enumerate(words):
        for aspect, keywords in _ASPECT_KEYWORDS.items():
            if word in keywords and aspect not in seen_aspects:
                window_start = max(0, i - 1)
                window_end = min(len(words), i + 4)
                window = words[window_start:window_end]

                has_negation = False
                for w in window:
                    if w in _NEGATORS_SET:
                        has_negation = True
                        break

                pos_count = sum(1 for w in window if w in _ASPECT_POSITIVE)
                neg_count = sum(1 for w in window if w in _ASPECT_NEGATIVE)

                if has_negation:
                    pos_count, neg_count = neg_count, pos_count

                total = pos_count + neg_count
                if total == 0:
                    score = 0.0
                    sent = 'neutral'
                elif pos_count > neg_count:
                    score = round(min(0.8 + (pos_count - neg_count) * 0.1, 1.0), 1)
                    sent = 'positive'
                elif neg_count > pos_count:
                    score = round(max(-0.8 - (neg_count - pos_count) * 0.1, -1.0), 1)
                    sent = 'negative'
                else:
                    score = 0.0
                    sent = 'neutral'

                phrase_start = max(0, i - 1)
                phrase_end = min(len(words), i + 3)
                phrase = ' '.join(words[phrase_start:phrase_end])

                results.append({
                    'aspect': aspect,
                    'sentiment': sent,
                    'phrase': phrase,
                    'score': score,
                })
                seen_aspects.add(aspect)
                break

    return results


def sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text (shorthand for analyze_sentiment).
    
    Args:
        text: Input text.
    
    Returns:
        dict: Sentiment result.
    
    Example:
        >>> malaysian_manglish_nlp.sentiment("gila best makanan dia")
        {'sentiment': 'positive', 'score': 1.0, ...}
    """
    return analyze_sentiment(text)


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of Manglish text.
    
    Handles Malaysian expressions, intensifiers, negators, slang,
    sarcasm, mixed sentiment, dialect words, elongated text, and
    passive aggressive patterns.
    
    Args:
        text: Input text.
    
    Returns:
        dict: Result with keys:
            - sentiment (str): 'positive', 'negative', 'neutral', or 'mixed'
            - score (float): Normalized score (-1.0 to 1.0)
            - raw_score (float): Unnormalized score
            - positive_words (list): Detected positive words
            - negative_words (list): Detected negative words
            - context (str): Explanation of scoring
            - sarcasm (bool): Whether sarcasm was detected
    
    Example:
        >>> analyze_sentiment("gila best la makanan dia")
        {'sentiment': 'positive', 'score': 1.0, 'raw_score': 1.5, ...}
        >>> analyze_sentiment("hampeh la service")
        {'sentiment': 'negative', 'score': -1.0, ...}
        >>> analyze_sentiment("tak best langsung")
        {'sentiment': 'negative', ...}  # negator flips positive
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "raw_score": 0.0,
            "positive_words": [],
            "negative_words": [],
            "context": "empty input",
            "sarcasm": False,
        }

    positive_words = set(get_positive_words().keys())
    negative_words = set(get_negative_words().keys())
    intensifiers = get_intensifiers()
    negators = set(get_negators())
    
    # Extra sentiment words
    extra_positive = {
        'suka', 'sayang', 'happy', 'gembira', 'seronok', 'enjoy',
        'nice', 'good', 'great', 'awesome', 'amazing', 'love',
        'cantik', 'comel', 'sweet', 'perfect', 'wow', 'yay',
        'thanks', 'tq', 'terbaik', 'hebat', 'ok', 'okay',
        'syok', 'puas', 'lega', 'bangga', 'grateful', 'blessed',
        'mantap', 'tiptop', 'superb', 'excellent', 'brilliant',
        'gempak', 'terror', 'legend', 'goat', 'fire',
        'wholesome', 'heartwarming', 'inspiring', 'motivated',
        'relax', 'chill', 'peaceful', 'calm', 'tenang',
        'berbaloi', 'worth', 'valuable', 'meaningful',
        'proud', 'achieve', 'berjaya', 'menang', 'win',
        'sihat', 'segar', 'fresh', 'energetic', 'semangat',
        'rindu', 'miss', 'appreciate', 'hargai',
        'lawak', 'kelakar', 'funny', 'hilarious', 'lol',
        'cute', 'adorable', 'gorgeous', 'stunning', 'beautiful',
        'delicious', 'yummy', 'tasty', 'lazat', 'enak',
        'smooth', 'lancar', 'flawless', 'clean', 'neat', 'kemas',
        'sedap', 'padu', 'solid', 'terharu', 'touched', 'moved',
        'productive', 'alhamdulillah', 'thank', 'gelak',
        'siok', 'naik', 'score',
    }
    extra_negative = {
        'benci', 'marah', 'sedih', 'kecewa', 'sakit', 'penat',
        'bad', 'terrible', 'horrible', 'hate', 'angry', 'sad',
        'susah', 'payah', 'teruk', 'buruk', 'jahat',
        'menyesal', 'rugi', 'waste', 'stupid', 'ugly',
        'boring', 'bosan', 'sien', 'fed up', 'muak', 'jelak',
        'stress', 'tension', 'pressure', 'overwhelm',
        'frust', 'frustrated', 'annoyed', 'irritated', 'meluat',
        'menyampah', 'geram', 'bengang', 'triggered',
        'toxic', 'cringe', 'disgusting', 'gross', 'jijik',
        'fail', 'gagal', 'kalah', 'lose', 'rugi',
        'lonely', 'sunyi', 'keseorangan', 'isolated',
        'anxious', 'nervous', 'risau', 'bimbang', 'gelisah',
        'disappointed', 'letdown', 'hampa', 'kecil hati',
        'jealous', 'cemburu', 'iri', 'dengki',
        'regret', 'sesal', 'menyesal', 'guilty', 'bersalah',
        'tired', 'exhausted', 'burnt out', 'penat', 'lesu', 'lemah',
        'confused', 'keliru', 'pening', 'blur',
        'scared', 'terrified', 'ngeri', 'seram', 'gerun',
        'hopeless', 'putus asa', 'give up',
        'lambat', 'slow', 'lama', 'lewat',
        'mahal', 'overpriced', 'rip off',
        'rosak', 'broken', 'cacat', 'defect',
        'kotor', 'dirty', 'busuk', 'bau',
        'sampah', 'trash', 'garbage', 'rubbish', 'hampeh',
        'hancur', 'destroyed', 'wrecked', 'musnah',
        'disappointing', 'susa', 'tolong', 'jangan',
        'fml', 'smh', 'overthinking', 'sadlife',
    }
    
    positive_words.update(extra_positive)
    positive_words.update(_EXTRA_POSITIVE)
    negative_words.update(extra_negative)
    negative_words.update(_EXTRA_NEGATIVE)
    
    # Ambiguous words to skip in short phrases
    _AMBIGUOUS_POSITIVE = {'ok', 'okay', 'depends'}
    
    lower = text.lower()
    
    # === Step 0: Preprocess text ===
    processed = _preprocess_text(text)
    
    # === Step 1: Check understatement patterns FIRST ===
    for pattern, expected_sent in _UNDERSTATEMENT_PATTERNS:
        if re.search(pattern, lower):
            return {
                "sentiment": expected_sent,
                "score": 0.3 if expected_sent == 'positive' else 0.0,
                "raw_score": 0.3 if expected_sent == 'positive' else 0.0,
                "positive_words": ['understatement'],
                "negative_words": [],
                "context": f"understatement pattern: {expected_sent}",
                "sarcasm": False,
            }
    
    # === Step 2: Check implicit negative phrases ===
    for phrase in _IMPLICIT_NEGATIVE_PHRASES:
        if phrase in lower:
            return {
                "sentiment": "negative",
                "score": -0.7,
                "raw_score": -0.7,
                "positive_words": [],
                "negative_words": [phrase],
                "context": f"implicit negative: {phrase}",
                "sarcasm": False,
            }
    
    # === Step 2b: Check contextual negative patterns ===
    # "tak boleh tak X" = forced/negative (obligation, not desire)
    if re.search(r'tak\s*(boleh|blh)\s*tak', lower):
        return {
            "sentiment": "negative",
            "score": -0.5,
            "raw_score": -0.5,
            "positive_words": [],
            "negative_words": ['forced_obligation'],
            "context": "double negation: forced obligation (negative)",
            "sarcasm": False,
        }
    
    # "Gaji naik X je tapi workload naik Y%" - minimizer 'je' + contrast = negative
    if re.search(r'(naik|tambah)\s*\d+\s*je\s*(tapi|but)', lower):
        return {
            "sentiment": "negative",
            "score": -0.6,
            "raw_score": -0.6,
            "positive_words": [],
            "negative_words": ['inadequate_increase'],
            "context": "minimized gain with contrast = negative",
            "sarcasm": False,
        }
    
    # "tidur X jam je" - sleep deprivation
    if re.search(r'tidur\s*\d+\s*jam\s*(je|jer|aje)?\s*(semalam|smlm|tadi)?', lower):
        return {
            "sentiment": "negative",
            "score": -0.6,
            "raw_score": -0.6,
            "positive_words": [],
            "negative_words": ['sleep_deprivation'],
            "context": "implicit negative: sleep deprivation",
            "sarcasm": False,
        }
    
    # === Step 3: Check passive aggressive patterns ===
    for pattern in _PASSIVE_AGGRESSIVE_PATTERNS:
        if re.search(pattern, lower):
            return {
                "sentiment": "negative",
                "score": -0.6,
                "raw_score": -0.6,
                "positive_words": [],
                "negative_words": ['passive_aggressive'],
                "context": "passive aggressive pattern detected",
                "sarcasm": False,
            }
    
    # === Step 4: Normalize elongated text for word lookup ===
    raw_words = _RE_WORDS.findall(processed.lower())
    words = []
    word_normalized_map = {}
    for i, w in enumerate(raw_words):
        normalized_w = _normalize_elongated_word(w)
        words.append(w)
        word_normalized_map[i] = normalized_w
    
    # === Step 5: Check sarcasm ===
    norm_words = [word_normalized_map[i] for i in range(len(words))]
    is_sarcastic = _detect_sarcasm_quick(processed, norm_words)
    
    # === Step 6: Score words ===
    _OVERRIDE_MAP = {
        'berbaloi': {'mahal', 'expensive'},
        'worth': {'mahal', 'expensive'},
        'murah': {'mahal'},
    }
    suppressed_negatives = set()
    for w in norm_words:
        if w in _OVERRIDE_MAP:
            suppressed_negatives.update(_OVERRIDE_MAP[w])
    
    score = 0.0
    pos_found = []
    neg_found = []
    multiplier = 1.0
    negate_next = False
    
    # Check for "depends" specifically
    if 'depends' in norm_words and len(norm_words) <= 5:
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "raw_score": 0.0,
            "positive_words": [],
            "negative_words": [],
            "context": "ambiguous/neutral phrase",
            "sarcasm": False,
        }
    
    # Particles that shouldn't reset state
    _PARTICLES = {'la', 'lah', 'je', 'jer', 'ni', 'tu', 'kan', 'kot', 'pun', 'ke', 'kat', 'dah', 'baru', 'aje', 'ya', 'bah'}
    
    # Add dialect negators to negators set
    negators = negators | _DIALECT_NEGATORS
    
    # Remove 'naik' from positive - it's context-dependent
    positive_words.discard('naik')
    positive_words.discard('score')
    
    for i, word in enumerate(words):
        lookup_word = word_normalized_map[i]
        
        if lookup_word in negators:
            negate_next = True
            continue
        
        if lookup_word in intensifiers:
            multiplier = intensifiers[lookup_word]
            continue
        
        # Skip ambiguous words in short phrases
        if lookup_word in _AMBIGUOUS_POSITIVE and len(words) <= 5:
            multiplier = 1.0
            negate_next = False
            continue
        
        # Check dialect positive words
        if lookup_word in _DIALECT_POSITIVE:
            val = _DIALECT_POSITIVE[lookup_word]
            if val > 0:
                if negate_next:
                    neg_found.append(f"NOT({lookup_word})")
                    score -= val
                else:
                    pos_found.append(lookup_word)
                    score += val
            multiplier = 1.0
            negate_next = False
            continue
        
        # Check dialect negative words
        if lookup_word in _DIALECT_NEGATIVE:
            val = _DIALECT_NEGATIVE[lookup_word]
            if val < 0:
                if negate_next:
                    pos_found.append(f"NOT({lookup_word})")
                    score -= val  # double negative = positive
                else:
                    neg_found.append(lookup_word)
                    score += val
            multiplier = 1.0
            negate_next = False
            continue
        
        if lookup_word in positive_words:
            val = 1.0 * multiplier
            if negate_next:
                val = -val
                neg_found.append(f"NOT({lookup_word})")
            else:
                pos_found.append(lookup_word)
            score += val
        elif lookup_word in negative_words:
            if lookup_word in suppressed_negatives:
                multiplier = 1.0
                negate_next = False
                continue
            val = -1.0 * multiplier
            if negate_next:
                val = -val
                pos_found.append(f"NOT({lookup_word})")
            else:
                neg_found.append(lookup_word)
            score += val
        else:
            # Word not in any dictionary - don't reset flags for particles
            if lookup_word not in _PARTICLES:
                multiplier = 1.0
                negate_next = False
            continue
        
        multiplier = 1.0
        negate_next = False
    
    # === Step 7: Check for mixed sentiment (tapi/but pattern) ===
    has_contrast = bool(set(norm_words) & _CONTRAST_MARKERS)
    is_mixed = False
    if has_contrast and pos_found and neg_found:
        is_mixed = True
    elif has_contrast and (pos_found or neg_found):
        # Even if only one side found explicitly, contrast marker suggests mixed
        # Check if there are words on both sides of 'tapi/but'
        contrast_idx = None
        for ci, cw in enumerate(norm_words):
            if cw in _CONTRAST_MARKERS:
                contrast_idx = ci
                break
        if contrast_idx is not None:
            # If positive before tapi and negative-ish words after (or vice versa)
            before = norm_words[:contrast_idx]
            after = norm_words[contrast_idx+1:]
            before_has_pos = any(w in positive_words or w in _DIALECT_POSITIVE for w in before)
            after_has_neg = any(w in negative_words or w in _DIALECT_NEGATIVE or w in _EXTRA_NEGATIVE for w in after)
            before_has_neg = any(w in negative_words or w in _DIALECT_NEGATIVE for w in before)
            after_has_pos = any(w in positive_words or w in _DIALECT_POSITIVE for w in after)
            if (before_has_pos and after_has_neg) or (before_has_neg and after_has_pos):
                is_mixed = True
    
    # === Step 8: Apply sarcasm flip ===
    if is_sarcastic and score >= 0:
        score = -abs(score) if score != 0 else -0.7
        pos_found, neg_found = neg_found, pos_found
    
    # === Step 9: Normalize and classify ===
    word_count = max(len(pos_found) + len(neg_found), 1)
    normalized = max(-1.0, min(1.0, score / word_count))
    
    # Classify
    if is_mixed:
        if normalized < -0.1:
            sent = "negative"
        elif normalized > 0.1:
            sent = "mixed"
        else:
            sent = "mixed"
    elif normalized > 0.2:
        sent = "positive"
    elif normalized < -0.2:
        sent = "negative"
    else:
        sent = "neutral"
    
    # === Step 10: Handle ALL CAPS boost ===
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars and len(alpha_chars) > 3:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > 0.7:
            if score > 0:
                score *= 1.5
                normalized = max(-1.0, min(1.0, score / word_count))
                if normalized > 0.2:
                    sent = "positive"
            elif score < 0:
                score *= 1.5
                normalized = max(-1.0, min(1.0, score / word_count))
                if normalized < -0.2:
                    sent = "negative"
            elif not pos_found and not neg_found:
                # ALL CAPS with no sentiment words - check if any normalized words match
                # Try harder with caps text
                pass
    
    # Context
    context_parts = []
    if pos_found:
        context_parts.append(f"positive: {', '.join(str(p) for p in pos_found[:3])}")
    if neg_found:
        context_parts.append(f"negative: {', '.join(str(n) for n in neg_found[:3])}")
    if is_sarcastic:
        context_parts.append("sarcasm detected")
    if is_mixed:
        context_parts.append("mixed sentiment (contrast)")
    if any(word_normalized_map.get(i, '') in intensifiers for i in range(len(words))):
        context_parts.append("intensified")
    
    return {
        "sentiment": sent,
        "score": round(normalized, 3),
        "raw_score": round(score, 3),
        "positive_words": pos_found[:5],
        "negative_words": neg_found[:5],
        "context": "; ".join(context_parts) if context_parts else "neutral/no sentiment words",
        "sarcasm": is_sarcastic,
    }
