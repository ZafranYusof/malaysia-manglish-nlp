"""
Intent classification for Manglish text.

Classifies text into 8 intent categories using rule-based pattern matching
with confidence scoring. Handles Malaysian shortforms and code-switching.

Categories:
    question    - Asking for information
    request     - Asking someone to do something
    complaint   - Expressing dissatisfaction
    greeting    - Social greetings
    opinion     - Expressing views
    statement   - Neutral factual
    command     - Direct instruction
    offer       - Offering something
"""

import re
from typing import Dict, List, Optional

# Shortform expansions relevant to intent detection
_SHORTFORMS = {
    'nk': 'nak', 'nk': 'nak', 'blh': 'boleh', 'tlg': 'tolong',
    'mcm': 'macam', 'cmne': 'macam mana', 'cmna': 'macam mana',
    'ape': 'apa', 'ap': 'apa', 'pe': 'apa', 'knp': 'kenapa',
    'knpe': 'kenapa', 'bpe': 'kenapa', 'npe': 'kenapa',
    'bile': 'bila', 'bl': 'bila', 'mn': 'mana', 'mne': 'mana',
    'spe': 'siapa', 'sp': 'siapa', 'brp': 'berapa', 'brpe': 'berapa',
    'x': 'tak', 'xde': 'takde', 'tk': 'tak', 'tkleh': 'tak boleh',
    'pls': 'please', 'plz': 'please', 'hlp': 'help',
    'jgn': 'jangan', 'prgi': 'pergi', 'pgi': 'pergi',
    'tlong': 'tolong', 'mntk': 'minta', 'bgi': 'bagi',
}

# Question markers
_QUESTION_WORDS_BM = {
    'apa', 'ape', 'berapa', 'bila', 'mana', 'siapa', 'kenapa',
    'macam mana', 'camne', 'cmne', 'cmna', 'mengapa', 'bagaimana',
}
_QUESTION_WORDS_EN = {
    'what', 'when', 'where', 'who', 'why', 'how', 'which',
    'whose', 'whom',
}
_QUESTION_PARTICLES = {'ke', 'ka', 'kah', 'tak', 'x', 'kan', 'eh'}

# Request markers
_REQUEST_MARKERS = {
    'tolong', 'tlg', 'tlong', 'please', 'pls', 'plz',
    'boleh tak', 'boleh x', 'blh tak', 'blh x',
    'can you', 'can u', 'could you',
    'bagi', 'bgi', 'help', 'hlp', 'minta', 'mntk',
    'would you', 'mind if',
}

# Complaint markers
_COMPLAINT_MARKERS = {
    'teruk', 'bodoh', 'slow', 'lambat', 'marah', 'fed up',
    'annoyed', 'disappointed', 'tak puas hati', 'x puas hati',
    'geram', 'bengang', 'benci', 'sucks', 'terrible', 'worst',
    'horrible', 'rubbish', 'sampah', 'useless', 'hopeless',
    'menyampah', 'meluat', 'hampa', 'celaka', 'sial',
    'damn', 'wtf', 'bullshit', 'nonsense', 'ridiculous',
}

# Greeting markers
_GREETING_MARKERS = {
    'hi', 'hello', 'hai', 'hey', 'helo',
    'assalamualaikum', 'salam', 'aslkm',
    'morning', 'evening', 'afternoon', 'night',
    'good morning', 'good evening', 'good night',
    'selamat pagi', 'selamat petang', 'selamat malam',
    'oi', 'weh', 'wei', 'yo', 'sup', 'hii', 'hiii',
}

# Opinion markers
_OPINION_MARKERS = {
    'aku rasa', 'i think', 'i feel', 'pada aku', 'bagi aku',
    'on my opinion', 'in my opinion', 'imo', 'imho',
    'best gila', 'worst gila', 'ok la', 'okay la', 'okla',
    'tak berbaloi', 'berbaloi', 'worth it', 'not worth',
    'overrated', 'underrated', 'patut', 'sepatutnya',
    'should', 'shouldnt', "shouldn't",
    'personally', 'for me', 'kalau aku',
}

# Command markers (imperative verbs typically at start)
_COMMAND_VERBS = {
    'pergi', 'pgi', 'prgi', 'tutup', 'buka', 'stop', 'go',
    'come', 'sit', 'duduk', 'diam', 'shut up', 'jangan',
    'jgn', 'keluar', 'masuk', 'ambil', 'letak', 'buang',
    'delete', 'remove', 'close', 'open', 'run', 'do',
    'berhenti', 'tunggu', 'wait', 'listen', 'dengar',
    'cepat', 'hurry', 'move', 'gerak', 'senyap',
}

# Offer markers
_OFFER_MARKERS = {
    'nak aku', 'nk aku', 'jom', 'let me', 'i can', 'i could',
    'boleh aku', 'blh aku', 'aku boleh', 'aku blh',
    'shall i', 'want me to', 'how about i',
    'aku tolong', 'aku tlg', 'nak tak aku',
    'kalau nak aku', 'if you want i',
}


def _expand_shortforms(text: str) -> str:
    """Expand common Manglish shortforms for better intent detection."""
    words = text.split()
    expanded = []
    for word in words:
        lower = word.lower()
        if lower in _SHORTFORMS:
            expanded.append(_SHORTFORMS[lower])
        else:
            expanded.append(word)
    return ' '.join(expanded)


def _normalize_text(text: str) -> str:
    """Lowercase and basic normalization for matching."""
    return text.lower().strip()


def get_intent_features(text: str) -> Dict:
    """
    Extract features used for intent classification.

    Args:
        text: Input Manglish text.

    Returns:
        Dict with feature counts and flags used for classification.
    """
    normalized = _normalize_text(text)
    expanded = _expand_shortforms(normalized)

    features = {
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'question_words': 0,
        'question_particles': 0,
        'request_markers': 0,
        'complaint_markers': 0,
        'greeting_markers': 0,
        'opinion_markers': 0,
        'command_verbs_at_start': False,
        'offer_markers': 0,
        'word_count': len(text.split()),
        'is_short': len(text.split()) <= 4,
        'ends_with_question_particle': False,
        'has_question_mark': '?' in text,
    }

    # Question words
    for qw in _QUESTION_WORDS_BM | _QUESTION_WORDS_EN:
        if qw in expanded.split() or expanded.startswith(qw + ' ') or f' {qw} ' in f' {expanded} ':
            features['question_words'] += 1

    # Question particles at end
    words = expanded.split()
    if words:
        last_word = words[-1].rstrip('?').rstrip('!')
        if last_word in _QUESTION_PARTICLES:
            features['question_particles'] += 1
            features['ends_with_question_particle'] = True

    # Request markers
    for marker in _REQUEST_MARKERS:
        if marker in expanded:
            features['request_markers'] += 1

    # Complaint markers
    for marker in _COMPLAINT_MARKERS:
        if marker in expanded or marker in normalized:
            features['complaint_markers'] += 1

    # Greeting markers
    # For greetings, check if the text starts with or is primarily a greeting
    for marker in _GREETING_MARKERS:
        if normalized == marker or normalized.startswith(marker + ' ') or normalized.startswith(marker + ','):
            features['greeting_markers'] += 1
        elif f' {marker}' == f' {normalized}' or normalized == marker:
            features['greeting_markers'] += 1

    # Opinion markers
    for marker in _OPINION_MARKERS:
        if marker in expanded or marker in normalized:
            features['opinion_markers'] += 1

    # Command verbs at start
    if words:
        first_word = re.sub(r'[^a-z]', '', words[0].lower())
        if first_word in _COMMAND_VERBS:
            features['command_verbs_at_start'] = True

    # Offer markers
    for marker in _OFFER_MARKERS:
        if marker in expanded or marker in normalized:
            features['offer_markers'] += 1

    return features


def classify_intent(text: str) -> Dict:
    """
    Classify the intent of Manglish text.

    Args:
        text: Input text to classify.

    Returns:
        Dict with keys:
            - intent: str (one of 8 categories)
            - confidence: float (0.0 to 1.0)
            - sub_type: str (more specific classification)
    """
    if not text or not text.strip():
        return {'intent': 'statement', 'confidence': 0.0, 'sub_type': 'empty'}

    features = get_intent_features(text)
    normalized = _normalize_text(text)
    expanded = _expand_shortforms(normalized)

    # Score each intent
    scores = {
        'question': 0.0,
        'request': 0.0,
        'complaint': 0.0,
        'greeting': 0.0,
        'opinion': 0.0,
        'statement': 0.0,
        'command': 0.0,
        'offer': 0.0,
    }

    sub_types = {
        'question': 'general',
        'request': 'general',
        'complaint': 'general',
        'greeting': 'general',
        'opinion': 'general',
        'statement': 'neutral',
        'command': 'general',
        'offer': 'general',
    }

    # --- Question scoring ---
    if features['has_question_mark']:
        scores['question'] += 0.4
    if features['question_words'] > 0:
        scores['question'] += 0.3 * min(features['question_words'], 2)
    if features['ends_with_question_particle']:
        scores['question'] += 0.25
    if features['question_particles'] > 0:
        scores['question'] += 0.15

    # Sub-type for question
    if any(w in expanded for w in ('berapa', 'how much', 'how many')):
        sub_types['question'] = 'quantity'
    elif any(w in expanded for w in ('bila', 'when', 'what time')):
        sub_types['question'] = 'time'
    elif any(w in expanded for w in ('mana', 'where', 'kat mana')):
        sub_types['question'] = 'location'
    elif any(w in expanded for w in ('siapa', 'who', 'whose')):
        sub_types['question'] = 'person'
    elif any(w in expanded for w in ('kenapa', 'why', 'sebab apa')):
        sub_types['question'] = 'reason'
    elif any(w in expanded for w in ('macam mana', 'how', 'camne', 'cmne')):
        sub_types['question'] = 'method'
    elif any(w in expanded for w in ('apa', 'what', 'which')):
        sub_types['question'] = 'information_seeking'
    else:
        sub_types['question'] = 'information_seeking'

    # --- Request scoring ---
    if features['request_markers'] > 0:
        scores['request'] += 0.35 * min(features['request_markers'], 2)
    # "boleh tak" / "can you" pattern
    if re.search(r'\b(boleh|blh)\s*(tak|x)\b', expanded):
        scores['request'] += 0.3
    if re.search(r'\b(can|could)\s*(you|u)\b', expanded):
        scores['request'] += 0.3

    # Sub-type for request
    if any(w in expanded for w in ('tolong', 'help', 'tlg')):
        sub_types['request'] = 'help'
    elif any(w in expanded for w in ('bagi', 'give', 'pass')):
        sub_types['request'] = 'give'
    elif any(w in expanded for w in ('minta', 'please')):
        sub_types['request'] = 'polite_request'
    else:
        sub_types['request'] = 'action_request'

    # --- Complaint scoring ---
    if features['complaint_markers'] > 0:
        scores['complaint'] += 0.3 * min(features['complaint_markers'], 3)
    if features['exclamation_marks'] > 0 and features['complaint_markers'] > 0:
        scores['complaint'] += 0.15
    # Negative intensifiers
    if re.search(r'\b(gila|sangat|very|so|damn)\b', expanded) and features['complaint_markers'] > 0:
        scores['complaint'] += 0.1

    # Sub-type for complaint
    if any(w in expanded for w in ('lambat', 'slow', 'lama')):
        sub_types['complaint'] = 'slow_service'
    elif any(w in expanded for w in ('bodoh', 'stupid', 'useless')):
        sub_types['complaint'] = 'incompetence'
    elif any(w in expanded for w in ('mahal', 'expensive', 'overpriced')):
        sub_types['complaint'] = 'pricing'
    else:
        sub_types['complaint'] = 'dissatisfaction'

    # --- Greeting scoring ---
    if features['greeting_markers'] > 0:
        scores['greeting'] += 0.5
    # Standalone greetings get higher confidence
    if features['word_count'] <= 3 and features['greeting_markers'] > 0:
        scores['greeting'] += 0.3
    # Check if entire text is basically a greeting
    words = normalized.split()
    if len(words) <= 5:
        for marker in _GREETING_MARKERS:
            if normalized == marker or normalized.startswith(marker):
                scores['greeting'] += 0.2
                break

    # Sub-type for greeting
    if any(w in normalized for w in ('assalamualaikum', 'salam', 'aslkm')):
        sub_types['greeting'] = 'islamic'
    elif any(w in normalized for w in ('morning', 'pagi')):
        sub_types['greeting'] = 'morning'
    elif any(w in normalized for w in ('evening', 'petang', 'night', 'malam')):
        sub_types['greeting'] = 'evening'
    else:
        sub_types['greeting'] = 'casual'

    # --- Opinion scoring ---
    if features['opinion_markers'] > 0:
        scores['opinion'] += 0.35 * min(features['opinion_markers'], 2)
    # Subjective language patterns
    if re.search(r'\b(best|worst|bagus|teruk|ok la|okay la)\b', expanded):
        scores['opinion'] += 0.2
    if re.search(r'\b(aku rasa|i think|i feel|bagi aku|pada aku)\b', expanded):
        scores['opinion'] += 0.25

    # Sub-type for opinion
    if any(w in expanded for w in ('best', 'bagus', 'good', 'nice', 'cantik')):
        sub_types['opinion'] = 'positive_evaluation'
    elif any(w in expanded for w in ('worst', 'teruk', 'bad', 'tak best')):
        sub_types['opinion'] = 'negative_evaluation'
    elif any(w in expanded for w in ('worth', 'berbaloi', 'value')):
        sub_types['opinion'] = 'value_judgment'
    else:
        sub_types['opinion'] = 'personal_view'

    # --- Command scoring ---
    if features['command_verbs_at_start']:
        scores['command'] += 0.4
    if features['is_short'] and features['command_verbs_at_start']:
        scores['command'] += 0.2
    if not features['has_question_mark'] and features['command_verbs_at_start']:
        scores['command'] += 0.15
    if features['exclamation_marks'] > 0 and features['command_verbs_at_start']:
        scores['command'] += 0.1

    # Sub-type for command
    if any(w in expanded for w in ('jangan', 'jgn', 'stop', 'dont', "don't")):
        sub_types['command'] = 'prohibition'
    elif any(w in expanded for w in ('cepat', 'hurry', 'quick')):
        sub_types['command'] = 'urgent'
    else:
        sub_types['command'] = 'directive'

    # --- Offer scoring ---
    if features['offer_markers'] > 0:
        scores['offer'] += 0.4 * min(features['offer_markers'], 2)
    if re.search(r'\b(nak|nk)\s*(aku|i)\b', expanded):
        scores['offer'] += 0.25
    if re.search(r'\bjom\b', expanded):
        scores['offer'] += 0.3

    # Sub-type for offer
    if 'jom' in expanded:
        sub_types['offer'] = 'invitation'
    elif any(w in expanded for w in ('aku tolong', 'let me help', 'i can help')):
        sub_types['offer'] = 'help_offer'
    else:
        sub_types['offer'] = 'voluntary'

    # --- Statement is the default/baseline ---
    scores['statement'] = 0.2  # Base score

    # Boost statement if nothing else scores high
    max_other = max(v for k, v in scores.items() if k != 'statement')
    if max_other < 0.3:
        scores['statement'] += 0.2

    # Determine winner
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # Normalize confidence to 0-1 range
    confidence = min(best_score, 1.0)
    # Ensure minimum confidence for non-statement
    if best_intent != 'statement' and confidence < 0.3:
        best_intent = 'statement'
        confidence = 0.4
        sub_types['statement'] = 'neutral'

    return {
        'intent': best_intent,
        'confidence': round(confidence, 2),
        'sub_type': sub_types[best_intent],
    }


def classify_intents_batch(texts: List[str]) -> List[Dict]:
    """
    Classify intents for multiple texts.

    Args:
        texts: List of input texts.

    Returns:
        List of classification results.
    """
    return [classify_intent(text) for text in texts]


def is_question(text: str) -> bool:
    """Check if text is a question."""
    result = classify_intent(text)
    return result['intent'] == 'question'


def is_request(text: str) -> bool:
    """Check if text is a request."""
    result = classify_intent(text)
    return result['intent'] == 'request'


def is_complaint(text: str) -> bool:
    """Check if text is a complaint."""
    result = classify_intent(text)
    return result['intent'] == 'complaint'
