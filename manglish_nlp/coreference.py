"""Coreference resolution for Malaysian Manglish text.

Resolve pronouns and references to their antecedents in Manglish text.
Handles Malay pronouns (dia, mereka, -nya), English pronouns, and
informal Manglish variants (die, dorg, tu, ni).
"""

import re
from manglish_nlp.ner import ner_tag

# Pronoun categories
_SINGULAR_MALAY = ['dia', 'beliau', 'baginda', 'die']
_PLURAL_MALAY = ['mereka', 'diorang', 'dorg', 'dorang']
_DEMONSTRATIVE_MALAY = ['itu', 'ini', 'tu', 'ni']
_POSSESSIVE_SUFFIX = ['-nya', 'nya']

_SINGULAR_MALE_EN = ['he', 'him', 'his']
_SINGULAR_FEMALE_EN = ['she', 'her', 'hers']
_PLURAL_EN = ['they', 'them', 'their', 'theirs']
_SINGULAR_NEUTRAL_EN = ['it', 'its']
_DEMONSTRATIVE_EN = ['this', 'that', 'these', 'those']

_ALL_PRONOUNS = (
    _SINGULAR_MALAY + _PLURAL_MALAY + _DEMONSTRATIVE_MALAY +
    _SINGULAR_MALE_EN + _SINGULAR_FEMALE_EN + _PLURAL_EN +
    _SINGULAR_NEUTRAL_EN + _DEMONSTRATIVE_EN
)

# Gender heuristics for Malaysian names
_MALE_PREFIXES = ['ahmad', 'muhammad', 'mohd', 'md', 'abu', 'abdul', 'wan']
_MALE_NAMES = [
    'ali', 'ahmad', 'hassan', 'hussein', 'ibrahim', 'ismail', 'omar',
    'othman', 'rahman', 'rahim', 'razak', 'aziz', 'azman', 'hafiz',
    'hakim', 'faiz', 'faisal', 'amir', 'amin', 'arif', 'adam',
    'daniel', 'david', 'james', 'john', 'michael', 'robert', 'william',
    'zafran', 'zul', 'zulkifli', 'rizal', 'rizwan', 'syafiq', 'izzat',
]
_FEMALE_PREFIXES = ['siti', 'nur', 'nurul', 'wan']
_FEMALE_NAMES = [
    'siti', 'nur', 'nurul', 'aisyah', 'fatimah', 'aminah', 'zainab',
    'khadijah', 'maryam', 'sarah', 'hannah', 'aina', 'aini', 'alya',
    'amira', 'anisa', 'diana', 'farah', 'hana', 'iman', 'jasmine',
    'lisa', 'maria', 'nadia', 'nina', 'puteri', 'rina', 'sofia',
    'syahira', 'yasmin', 'zahra', 'liyana', 'aqilah', 'balqis',
]

# Chinese/Indian common names (gender-ambiguous without more context)
_MALE_NAMES_OTHER = [
    'wei', 'jun', 'ming', 'hong', 'chong', 'tan', 'lee', 'lim',
    'kumar', 'raj', 'ravi', 'muthu', 'ganesh', 'suresh', 'vikram',
]
_FEMALE_NAMES_OTHER = [
    'mei', 'ling', 'xin', 'hui', 'yan', 'fang',
    'priya', 'lakshmi', 'devi', 'anitha', 'kavitha', 'shalini',
]


def _infer_gender(name):
    """Infer gender from a Malaysian name. Returns 'male', 'female', or 'unknown'."""
    if not name:
        return 'unknown'
    name_lower = name.lower().strip()
    first_word = name_lower.split()[0] if name_lower.split() else name_lower

    # Check prefixes
    for prefix in _MALE_PREFIXES:
        if name_lower.startswith(prefix):
            return 'male'
    for prefix in _FEMALE_PREFIXES:
        if name_lower.startswith(prefix):
            return 'female'

    # Check name lists
    if first_word in _MALE_NAMES or first_word in _MALE_NAMES_OTHER:
        return 'male'
    if first_word in _FEMALE_NAMES or first_word in _FEMALE_NAMES_OTHER:
        return 'female'

    return 'unknown'


def _infer_number(name):
    """Infer if entity is singular or plural."""
    # Most named entities are singular
    return 'singular'


def _is_animate(entity_type):
    """Check if entity type is animate (can be referred to by personal pronouns)."""
    return entity_type in ('PERSON',)


def _extract_entities_from_text(text):
    """Extract named entities with positions using NER module."""
    entities = ner_tag(text)
    # Filter to PERSON and ORGANIZATION primarily
    result = []
    for ent in entities:
        result.append({
            'text': ent['text'],
            'type': ent['type'],
            'start': ent['start'],
            'end': ent['end'],
            'gender': _infer_gender(ent['text']) if ent['type'] == 'PERSON' else 'unknown',
            'number': 'singular',
        })
    return result


def _find_pronouns(text):
    """Find all pronouns in text with positions."""
    pronouns = []
    text_lower = text.lower()

    # Handle -nya suffix specially
    for m in re.finditer(r'(\w+)(nya)\b', text_lower):
        pronouns.append({
            'pronoun': '-nya',
            'position': (m.start(2), m.end(2)),
            'type': 'possessive',
            'number': 'singular',
            'gender': 'unknown',
        })

    # Find standalone pronouns
    for pronoun in _ALL_PRONOUNS:
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        for m in re.finditer(pattern, text_lower):
            # Determine properties
            if pronoun in _SINGULAR_MALAY:
                ptype = 'personal'
                number = 'singular'
                gender = 'unknown'
            elif pronoun in _PLURAL_MALAY:
                ptype = 'personal'
                number = 'plural'
                gender = 'unknown'
            elif pronoun in _DEMONSTRATIVE_MALAY:
                ptype = 'demonstrative'
                number = 'singular'
                gender = 'unknown'
            elif pronoun in _SINGULAR_MALE_EN:
                ptype = 'personal'
                number = 'singular'
                gender = 'male'
            elif pronoun in _SINGULAR_FEMALE_EN:
                ptype = 'personal'
                number = 'singular'
                gender = 'female'
            elif pronoun in _PLURAL_EN:
                ptype = 'personal'
                number = 'plural'
                gender = 'unknown'
            elif pronoun in _SINGULAR_NEUTRAL_EN:
                ptype = 'personal'
                number = 'singular'
                gender = 'neutral'
            elif pronoun in _DEMONSTRATIVE_EN:
                ptype = 'demonstrative'
                number = 'singular' if pronoun in ('this', 'that') else 'plural'
                gender = 'unknown'
            else:
                ptype = 'unknown'
                number = 'singular'
                gender = 'unknown'

            pronouns.append({
                'pronoun': pronoun,
                'position': (m.start(), m.end()),
                'type': ptype,
                'number': number,
                'gender': gender,
            })

    # Sort by position
    pronouns.sort(key=lambda x: x['position'][0])
    return pronouns


def _is_compatible(pronoun_info, entity_info):
    """Check if a pronoun is compatible with an entity for coreference."""
    # Gender compatibility
    p_gender = pronoun_info['gender']
    e_gender = entity_info['gender']
    if p_gender != 'unknown' and e_gender != 'unknown' and p_gender != e_gender:
        return False

    # Number compatibility
    p_number = pronoun_info['number']
    e_number = entity_info['number']
    if p_number != e_number:
        return False

    # Animacy: personal pronouns need animate entities
    if pronoun_info['type'] == 'personal' and pronoun_info['gender'] != 'neutral':
        if entity_info['type'] not in ('PERSON', 'ORGANIZATION'):
            return False

    return True


def _resolve_pronoun(pronoun_info, entities, pronoun_position):
    """Resolve a single pronoun to its most likely antecedent."""
    candidates = []

    for entity in entities:
        # Entity must appear before the pronoun (or in prior context)
        if entity['end'] >= pronoun_position:
            continue

        if _is_compatible(pronoun_info, entity):
            # Score by recency (closer = better)
            distance = pronoun_position - entity['end']
            candidates.append((entity, distance))

    if not candidates:
        return None

    # Sort by distance (prefer most recent)
    candidates.sort(key=lambda x: x[1])
    best = candidates[0]

    # Confidence based on distance and number of candidates
    distance = best[1]
    if distance < 50:
        confidence = 0.9
    elif distance < 100:
        confidence = 0.75
    elif distance < 200:
        confidence = 0.6
    else:
        confidence = 0.4

    # Reduce confidence if multiple candidates
    if len(candidates) > 1:
        confidence *= 0.9

    return {
        'antecedent': best[0]['text'],
        'antecedent_position': (best[0]['start'], best[0]['end']),
        'confidence': round(confidence, 2),
    }


def resolve_coreferences(text):
    """Resolve pronouns and references to their antecedents.

    Parameters:
        text (str): Input text to analyze.

    Returns:
        list[dict]: List of resolved coreferences, each with:
            - pronoun (str): The pronoun found
            - position (tuple): (start, end) of pronoun in text
            - antecedent (str): The resolved entity name
            - antecedent_position (tuple): (start, end) of antecedent
            - confidence (float): Resolution confidence (0.0-1.0)

    Example:
        >>> resolve_coreferences("Ali pergi kedai. Dia beli roti.")
        [{"pronoun": "dia", "position": (16, 19), "antecedent": "Ali",
          "antecedent_position": (0, 3), "confidence": 0.9}]
    """
    if not text or not text.strip():
        return []

    entities = _extract_entities_from_text(text)
    pronouns = _find_pronouns(text)

    results = []
    for pron in pronouns:
        resolution = _resolve_pronoun(pron, entities, pron['position'][0])
        if resolution:
            results.append({
                'pronoun': pron['pronoun'],
                'position': pron['position'],
                'antecedent': resolution['antecedent'],
                'antecedent_position': resolution['antecedent_position'],
                'confidence': resolution['confidence'],
            })

    return results


def resolve_in_context(text, context=None):
    """Resolve coreferences with prior context for multi-turn conversations.

    Parameters:
        text (str): Current text to analyze.
        context (str, optional): Prior context (previous turns/sentences).

    Returns:
        list[dict]: Same format as resolve_coreferences().

    Example:
        >>> resolve_in_context("Dia kata ok", context="Aku jumpa Ahmad semalam.")
        [{"pronoun": "dia", "position": (0, 3), "antecedent": "Ahmad", ...}]
    """
    if not text or not text.strip():
        return []

    if context:
        # Combine context + text, but track offset
        combined = context.rstrip() + " " + text.lstrip()
        offset = len(context.rstrip()) + 1

        entities = _extract_entities_from_text(combined)
        pronouns = _find_pronouns(text)

        results = []
        for pron in pronouns:
            # Adjust pronoun position to combined text
            adjusted_pos = pron['position'][0] + offset
            resolution = _resolve_pronoun(pron, entities, adjusted_pos)
            if resolution:
                # Adjust antecedent position back if it's in context
                ant_start = resolution['antecedent_position'][0]
                ant_end = resolution['antecedent_position'][1]
                if ant_start < offset:
                    # Antecedent is in context, keep original context positions
                    ant_pos = (ant_start, ant_end)
                else:
                    # Antecedent is in current text
                    ant_pos = (ant_start - offset, ant_end - offset)

                results.append({
                    'pronoun': pron['pronoun'],
                    'position': pron['position'],
                    'antecedent': resolution['antecedent'],
                    'antecedent_position': ant_pos,
                    'confidence': resolution['confidence'],
                })

        return results
    else:
        return resolve_coreferences(text)


def get_entities_and_references(text):
    """Map entities to all their references (pronouns) in text.

    Parameters:
        text (str): Input text.

    Returns:
        dict: Mapping of entity names to list of references.
            {entity_name: [{"pronoun": str, "position": (int, int)}]}

    Example:
        >>> get_entities_and_references("Siti beli buku. Dia baca buku tu.")
        {"Siti": [{"pronoun": "dia", "position": (16, 19)}]}
    """
    if not text or not text.strip():
        return {}

    coreferences = resolve_coreferences(text)
    entity_map = {}

    for coref in coreferences:
        entity = coref['antecedent']
        if entity not in entity_map:
            entity_map[entity] = []
        entity_map[entity].append({
            'pronoun': coref['pronoun'],
            'position': coref['position'],
        })

    return entity_map


def replace_pronouns(text):
    """Replace pronouns with their resolved antecedents.

    Parameters:
        text (str): Input text with pronouns.

    Returns:
        str: Text with pronouns replaced by antecedent names.

    Example:
        >>> replace_pronouns("Ali pergi kedai. Dia beli roti.")
        "Ali pergi kedai. Ali beli roti."
    """
    if not text or not text.strip():
        return text

    coreferences = resolve_coreferences(text)

    if not coreferences:
        return text

    # Sort by position descending so replacements don't shift indices
    coreferences.sort(key=lambda x: x['position'][0], reverse=True)

    result = text
    for coref in coreferences:
        start, end = coref['position']
        pronoun = coref['pronoun']
        antecedent = coref['antecedent']

        # Handle -nya suffix: replace "wordnya" with "word antecedent"
        if pronoun == '-nya':
            # Replace just the "nya" part with possessive construction
            result = result[:start] + f" {antecedent}" + result[end:]
        else:
            # Replace pronoun with antecedent, preserving case
            if result[start:end][0].isupper():
                replacement = antecedent[0].upper() + antecedent[1:]
            else:
                replacement = antecedent
            result = result[:start] + replacement + result[end:]

    return result
