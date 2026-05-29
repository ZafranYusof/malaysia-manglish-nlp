"""Pipeline mode — chain multiple NLP modules in a single call.

Reduces boilerplate and enables efficient multi-step processing.
"""

import manglish_nlp
from manglish_nlp.emotion import detect_emotion
from manglish_nlp.profanity import detect_profanity
from manglish_nlp.dialect import detect_dialect
from manglish_nlp.sarcasm import detect_sarcasm
from manglish_nlp.contextual_spelling import correct_contextual
from manglish_nlp.dependency import parse_dependencies


# Available pipeline steps
_STEPS = {
    'normalize', 'sentiment', 'language', 'emotion', 'profanity',
    'dialect', 'sarcasm', 'tokenize', 'stem', 'pos', 'ner',
    'segment', 'formalize', 'clean', 'keywords', 'correct',
    'dependency', 'all',
}

# Default pipeline
_DEFAULT_STEPS = ['normalize', 'sentiment', 'language', 'emotion']


def pipeline(text, steps=None, normalize_first=True):
    """Run multiple NLP modules on text in one call.
    
    Parameters:
        text (str): Input text.
        steps (list[str]): Pipeline steps to run. Default: normalize, sentiment, language, emotion.
            Available: normalize, sentiment, language, emotion, profanity, dialect,
            sarcasm, tokenize, stem, pos, ner, segment, formalize, clean,
            keywords, correct, dependency, all.
        normalize_first (bool): Normalize text before other steps (default True).
    
    Returns:
        dict: Combined results from all steps.
    
    Example:
        >>> pipeline("gila best mkn dia")
        {'original': 'gila best mkn dia', 'normalized': 'gila best makan dia',
         'sentiment': {'sentiment': 'positive', ...}, 'language': {...}, 'emotion': {...}}
        
        >>> pipeline("ambo nok make nasi", steps=['dialect', 'normalize', 'sentiment'])
        {'original': '...', 'dialect': {...}, 'normalized': '...', 'sentiment': {...}}
        
        >>> pipeline("kau ni bodoh ke", steps=['all'])
        {... all modules ...}
    """
    if steps is None:
        steps = _DEFAULT_STEPS
    
    if 'all' in steps:
        steps = ['normalize', 'sentiment', 'language', 'emotion', 'profanity',
                 'dialect', 'sarcasm', 'tokenize', 'stem', 'pos', 'ner',
                 'segment', 'keywords', 'correct', 'dependency']
    
    result = {'original': text}
    
    # Normalize first if requested (other steps use normalized text)
    working_text = text
    if normalize_first and 'normalize' in steps:
        normalized = manglish_nlp.normalize(text)
        result['normalized'] = normalized
        working_text = normalized
    elif normalize_first:
        working_text = manglish_nlp.normalize(text)
    
    for step in steps:
        if step == 'normalize' and 'normalized' in result:
            continue  # Already done
        
        if step == 'normalize':
            result['normalized'] = manglish_nlp.normalize(text)
        elif step == 'sentiment':
            result['sentiment'] = manglish_nlp.sentiment(working_text)
        elif step == 'language':
            result['language'] = manglish_nlp.detect_language(working_text)
        elif step == 'emotion':
            result['emotion'] = detect_emotion(working_text)
        elif step == 'profanity':
            result['profanity'] = detect_profanity(working_text)
        elif step == 'dialect':
            result['dialect'] = detect_dialect(text)  # Use original for dialect
        elif step == 'sarcasm':
            result['sarcasm'] = detect_sarcasm(working_text)
        elif step == 'tokenize':
            result['tokens'] = manglish_nlp.tokenize(working_text)
        elif step == 'stem':
            result['stemmed'] = manglish_nlp.stem(working_text)
        elif step == 'pos':
            result['pos_tags'] = manglish_nlp.pos_tag(working_text)
        elif step == 'ner':
            result['entities'] = manglish_nlp.ner_tag(working_text)
        elif step == 'segment':
            result['segments'] = manglish_nlp.segment(working_text)
        elif step == 'formalize':
            result['formalized'] = manglish_nlp.formalize(working_text)
        elif step == 'clean':
            result['cleaned'] = manglish_nlp.clean(working_text)
        elif step == 'keywords':
            result['keywords'] = manglish_nlp.extract_keywords(working_text)
        elif step == 'correct':
            result['corrected'] = correct_contextual(text)
        elif step == 'dependency':
            result['dependency'] = parse_dependencies(working_text)
    
    return result


def batch_pipeline(texts, steps=None, normalize_first=True):
    """Run pipeline on multiple texts.
    
    Parameters:
        texts (list[str]): Input texts.
        steps (list[str]): Pipeline steps.
        normalize_first (bool): Normalize first.
    
    Returns:
        list[dict]: Results per text.
    
    Example:
        >>> batch_pipeline(["best gila", "teruk la"], steps=['sentiment'])
        [{'original': 'best gila', 'sentiment': {...}}, ...]
    """
    return [pipeline(t, steps, normalize_first) for t in texts]


def analyze(text):
    """Quick full analysis — shorthand for pipeline(text, steps=['all']).
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Full analysis results.
    
    Example:
        >>> analyze("aku nak pergi makan nasi goreng")
        {'original': '...', 'normalized': '...', 'sentiment': {...}, ...}
    """
    return pipeline(text, steps=['all'])
