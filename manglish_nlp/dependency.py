"""Basic dependency parsing for Malaysian text.

Extracts subject-verb-object (SVO) relations and basic syntactic structure.
Rule-based approach using POS tags and word order patterns.
"""

import re
from manglish_nlp.pos import pos_tag
from manglish_nlp.tokenizer import word_tokenize


def parse_dependencies(text):
    """Extract basic dependency relations from text.
    
    Identifies:
    - Subject (who/what does the action)
    - Verb (the action)
    - Object (what receives the action)
    - Modifiers (adjectives, adverbs)
    - Negation
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Parsed structure with 'subject', 'verb', 'object', 'modifiers', 'relations'.
    
    Example:
        >>> parse_dependencies("aku nak makan nasi goreng")
        {'subject': 'aku', 'verb': 'makan', 'object': 'nasi goreng',
         'modifiers': {'modal': 'nak'}, ...}
        >>> parse_dependencies("dia tak pergi sekolah semalam")
        {'subject': 'dia', 'verb': 'pergi', 'object': 'sekolah',
         'modifiers': {'negation': 'tak', 'time': 'semalam'}, ...}
    """
    tagged = pos_tag(text)
    
    subject = None
    verb = None
    obj = None
    modal = None
    negation = None
    preposition = None
    prep_obj = None
    adjectives = []
    adverbs = []
    particles = []
    relations = []
    
    # State machine for SVO extraction
    state = 'seeking_subject'
    obj_words = []
    
    for i, (word, tag) in enumerate(tagged):
        if state == 'seeking_subject':
            if tag == 'PRP':
                subject = word
                state = 'seeking_verb'
                relations.append({'type': 'nsubj', 'head': word, 'dep': None})
            elif tag == 'NN':
                subject = word
                state = 'seeking_verb'
                relations.append({'type': 'nsubj', 'head': word, 'dep': None})
            elif tag == 'NEG':
                negation = word
            elif tag == 'MD':
                modal = word
                state = 'seeking_verb'
        
        elif state == 'seeking_verb':
            if tag == 'NEG':
                negation = word
            elif tag == 'MD':
                modal = word
            elif tag == 'VB':
                verb = word
                state = 'seeking_object'
                # Update nsubj relation
                for r in relations:
                    if r['type'] == 'nsubj':
                        r['dep'] = word
            elif tag == 'INT':
                adverbs.append(word)
        
        elif state == 'seeking_object':
            if tag in ('NN', 'UNK'):
                obj_words.append(word)
            elif tag == 'JJ':
                if obj_words:
                    obj_words.append(word)  # Part of noun phrase
                else:
                    adjectives.append(word)
            elif tag == 'IN':
                preposition = word
                state = 'seeking_prep_obj'
            elif tag == 'VB' and not obj_words:
                # Serial verb
                obj_words.append(word)
            elif tag == 'PTL':
                particles.append(word)
            elif tag == 'NUM':
                obj_words.append(word)
        
        elif state == 'seeking_prep_obj':
            if tag in ('NN', 'UNK', 'PRP'):
                prep_obj = word
                state = 'done'
    
    # Combine object words
    if obj_words:
        obj = ' '.join(obj_words)
    
    # Build modifiers
    modifiers = {}
    if modal:
        modifiers['modal'] = modal
    if negation:
        modifiers['negation'] = negation
    if adjectives:
        modifiers['adjectives'] = adjectives
    if adverbs:
        modifiers['adverbs'] = adverbs
    if particles:
        modifiers['particles'] = particles
    if preposition and prep_obj:
        modifiers['prepositional'] = f"{preposition} {prep_obj}"
    
    # Add object relation
    if verb and obj:
        relations.append({'type': 'dobj', 'head': verb, 'dep': obj})
    if preposition and prep_obj:
        relations.append({'type': 'prep', 'head': preposition, 'dep': prep_obj})
    if negation and verb:
        relations.append({'type': 'neg', 'head': verb, 'dep': negation})
    if modal and verb:
        relations.append({'type': 'aux', 'head': verb, 'dep': modal})
    
    return {
        'subject': subject,
        'verb': verb,
        'object': obj,
        'modifiers': modifiers,
        'relations': relations,
        'is_negated': negation is not None,
        'tree': _build_tree(subject, verb, obj, modifiers),
    }


def _build_tree(subject, verb, obj, modifiers):
    """Build a simple dependency tree string."""
    parts = []
    if subject:
        parts.append(f"[S: {subject}]")
    if modifiers.get('negation'):
        parts.append(f"[NEG: {modifiers['negation']}]")
    if modifiers.get('modal'):
        parts.append(f"[MOD: {modifiers['modal']}]")
    if verb:
        parts.append(f"[V: {verb}]")
    if obj:
        parts.append(f"[O: {obj}]")
    if modifiers.get('prepositional'):
        parts.append(f"[PP: {modifiers['prepositional']}]")
    return ' -> '.join(parts) if parts else '(empty)'


def extract_svo(text):
    """Extract Subject-Verb-Object triples from text.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list[dict]: List of SVO triples found.
    
    Example:
        >>> extract_svo("aku makan nasi, dia minum air")
        [{'subject': 'aku', 'verb': 'makan', 'object': 'nasi'},
         {'subject': 'dia', 'verb': 'minum', 'object': 'air'}]
    """
    # Split on common clause separators
    clauses = re.split(r'[,;]|\bdan\b|\btapi\b|\blepas tu\b', text)
    
    triples = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        result = parse_dependencies(clause)
        if result['verb']:
            triples.append({
                'subject': result['subject'],
                'verb': result['verb'],
                'object': result['object'],
                'negated': result['is_negated'],
            })
    
    return triples


def get_verb_frame(text):
    """Extract verb and its arguments.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Verb frame with agent, action, patient, instrument, location.
    
    Example:
        >>> get_verb_frame("aku potong roti dengan pisau kat dapur")
        {'agent': 'aku', 'action': 'potong', 'patient': 'roti',
         'instrument': 'pisau', 'location': 'dapur'}
    """
    tagged = pos_tag(text)
    
    frame = {
        'agent': None,
        'action': None,
        'patient': None,
        'instrument': None,
        'location': None,
        'time': None,
    }
    
    # Time words
    time_words = {'semalam', 'esok', 'tadi', 'nanti', 'sekarang', 'pagi', 'petang', 'malam'}
    # Location prepositions
    loc_preps = {'kat', 'di', 'dekat'}
    # Instrument prepositions
    inst_preps = {'dengan', 'guna', 'pakai'}
    
    current_prep = None
    
    for i, (word, tag) in enumerate(tagged):
        lower = word.lower()
        
        if tag == 'PRP' and frame['agent'] is None:
            frame['agent'] = word
        elif tag == 'VB' and frame['action'] is None:
            frame['action'] = word
        elif tag == 'NN' and frame['action'] is not None:
            if current_prep in loc_preps:
                frame['location'] = word
                current_prep = None
            elif current_prep in inst_preps:
                frame['instrument'] = word
                current_prep = None
            elif frame['patient'] is None:
                frame['patient'] = word
        elif tag == 'IN':
            current_prep = lower
        elif lower in time_words:
            frame['time'] = word
    
    return frame
