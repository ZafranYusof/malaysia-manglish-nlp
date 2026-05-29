"""Part-of-Speech tagging for Manglish text."""

import re
from manglish_nlp.utils import get_shortforms, get_particles


# POS tag definitions
TAGS = {
    'NN': 'Noun',
    'VB': 'Verb',
    'JJ': 'Adjective',
    'RB': 'Adverb',
    'PRP': 'Pronoun',
    'DT': 'Determiner',
    'IN': 'Preposition',
    'CC': 'Conjunction',
    'MD': 'Modal',
    'NEG': 'Negation',
    'PTL': 'Particle',
    'INT': 'Intensifier',
    'NUM': 'Number',
    'PUNCT': 'Punctuation',
    'UNK': 'Unknown',
}

# Word lists for rule-based tagging
_PRONOUNS = {
    'saya', 'aku', 'ak', 'aq', 'sy', 'awak', 'kau', 'ko', 'kamu', 'anda',
    'dia', 'beliau', 'mereka', 'dorang', 'diorang', 'kami', 'kita',
    'kitorg', 'kitorang', 'korang', 'hang', 'mu',
}

_DETERMINERS = {
    'ini', 'itu', 'ni', 'tu', 'the', 'a', 'an', 'si', 'sang',
    'semua', 'setiap', 'para', 'beberapa', 'some', 'any', 'all',
}

_PREPOSITIONS = {
    'di', 'ke', 'dari', 'pada', 'untuk', 'utk', 'dengan', 'dgn',
    'dalam', 'dlm', 'luar', 'atas', 'bawah', 'antara', 'oleh',
    'tentang', 'tanpa', 'sejak', 'hingga', 'sampai',
    'in', 'on', 'at', 'to', 'from', 'with', 'by', 'for', 'about',
}

_CONJUNCTIONS = {
    'dan', 'atau', 'tetapi', 'tapi', 'tp', 'namun', 'serta',
    'kerana', 'sebab', 'sbb', 'kalau', 'jika', 'walaupun',
    'and', 'or', 'but', 'because', 'if', 'although', 'while',
}

_MODALS = {
    'boleh', 'blh', 'dapat', 'dpt', 'harus', 'perlu', 'mesti',
    'akan', 'mahu', 'nak', 'nk', 'hendak', 'ingin',
    'can', 'could', 'will', 'would', 'should', 'must', 'may', 'might',
}

_NEGATIONS = {
    'tidak', 'tak', 'x', 'bukan', 'bkn', 'belum', 'blm',
    'jangan', 'jgn', 'tiada', 'xde', 'takde',
    'not', 'never', 'no',
}

_VERBS = {
    'pergi', 'pegi', 'gi', 'datang', 'balik', 'blk', 'makan', 'mkn',
    'minum', 'mnm', 'tidur', 'tdo', 'tdr', 'kerja', 'kje', 'buat',
    'ambil', 'amik', 'beri', 'bagi', 'bg', 'cari', 'guna', 'pakai',
    'tulis', 'baca', 'dengar', 'tengok', 'tgk', 'nampak', 'rasa',
    'fikir', 'tanya', 'jawab', 'cakap', 'ckp', 'kata', 'hantar',
    'belajar', 'blaja', 'ajar', 'tolong', 'tlg', 'minta', 'suruh',
    'go', 'come', 'eat', 'drink', 'sleep', 'work', 'make', 'take',
    'give', 'get', 'buy', 'sell', 'say', 'tell', 'think', 'know',
    'see', 'want', 'need', 'like', 'love', 'hate', 'try', 'help',
}

_ADJECTIVES = {
    'baik', 'buruk', 'cantik', 'cun', 'lawa', 'hodoh', 'besar',
    'kecil', 'tinggi', 'rendah', 'panjang', 'pendek', 'baru', 'lama',
    'muda', 'tua', 'pandai', 'bodoh', 'rajin', 'malas', 'penat',
    'best', 'power', 'padu', 'mantap', 'solid', 'hampeh', 'teruk',
    'good', 'bad', 'big', 'small', 'new', 'old', 'nice', 'cool',
    'sedap', 'mahal', 'murah', 'senang', 'susah', 'cepat', 'lambat',
}

_NOUNS = {
    'orang', 'org', 'budak', 'bdk', 'kawan', 'kwn', 'rumah', 'rmh',
    'sekolah', 'sklh', 'kedai', 'kereta', 'keta', 'telefon', 'fon',
    'duit', 'masa', 'hari', 'tempat', 'tmpt', 'kerja', 'makanan',
    'air', 'nasi', 'roti', 'ayam', 'ikan', 'buku', 'bilik',
    'person', 'house', 'school', 'car', 'phone', 'money', 'time',
    'food', 'water', 'book', 'room', 'place', 'thing', 'way',
}


def pos_tag(text):
    """Tag each word with its Part-of-Speech.
    
    Uses rule-based approach with Manglish-aware word lists.
    Tags: NN (noun), VB (verb), JJ (adjective), RB (adverb),
    PRP (pronoun), DT (determiner), IN (preposition), CC (conjunction),
    MD (modal), NEG (negation), PTL (particle), INT (intensifier),
    NUM (number), PUNCT (punctuation), UNK (unknown).
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list[tuple]: List of (word, tag) tuples.
    
    Example:
        >>> manglish_nlp.pos_tag("aku nak pergi kedai")
        [('aku', 'PRP'), ('nak', 'MD'), ('pergi', 'VB'), ('kedai', 'NN')]
        >>> manglish_nlp.pos_tag("gila best la weh")
        [('gila', 'INT'), ('best', 'JJ'), ('la', 'PTL'), ('weh', 'PTL')]
    """
    particles = set(get_particles().keys())
    intensifiers = {'gila', 'giler', 'gile', 'sangat', 'sgt', 'memang', 'mmg',
                    'betul2', 'btl2', 'habis', 'teramat', 'super', 'ultra',
                    'very', 'really', 'so', 'damn', 'totally'}
    
    tokens = re.findall(r"[\w']+|[^\w\s]", text)
    result = []
    
    for token in tokens:
        lower = token.lower()
        
        # Check categories in order of specificity
        if re.match(r'^[^\w\s]$', token):
            result.append((token, 'PUNCT'))
        elif re.match(r'^\d+$', token):
            result.append((token, 'NUM'))
        elif lower in _NEGATIONS:
            result.append((token, 'NEG'))
        elif lower in particles:
            result.append((token, 'PTL'))
        elif lower in intensifiers:
            result.append((token, 'INT'))
        elif lower in _PRONOUNS:
            result.append((token, 'PRP'))
        elif lower in _MODALS:
            result.append((token, 'MD'))
        elif lower in _DETERMINERS:
            result.append((token, 'DT'))
        elif lower in _PREPOSITIONS:
            result.append((token, 'IN'))
        elif lower in _CONJUNCTIONS:
            result.append((token, 'CC'))
        elif lower in _VERBS:
            result.append((token, 'VB'))
        elif lower in _ADJECTIVES:
            result.append((token, 'JJ'))
        elif lower in _NOUNS:
            result.append((token, 'NN'))
        # Heuristics for unknown words
        elif lower.startswith(('ber', 'me', 'di', 'ter')):
            result.append((token, 'VB'))  # Likely verb with prefix
        elif lower.endswith(('an', 'nya')):
            result.append((token, 'NN'))  # Likely noun with suffix
        elif lower.endswith(('ly', 'nya')):
            result.append((token, 'RB'))  # Likely adverb
        else:
            result.append((token, 'UNK'))
    
    return result


def pos_tag_detailed(text):
    """POS tag with additional metadata.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list[dict]: List of dicts with 'word', 'tag', 'tag_name', 'confidence'.
    
    Example:
        >>> pos_tag_detailed("aku nak makan")
        [{'word': 'aku', 'tag': 'PRP', 'tag_name': 'Pronoun', 'confidence': 'high'},
         {'word': 'nak', 'tag': 'MD', 'tag_name': 'Modal', 'confidence': 'high'},
         {'word': 'makan', 'tag': 'VB', 'tag_name': 'Verb', 'confidence': 'high'}]
    """
    tagged = pos_tag(text)
    result = []
    
    for word, tag in tagged:
        lower = word.lower()
        # Determine confidence
        if tag == 'UNK':
            confidence = 'low'
        elif tag in ('PUNCT', 'NUM'):
            confidence = 'high'
        elif lower in _PRONOUNS or lower in _NEGATIONS:
            confidence = 'high'
        elif tag == 'VB' and lower.startswith(('ber', 'me', 'di', 'ter')):
            confidence = 'medium'  # Heuristic-based
        else:
            confidence = 'high'
        
        result.append({
            'word': word,
            'tag': tag,
            'tag_name': TAGS.get(tag, 'Unknown'),
            'confidence': confidence,
        })
    
    return result
