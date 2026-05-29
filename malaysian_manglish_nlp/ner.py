"""Named Entity Recognition for Malaysian text."""

from __future__ import annotations

from typing import Dict, List

import re
from malaysian_manglish_nlp.utils import get_shortforms

# Entity patterns
_PERSON_TITLES = {'encik', 'puan', 'cik', 'dato', 'datuk', 'datin', 'tan sri', 'tun', 'dr', 'prof', 'ustaz', 'ustazah', 'haji', 'hajjah'}
_STATES = {'johor', 'kedah', 'kelantan', 'melaka', 'negeri sembilan', 'pahang', 'perak', 'perlis', 'pulau pinang', 'sabah', 'sarawak', 'selangor', 'terengganu', 'kuala lumpur', 'putrajaya', 'labuan', 'penang', 'ns', 'n9'}
_CITIES = {'kl', 'jb', 'penang', 'ipoh', 'kuantan', 'kota bharu', 'shah alam', 'petaling jaya', 'pj', 'subang', 'cyberjaya', 'putrajaya', 'melaka', 'alor setar', 'kuching', 'kk', 'kota kinabalu', 'miri', 'sibu', 'sandakan', 'tawau', 'bintulu', 'klang', 'ampang', 'cheras', 'bangsar', 'damansara', 'setapak', 'wangsa maju', 'kepong', 'rawang', 'kajang', 'bangi', 'serdang', 'puchong', 'usj', 'ara damansara', 'mont kiara', 'bukit bintang', 'sentul', 'segambut', 'gombak', 'seremban', 'nilai', 'port dickson', 'taiping', 'teluk intan', 'lumut', 'manjung', 'batu pahat', 'muar', 'kluang', 'segamat', 'pontian', 'kulai', 'senai', 'pasir gudang', 'iskandar', 'nusajaya', 'temerloh', 'bentong', 'raub', 'jerantut', 'kuala terengganu', 'dungun', 'kemaman', 'kangar', 'langkawi', 'sungai petani', 'kulim', 'baling'}
_UNIVERSITIES = {'um', 'ukm', 'usm', 'upm', 'utm', 'uitm', 'ump', 'umpsa', 'unimap', 'unimas', 'ums', 'upsi', 'umt', 'unisza', 'utem', 'uthm', 'iium', 'uiam', 'mmu', 'uniten', 'utp', 'taylor', 'sunway', 'monash', 'nottingham', 'apu', 'inti', 'help', 'ucsi', 'msu', 'segi', 'kdu', 'limkokwing', 'politeknik', 'kolej'}
_ORGS = {'petronas', 'maybank', 'cimb', 'tnb', 'telekom', 'maxis', 'celcom', 'digi', 'grab', 'shopee', 'lazada', 'tng', 'epf', 'kwsp', 'socso', 'perkeso', 'lhdn', 'jpj', 'jkr', 'kkm', 'kpm', 'kdn', 'pdrm', 'bomba', 'jakim', 'dbkl', 'mbpj', 'mpsj', 'proton', 'perodua', 'airasia', 'mas', 'malaysia airlines', 'pos malaysia', 'astro', 'tm', 'unifi', 'hotlink', 'yes4g', 'boost', 'tng ewallet', 'touch n go', 'foodpanda', 'dahmakan', 'mydin', 'aeon', 'tesco', 'lotus', 'giant', 'speedmart', 'kfc', 'mcd', 'mcdonalds', 'pizza hut', 'dominos', 'nandos', 'secret recipe', 'oldtown', 'mamak', 'pos laju', 'jnt', 'shopee express', 'ninja van', 'gdex', 'dhl'}
_CURRENCIES = {'rm', 'myr', 'usd', 'sgd'}

# Product entities - phone models, car brands, food brands, tech
_PRODUCTS = {
    # Phone brands & models
    'iphone', 'iphone 15', 'iphone 14', 'iphone 13', 'iphone 12',
    'samsung', 'galaxy', 'galaxy s24', 'galaxy s23', 'galaxy a54',
    'xiaomi', 'redmi', 'poco', 'realme', 'oppo', 'vivo', 'huawei',
    'oneplus', 'google pixel', 'pixel', 'nothing phone',
    # Car brands & models (Malaysian market)
    'myvi', 'vios', 'axia', 'bezza', 'saga', 'persona', 'x50', 'x70',
    'ativa', 'aruz', 'alza', 'ertiga', 'city', 'civic', 'accord',
    'camry', 'hilux', 'fortuner', 'innova', 'rush', 'wira', 'waja',
    'iriz', 'exora', 'preve', 'suprima', 'perdana', 'x90',
    'hr-v', 'cr-v', 'br-v', 'wr-v', 'jazz', 'fit',
    'ranger', 'triton', 'navara', 'd-max', 'everest', 'territory',
    'almera', 'kicks', 'serena', 'x-trail',
    # Food & beverage brands
    'maggi', 'milo', 'nescafe', 'dutch lady', 'gardenia', 'massimo',
    'munchy', 'julie', 'mamee', 'cintan', 'boh', 'lipton',
    'yakult', 'vitagen', 'f&n', 'kickapoo', '100plus', 'revive',
    # Tech products
    'macbook', 'ipad', 'airpods', 'apple watch', 'ps5', 'ps4',
    'xbox', 'nintendo switch', 'switch', 'steam deck',
    'chatgpt', 'claude', 'gemini',
}

# Event entities - festivals, celebrations, gatherings
_EVENTS = {
    # Malaysian festivals & holidays
    'hari raya', 'hari raya aidilfitri', 'hari raya haji', 'aidilfitri',
    'aidiladha', 'raya', 'deepavali', 'diwali', 'thaipusam',
    'cny', 'chinese new year', 'tahun baru cina', 'gong xi fa cai',
    'wesak', 'christmas', 'krismas', 'easter',
    'merdeka', 'hari merdeka', 'malaysia day', 'hari malaysia',
    'hari kebangsaan', 'hari wilayah', 'hari pekerja', 'labour day',
    'israk mikraj', 'nuzul quran', 'maal hijrah', 'maulidur rasul',
    'hari guru', 'hari ibu', 'hari bapa', 'hari kanak-kanak',
    'tahun baru', 'new year', 'countdown',
    # Social events
    'majlis', 'kenduri', 'kenduri kahwin', 'wedding', 'nikah',
    'akad', 'resepsi', 'reception', 'tunang', 'engagement',
    'birthday', 'hari jadi', 'anniversary', 'ulang tahun',
    'reunion', 'gathering', 'jamuan', 'open house',
    'convocation', 'konvokesyen', 'graduation', 'grad',
    'concert', 'konsert', 'festival', 'carnival', 'karnival',
    'pameran', 'exhibition', 'expo', 'seminar', 'workshop',
    'hackathon', 'webinar', 'conference', 'persidangan',
    'pertandingan', 'tournament', 'competition',
    # Sports events
    'piala malaysia', 'liga super', 'piala fa', 'sea games',
    'sukan sea', 'olimpik', 'olympics', 'world cup', 'piala dunia',
    'f1', 'motogp', 'formula 1',
}

# Pre-compiled regex patterns for NER
_RE_PHONE = re.compile(r'\+?6?0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}')
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_RE_MONEY = re.compile(r'(?:RM|MYR|rm)\s?[\d,]+(?:\.\d{1,2})?')
_RE_DATE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
_RE_TIME = re.compile(r'\d{1,2}[:.]\d{2}\s*(?:am|pm|AM|PM)?')


def ner_tag(text: str) -> List[Dict[str, Any]]:
    """Identify named entities in Malaysian text.
    
    Entity types:
    - PERSON: Person names (detected via titles, capitalization)
    - LOCATION: Malaysian states, cities, places
    - ORGANIZATION: Companies, government bodies, universities
    - MONEY: Currency amounts
    - DATE: Date expressions
    - TIME: Time expressions
    - PHONE: Malaysian phone numbers
    - URL: Web URLs
    - EMAIL: Email addresses
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list[dict]: List of entities with 'text', 'type', 'start', 'end'.
    
    Example:
        >>> malaysian_manglish_nlp.ner_tag("Aku kat KL, nak jumpa Encik Ahmad esok")
        [{'text': 'KL', 'type': 'LOCATION', 'start': 7, 'end': 9},
         {'text': 'Encik Ahmad', 'type': 'PERSON', 'start': 16, 'end': 27}]
    """
    entities = []
    
    # Phone numbers (Malaysian format)
    for m in _RE_PHONE.finditer(text):
        entities.append({'text': m.group(), 'type': 'PHONE', 'start': m.start(), 'end': m.end()})
    
    # URLs
    for m in _RE_URL.finditer(text):
        entities.append({'text': m.group(), 'type': 'URL', 'start': m.start(), 'end': m.end()})
    
    # Email
    for m in _RE_EMAIL.finditer(text):
        entities.append({'text': m.group(), 'type': 'EMAIL', 'start': m.start(), 'end': m.end()})
    
    # Money (RM50, RM 1,000.50, MYR 100)
    for m in _RE_MONEY.finditer(text):
        entities.append({'text': m.group(), 'type': 'MONEY', 'start': m.start(), 'end': m.end()})
    
    # Date patterns
    for m in _RE_DATE.finditer(text):
        entities.append({'text': m.group(), 'type': 'DATE', 'start': m.start(), 'end': m.end()})
    
    # Time patterns
    for m in _RE_TIME.finditer(text):
        entities.append({'text': m.group(), 'type': 'TIME', 'start': m.start(), 'end': m.end()})
    
    # Word-level entity detection
    words = text.split()
    pos = 0
    skip_until = -1
    
    for i, word in enumerate(words):
        if i <= skip_until:
            pos = text.find(word, pos) + len(word)
            continue
        
        word_start = text.find(word, pos)
        word_end = word_start + len(word)
        pos = word_end
        
        lower = word.lower().strip('.,!?;:')
        
        # Skip if already captured by regex patterns
        if any(e['start'] <= word_start < e['end'] for e in entities):
            continue
        
        # Multi-word location detection (check 2-3 word combos first)
        if i + 1 < len(words):
            two_word = lower + ' ' + words[i + 1].lower().strip('.,!?;:')
            if two_word in _CITIES or two_word in _STATES:
                full = word + ' ' + words[i + 1].strip('.,!?;:')
                entities.append({'text': full, 'type': 'LOCATION', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 1
                continue
            if two_word in _ORGS:
                full = word + ' ' + words[i + 1].strip('.,!?;:')
                entities.append({'text': full, 'type': 'ORGANIZATION', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 1
                continue
            if two_word in _PRODUCTS:
                full = word + ' ' + words[i + 1].strip('.,!?;:')
                entities.append({'text': full, 'type': 'PRODUCT', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 1
                continue
            if two_word in _EVENTS:
                full = word + ' ' + words[i + 1].strip('.,!?;:')
                entities.append({'text': full, 'type': 'EVENT', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 1
                continue
        if i + 2 < len(words):
            three_word = lower + ' ' + words[i + 1].lower().strip('.,!?;:') + ' ' + words[i + 2].lower().strip('.,!?;:')
            if three_word in _CITIES or three_word in _STATES:
                full = word + ' ' + words[i + 1] + ' ' + words[i + 2].strip('.,!?;:')
                entities.append({'text': full, 'type': 'LOCATION', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 2
                continue
            if three_word in _ORGS:
                full = word + ' ' + words[i + 1] + ' ' + words[i + 2].strip('.,!?;:')
                entities.append({'text': full, 'type': 'ORGANIZATION', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 2
                continue
            if three_word in _PRODUCTS:
                full = word + ' ' + words[i + 1] + ' ' + words[i + 2].strip('.,!?;:')
                entities.append({'text': full, 'type': 'PRODUCT', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 2
                continue
            if three_word in _EVENTS:
                full = word + ' ' + words[i + 1] + ' ' + words[i + 2].strip('.,!?;:')
                entities.append({'text': full, 'type': 'EVENT', 'start': word_start, 'end': word_start + len(full)})
                skip_until = i + 2
                continue
        
        # Person detection (title + capitalized word)
        if lower in _PERSON_TITLES and i + 1 < len(words):
            next_word = words[i + 1].strip('.,!?;:')
            if next_word[0:1].isupper():
                full_name = word + ' ' + next_word
                # Check for multi-word name (up to 3 parts)
                j = i + 2
                while j < len(words) and j < i + 4:
                    candidate = words[j].strip('.,!?;:')
                    if candidate[0:1].isupper() and candidate.lower() not in _STATES and candidate.lower() not in _CITIES and candidate.lower() not in ('dan', 'di', 'ke', 'dari'):
                        full_name += ' ' + candidate
                        j += 1
                    else:
                        break
                name_start = text.find(word, word_start)
                entities.append({'text': full_name, 'type': 'PERSON', 'start': name_start, 'end': name_start + len(full_name)})
                skip_until = j - 1
        
        # Standalone capitalized words (potential person names)
        elif word[0:1].isupper() and lower not in _STATES and lower not in _CITIES and lower not in _UNIVERSITIES and lower not in _ORGS:
            # Check if it's a sequence of capitalized words (likely a name)
            if i + 1 < len(words) and words[i + 1][0:1].isupper():
                next_lower = words[i + 1].lower().strip('.,!?;:')
                if next_lower not in _STATES and next_lower not in _CITIES and next_lower not in ('dan', 'di', 'ke', 'dari', 'yang', 'ini', 'itu'):
                    # Likely a person name
                    full_name = word + ' ' + words[i + 1].strip('.,!?;:')
                    entities.append({'text': full_name, 'type': 'PERSON', 'start': word_start, 'end': word_start + len(full_name)})
                    skip_until = i + 1
                    continue
        
        # Location (single word)
        if lower in _STATES or lower in _CITIES:
            entities.append({'text': word.strip('.,!?;:'), 'type': 'LOCATION', 'start': word_start, 'end': word_end})
        
        # Organization (single word)
        elif lower in _UNIVERSITIES:
            entities.append({'text': word.strip('.,!?;:'), 'type': 'ORGANIZATION', 'start': word_start, 'end': word_end})
        elif lower in _ORGS:
            entities.append({'text': word.strip('.,!?;:'), 'type': 'ORGANIZATION', 'start': word_start, 'end': word_end})
        
        # Product (single word)
        elif lower in _PRODUCTS:
            entities.append({'text': word.strip('.,!?;:'), 'type': 'PRODUCT', 'start': word_start, 'end': word_end})
        
        # Event (single word)
        elif lower in _EVENTS:
            entities.append({'text': word.strip('.,!?;:'), 'type': 'EVENT', 'start': word_start, 'end': word_end})
    
    # Sort by position
    entities.sort(key=lambda e: e['start'])
    
    return entities

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract entities grouped by type.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        dict: Entities grouped by type.
    
    Example:
        >>> extract_entities("Jumpa kat UMPSA, KL pukul 3.30pm. Bayar RM50")
        {'LOCATION': ['KL'], 'ORGANIZATION': ['UMPSA'], 'TIME': ['3.30pm'], 'MONEY': ['RM50']}
    """
    entities = ner_tag(text)
    grouped = {}
    
    for e in entities:
        etype = e['type']
        if etype not in grouped:
            grouped[etype] = []
        if e['text'] not in grouped[etype]:
            grouped[etype].append(e['text'])
    
    return grouped
