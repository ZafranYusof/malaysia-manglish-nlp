"""
Data augmentation for Manglish NLP training data.
Generates ~5000-8000 new labeled examples to reach ~13k-15k total.

Methods:
1. Synonym replacement (using package's own augmentation module)
2. Back-translation style paraphrase (BM>EN>BM using simple word mapping)
3. Masked token replacement (random word dropout/swap)
4. Code-switching augmentation (swap BM<->EN words)
"""

import json
import random
import re
import sys
from pathlib import Path
from collections import Counter
from copy import deepcopy

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malaysian_manglish_nlp.augmentation import (
    augment as pkg_augment,
    synonym as pkg_synonym,
    vowel_alternate,
    replace_similar_consonants,
    replace_similar_vowels,
    socialmedia_form,
)

random.seed(42)

# BM <-> EN word pairs for code-switching augmentation
BM_EN_PAIRS = {
    'saya': 'I', 'aku': 'I', 'kau': 'you', 'awak': 'you',
    'dia': 'he/she', 'mereka': 'they', 'kami': 'we', 'kita': 'we',
    'ini': 'this', 'itu': 'that', 'ada': 'have/there is',
    'tidak': 'no/not', 'tak': 'not', 'bukan': 'not',
    'dan': 'and', 'atau': 'or', 'tapi': 'but', 'tetapi': 'but',
    'sebab': 'because', 'kerana': 'because',
    'besar': 'big', 'kecil': 'small', 'banyak': 'many/much',
    'sikit': 'little', 'sedikit': 'little',
    'baik': 'good', 'buruk': 'bad', 'cantik': 'beautiful',
    'makan': 'eat', 'minum': 'drink', 'tidur': 'sleep',
    'pergi': 'go', 'datang': 'come', 'balik': 'return/go back',
    'kerja': 'work', 'belajar': 'study', 'main': 'play',
    'rumah': 'house', 'kereta': 'car', 'telefon': 'phone',
    'duit': 'money', 'masa': 'time', 'hari': 'day',
    'orang': 'people/person', 'kawan': 'friend',
    'suka': 'like', 'takut': 'scared', 'marah': 'angry',
    'gembira': 'happy', 'sedih': 'sad', 'penat': 'tired',
    'lapar': 'hungry', 'dahaga': 'thirsty',
    'panas': 'hot', 'sejuk': 'cold', 'hujan': 'rain',
    'cepat': 'fast', 'lambat': 'slow',
    'mahal': 'expensive', 'murah': 'cheap',
    'senang': 'easy', 'susah': 'difficult',
    'sudah': 'already', 'belum': 'not yet',
    'boleh': 'can', 'nak': 'want', 'mahu': 'want',
    'tahu': 'know', 'faham': 'understand',
    'cakap': 'say/talk', 'tanya': 'ask', 'jawab': 'answer',
    'buat': 'do/make', 'guna': 'use', 'cuba': 'try',
    'tengok': 'watch/look', 'dengar': 'listen/hear',
    'rasa': 'feel', 'pikir': 'think',
    'betul': 'correct', 'salah': 'wrong',
    'sama': 'same', 'beza': 'different',
    'baru': 'new', 'lama': 'old/long time',
    'dekat': 'near', 'jauh': 'far',
    'naik': 'go up', 'turun': 'go down',
    'buka': 'open', 'tutup': 'close',
    'ambil': 'take', 'bagi': 'give',
    'hantar': 'send', 'terima': 'receive',
    'beli': 'buy', 'jual': 'sell',
    'bayar': 'pay', 'pinjam': 'borrow',
    'tolong': 'help', 'ajar': 'teach',
}

EN_BM_PAIRS = {v: k for k, v in BM_EN_PAIRS.items()}

# Common Manglish fillers/particles to add/remove
PARTICLES = ['lah', 'la', 'lor', 'meh', 'kan', 'ah', 'eh', 'wei', 'weh', 'oi']
FILLERS = ['macam', 'macam tu', 'macam ni', 'macam biasa', 'gitu', 'gini']


def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def synonym_augment(text, n_replacements=2):
    """Replace words with synonyms from augmentation module."""
    words = text.split()
    if len(words) < 3:
        return None
    
    new_words = list(words)
    changed = False
    
    for i, word in enumerate(words):
        if random.random() < 0.25:  # 25% chance per word
            try:
                syns = pkg_synonym(word.lower(), top_n=3)
                if syns and len(syns) > 0:
                    replacement = random.choice(syns)
                    if replacement != word.lower():
                        new_words[i] = replacement
                        changed = True
            except:
                pass
    
    if not changed:
        # Fallback: use package augment
        try:
            results = pkg_augment(text, n=1)
            if results and results[0] != text:
                return results[0]
        except:
            pass
        return None
    
    return ' '.join(new_words)


def code_switch_augment(text):
    """Swap BM words to EN or vice versa (code-switching augmentation)."""
    words = text.split()
    new_words = []
    changed = False
    
    for word in words:
        lower = word.lower().strip('.,!?;:')
        # Try BM -> EN
        if lower in BM_EN_PAIRS and random.random() < 0.3:
            new_words.append(BM_EN_PAIRS[lower])
            changed = True
        # Try EN -> BM
        elif lower in EN_BM_PAIRS and random.random() < 0.3:
            new_words.append(EN_BM_PAIRS[lower])
            changed = True
        else:
            new_words.append(word)
    
    if not changed:
        return None
    return ' '.join(new_words)


def word_dropout_augment(text, dropout_rate=0.1):
    """Randomly drop words (simulates casual speech patterns)."""
    words = text.split()
    if len(words) < 4:
        return None
    
    new_words = [w for w in words if random.random() > dropout_rate]
    if len(new_words) < len(words) * 0.7 or len(new_words) < 3:
        return None
    
    result = ' '.join(new_words)
    return result if result != text else None


def particle_augment(text):
    """Add/remove/swap Manglish particles."""
    words = text.split()
    
    if random.random() < 0.5 and len(words) > 2:
        # Add a particle at end or after first clause
        particle = random.choice(PARTICLES)
        insert_pos = random.choice([len(words), len(words)//2])
        words.insert(insert_pos, particle)
    else:
        # Remove a particle if present
        for i, w in enumerate(words):
            if w.lower() in PARTICLES:
                words.pop(i)
                break
    
    result = ' '.join(words)
    return result if result != text else None


def word_order_shuffle(text):
    """Slightly shuffle adjacent words (simulates casual word order)."""
    words = text.split()
    if len(words) < 4:
        return None
    
    new_words = list(words)
    # Swap 1-2 pairs of adjacent words
    n_swaps = random.randint(1, 2)
    for _ in range(n_swaps):
        i = random.randint(1, len(new_words) - 2)
        new_words[i], new_words[i-1] = new_words[i-1], new_words[i]
    
    result = ' '.join(new_words)
    return result if result != text else None


def augment_sample(sample, method=None):
    """Apply augmentation to a single sample."""
    text = sample['text']
    
    methods = [
        synonym_augment,
        code_switch_augment,
        word_dropout_augment,
        particle_augment,
        word_order_shuffle,
    ]
    
    if method is None:
        method = random.choice(methods)
    
    try:
        new_text = method(text)
    except Exception:
        return None
    
    if new_text is None or new_text == text or len(new_text.strip()) < 3:
        return None
    
    new_sample = deepcopy(sample)
    new_sample['text'] = new_text
    new_sample['is_augmented'] = True
    return new_sample


def generate_augmented_data(input_path, output_path, target_new=6000):
    """Generate augmented dataset."""
    data = load_data(input_path)
    print(f"Loaded {len(data)} original samples")
    
    # Analyze class distribution for targeted augmentation
    emotion_counts = Counter(d['emotion'] for d in data)
    intent_counts = Counter(d['intent'] for d in data)
    sentiment_counts = Counter(d['sentiment'] for d in data)
    
    # Identify minority classes to oversample
    minority_emotions = {e for e, c in emotion_counts.items() if c < 300}  # love, disgust, surprise, fear
    minority_intents = {i for i, c in intent_counts.items() if c < 400}  # greeting, request, question
    
    print(f"Minority emotions: {minority_emotions}")
    print(f"Minority intents: {minority_intents}")
    
    augmented = []
    attempts = 0
    max_attempts = target_new * 5
    
    # Phase 1: Oversample minority classes (40% of target)
    minority_target = int(target_new * 0.4)
    minority_samples = [d for d in data if d['emotion'] in minority_emotions or d['intent'] in minority_intents]
    
    print(f"Phase 1: Oversampling {len(minority_samples)} minority-class samples (target: {minority_target})")
    
    while len(augmented) < minority_target and attempts < max_attempts:
        sample = random.choice(minority_samples)
        new_sample = augment_sample(sample)
        if new_sample:
            augmented.append(new_sample)
        attempts += 1
    
    # Phase 2: General augmentation (60% of target)
    general_target = target_new - len(augmented)
    print(f"Phase 2: General augmentation (target: {general_target})")
    
    while len(augmented) < minority_target + general_target and attempts < max_attempts:
        sample = random.choice(data)
        new_sample = augment_sample(sample)
        if new_sample:
            augmented.append(new_sample)
        attempts += 1
    
    print(f"Generated {len(augmented)} augmented samples (attempts: {attempts})")
    
    # Combine original + augmented
    combined = data + augmented
    
    # Mark originals
    for d in data:
        d.setdefault('is_augmented', False)
    
    # Write combined dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in combined:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Combined dataset: {len(combined)} samples saved to {output_path}")
    
    # Print new distribution
    new_emotions = Counter(d['emotion'] for d in combined)
    new_intents = Counter(d['intent'] for d in combined)
    new_sentiments = Counter(d['sentiment'] for d in combined)
    
    print("\n=== NEW DISTRIBUTION ===")
    print("\nSentiment:")
    for k, v in new_sentiments.most_common():
        print(f"  {k}: {v} ({v/len(combined)*100:.1f}%)")
    print("\nEmotion:")
    for k, v in new_emotions.most_common():
        print(f"  {k}: {v} ({v/len(combined)*100:.1f}%)")
    print("\nIntent:")
    for k, v in new_intents.most_common():
        print(f"  {k}: {v} ({v/len(combined)*100:.1f}%)")
    
    return combined


if __name__ == '__main__':
    input_path = 'datasets/manglish_7884.jsonl'
    output_path = 'datasets/manglish_augmented.jsonl'
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    generate_augmented_data(input_path, output_path, target_new=6500)
