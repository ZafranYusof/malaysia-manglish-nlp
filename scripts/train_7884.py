"""
Train Manglish multi-task sentiment model on 7884 examples.
Augments existing dataset, then runs training.
"""
import json
import os
import random
import sys
import copy

random.seed(42)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_PATH = os.path.join(PROJECT_DIR, 'datasets', 'manglish_full.jsonl')
AUGMENTED_PATH = os.path.join(PROJECT_DIR, 'datasets', 'manglish_7884.jsonl')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'malaysian_manglish_nlp', 'resources', 'manglish_finetuned')

TARGET_COUNT = 7884

def load_data(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

# Manglish synonym maps for augmentation
SYNONYMS = {
    'sedap': ['lazat', 'enak', 'yummy', 'delicious', 'tasty'],
    'best': ['terbaik', 'hebat', 'power', 'mantap', 'padu'],
    'teruk': ['hampeh', 'hancur', 'parah', 'buruk', 'sampah'],
    'cantik': ['lawa', 'cun', 'gorgeous', 'beautiful', 'stunning'],
    'marah': ['bengang', 'geram', 'naik angin', 'triggered', 'murka'],
    'gembira': ['happy', 'seronok', 'syok', 'ceria', 'excited'],
    'sedih': ['pilu', 'sayu', 'down', 'murung', 'duka'],
    'takut': ['gerun', 'seram', 'cuak', 'nervous', 'gabra'],
    'suka': ['minat', 'gemar', 'enjoy', 'love', 'fancy'],
    'besar': ['gedang', 'luas', 'mega', 'huge', 'big'],
    'kecil': ['mungil', 'mini', 'comel', 'small', 'tiny'],
    'pandai': ['bijak', 'cerdik', 'smart', 'genius', 'pintar'],
    'bodoh': ['bangang', 'bebal', 'stupid', 'dungu', 'bengap'],
    'cepat': ['laju', 'pantas', 'fast', 'speed', 'express'],
    'lambat': ['lewat', 'slow', 'lembab', 'lengah', 'perlahan'],
    'makan': ['jamu', 'santap', 'ngap', 'ratah', 'telan'],
    'pergi': ['pegi', 'gi', 'gerak', 'berangkat', 'bertolak'],
    'balik': ['pulang', 'blk', 'cabut', 'chow', 'return'],
    'bagus': ['good', 'great', 'nice', 'excellent', 'superb'],
    'susah': ['payah', 'hard', 'difficult', 'tough', 'mencabar'],
    'senang': ['mudah', 'easy', 'simple', 'ringkas', 'straightforward'],
    'murah': ['cheap', 'affordable', 'berpatutan', 'jimat', 'value'],
    'mahal': ['expensive', 'costly', 'pricey', 'premium', 'overpriced'],
    'kawan': ['member', 'bro', 'geng', 'buddy', 'fren'],
    'kerja': ['work', 'job', 'keje', 'hustle', 'grind'],
    'tidur': ['sleep', 'tido', 'rest', 'zzz', 'pengsan'],
    'rumah': ['umah', 'rmh', 'home', 'kediaman', 'crib'],
    'kereta': ['keta', 'car', 'ride', 'whip', 'kenderaan'],
    'duit': ['wang', 'ringgit', 'cash', 'money', 'pitih'],
    'telefon': ['phone', 'hp', 'handphone', 'fon', 'device'],
    'penat': ['tired', 'exhausted', 'lelah', 'lesu', 'burnt out'],
    'stress': ['tension', 'pressure', 'overwhelm', 'serabut', 'pening'],
    'boring': ['bosan', 'sien', 'dull', 'muak', 'jelak'],
    'happy': ['gembira', 'seronok', 'syok', 'glad', 'pleased'],
    'love': ['cinta', 'sayang', 'adore', 'suka sangat', 'cherish'],
    'nice': ['bagus', 'elok', 'cantik', 'good', 'pleasant'],
    'good': ['bagus', 'baik', 'great', 'fine', 'solid'],
    'bad': ['teruk', 'buruk', 'hampeh', 'poor', 'awful'],
    'food': ['makanan', 'lauk', 'hidangan', 'juadah', 'menu'],
    'service': ['layanan', 'servis', 'pelayanan', 'attention', 'hospitality'],
}

NEGATORS = ['tak', 'tidak', 'bukan', 'x', 'takde', 'xde']
INTENSIFIERS = ['gila', 'sangat', 'betul', 'memang', 'super', 'really', 'very', 'extremely', 'sgt', 'mmg']
PARTICLES = ['la', 'lah', 'je', 'jer', 'ni', 'tu', 'kan', 'kot', 'pun', 'ke', 'dah', 'wei', 'weh']

TOPICS = ['food', 'service', 'transport', 'weather', 'entertainment', 'sports',
          'politics', 'education', 'health', 'technology', 'lifestyle', 'travel',
          'shopping', 'work', 'relationships', 'housing', 'finance']

SOURCES = ['twitter', 'reddit', 'whatsapp', 'lowyat', 'facebook', 'instagram', 'tiktok']

EMOTIONS_MAP = {
    'positive': ['happy', 'love', 'surprise', 'neutral'],
    'negative': ['sad', 'angry', 'fear', 'disgust', 'neutral'],
    'neutral': ['neutral', 'surprise'],
}

INTENTS = ['opinion', 'statement', 'complaint', 'question', 'request', 'greeting']


def synonym_replace(text, max_replacements=2):
    """Replace words with synonyms."""
    words = text.split()
    replaced = 0
    for i, word in enumerate(words):
        if replaced >= max_replacements:
            break
        wl = word.lower().strip('.,!?')
        if wl in SYNONYMS:
            syn = random.choice(SYNONYMS[wl])
            # Preserve capitalization
            if word[0].isupper():
                syn = syn.capitalize()
            words[i] = syn
            replaced += 1
    if replaced == 0:
        return None
    return ' '.join(words)


def add_particle(text):
    """Add Manglish particle."""
    words = text.split()
    if len(words) < 3:
        return None
    particle = random.choice(PARTICLES)
    # Insert at random position (not first/last)
    pos = random.randint(1, len(words) - 1)
    words.insert(pos, particle)
    return ' '.join(words)


def add_intensifier(text):
    """Add intensifier before sentiment word."""
    words = text.split()
    if len(words) < 2:
        return None
    intensifier = random.choice(INTENSIFIERS)
    pos = random.randint(0, max(0, len(words) - 2))
    words.insert(pos, intensifier)
    return ' '.join(words)


def negate_flip(text, sentiment):
    """Add negation and flip sentiment label."""
    words = text.split()
    negator = random.choice(NEGATORS)
    # Insert after first 1-2 words
    pos = min(random.randint(1, 2), len(words))
    words.insert(pos, negator)
    new_sentiment = 'negative' if sentiment == 'positive' else 'positive'
    return ' '.join(words), new_sentiment


def augment_sample(sample):
    """Create augmented version of a sample."""
    text = sample['text']
    sentiment = sample.get('sentiment', 'neutral')
    
    strategy = random.choice(['synonym', 'particle', 'intensifier', 'negate', 'synonym2'])
    
    new_text = None
    new_sentiment = sentiment
    
    if strategy == 'synonym' or strategy == 'synonym2':
        new_text = synonym_replace(text, max_replacements=random.randint(1, 3))
    elif strategy == 'particle':
        new_text = add_particle(text)
    elif strategy == 'intensifier':
        new_text = add_intensifier(text)
    elif strategy == 'negate':
        result = negate_flip(text, sentiment)
        if result:
            new_text, new_sentiment = result
    
    if not new_text or new_text == text:
        # Fallback: synonym replace
        new_text = synonym_replace(text, max_replacements=1)
        if not new_text or new_text == text:
            return None
    
    new_sample = copy.deepcopy(sample)
    new_sample['text'] = new_text
    new_sample['sentiment'] = new_sentiment
    
    # Update emotion based on new sentiment
    if new_sentiment != sentiment:
        emotions = EMOTIONS_MAP.get(new_sentiment, ['neutral'])
        new_sample['emotion'] = random.choice(emotions)
    
    return new_sample


def main():
    print(f"Loading data from: {DATA_PATH}")
    data = load_data(DATA_PATH)
    print(f"Original count: {len(data)}")
    
    needed = TARGET_COUNT - len(data)
    print(f"Need {needed} more examples to reach {TARGET_COUNT}")
    
    if needed > 0:
        augmented = []
        attempts = 0
        max_attempts = needed * 10
        
        # Group by sentiment for balanced augmentation
        by_sentiment = {}
        for s in data:
            sent = s.get('sentiment', 'neutral')
            by_sentiment.setdefault(sent, []).append(s)
        
        print(f"Sentiment distribution: {', '.join(f'{k}={len(v)}' for k, v in by_sentiment.items())}")
        
        while len(augmented) < needed and attempts < max_attempts:
            # Prefer augmenting underrepresented classes
            if len(augmented) % 3 == 0 and 'neutral' in by_sentiment:
                source = random.choice(by_sentiment['neutral'])
            elif len(augmented) % 3 == 1 and 'negative' in by_sentiment:
                source = random.choice(by_sentiment['negative'])
            else:
                source = random.choice(data)
            
            new_sample = augment_sample(source)
            if new_sample and new_sample['text'] != source['text']:
                augmented.append(new_sample)
            attempts += 1
        
        print(f"Augmented: {len(augmented)} new examples (attempts: {attempts})")
        data = data + augmented
    
    # Trim to exact target
    data = data[:TARGET_COUNT]
    print(f"Final count: {len(data)}")
    
    # Fix labels: map 'mixed' -> 'neutral', 'sarcasm' intent -> 'opinion'
    for s in data:
        if s.get('sentiment') == 'mixed':
            s['sentiment'] = 'neutral'
        if s.get('intent') == 'sarcasm':
            s['intent'] = 'opinion'
    
    # Save augmented dataset
    with open(AUGMENTED_PATH, 'w', encoding='utf-8') as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Saved to: {AUGMENTED_PATH}")
    
    # Print final distribution
    sent_dist = {}
    emo_dist = {}
    intent_dist = {}
    for s in data:
        sent_dist[s.get('sentiment', '?')] = sent_dist.get(s.get('sentiment', '?'), 0) + 1
        emo_dist[s.get('emotion', '?')] = emo_dist.get(s.get('emotion', '?'), 0) + 1
        intent_dist[s.get('intent', '?')] = intent_dist.get(s.get('intent', '?'), 0) + 1
    
    print(f"\nFinal sentiment: {sent_dist}")
    print(f"Final emotion: {emo_dist}")
    print(f"Final intent: {intent_dist}")
    
    # Now train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Add project to path
    sys.path.insert(0, PROJECT_DIR)
    
    from malaysian_manglish_nlp.transformers.finetune import train
    
    results = train(
        data_path=AUGMENTED_PATH,
        output_dir=OUTPUT_DIR,
        epochs=5,
        batch_size=16,
        lr=2e-5,
        max_length=128,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best val accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Model saved to: {results['output_dir']}")
    print(f"History: {json.dumps(results['history'], indent=2)}")
    
    # Verify model file exists
    model_path = os.path.join(OUTPUT_DIR, 'model.pt')
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\nmodel.pt exists: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"\nERROR: model.pt NOT found at {model_path}")


if __name__ == '__main__':
    main()
