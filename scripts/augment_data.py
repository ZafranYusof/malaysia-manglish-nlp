"""
Augment existing Manglish labeled data with 10 synthetic strategies.

Reads manglish_labeled.jsonl + manglish_labeled_v2.jsonl (~1139 examples),
generates 5-8 augmented variants per example, outputs ~8000+ new examples
to manglish_labeled_v3.jsonl.

Strategies:
  1. Synonym replacement
  2. Shortform variation (toggle shortforms <-> full words)
  3. Code-switching injection
  4. Slang particle injection
  5. Emoji variation
  6. Negation flip (positive <-> negative)
  7. Aspect swap (food <-> service <-> price etc)
  8. Length variation (shorten/lengthen)
  9. Spelling variation (common misspellings)
 10. Dialect variation

Usage:
    python scripts/augment_data.py
    python scripts/augment_data.py --variants 6 --output datasets/manglish_labeled_v3.jsonl
"""

import json
import random
import re
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------------

def load_jsonl(filepath: Path) -> list[dict]:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Dictionaries / lookup tables
# ---------------------------------------------------------------------------

# Shortforms -> full forms (from manglish_nlp dictionary + extras)
SHORTFORMS = {
    "nk": "nak", "mcm": "macam", "mcam": "macam", "brp": "berapa",
    "dkt": "dekat", "utk": "untuk", "dgn": "dengan", "yg": "yang",
    "sbb": "sebab", "xpe": "takpe", "xd": "tiada", "xde": "tiada",
    "tk": "tak", "dh": "dah", "je": "sahaja", "kot": "barangkali",
    "mmg": "memang", "bg": "bagi", "blh": "boleh", "bkn": "bukan",
    "dr": "dari", "dpn": "depan", "blkng": "belakang", "kt": "kat",
    "skrg": "sekarang", "skolah": "sekolah", "sklh": "sekolah",
    "uni": "universiti", "keta": "kereta", "keje": "kerja",
    "tido": "tidur", "mkn": "makan", "minum": "minum", "blk": "balik",
    "pd": "pada", "utk": "untuk", "drpd": "daripada", "kpd": "kepada",
    "sj": "sahaja", "shj": "sahaja", "jgn": "jangan",
    "sy": "saya", "awak": "awak", "org": "orang",
    "tmpt": "tempat", "umah": "rumah", "rmh": "rumah",
    "hp": "handphone", "fon": "telefon",
    "cmna": "macam mana", "cmne": "macam mana", "cane": "macam mana",
    "gini": "begini", "gitu": "begitu", "ni": "ini", "tu": "itu",
}

# Reverse: full -> shortform
FULL_TO_SHORT = {}
for k, v in SHORTFORMS.items():
    if v not in FULL_TO_SHORT:
        FULL_TO_SHORT[v] = []
    FULL_TO_SHORT[v].append(k)

# Synonym map (Malay + Manglish)
SYNONYMS = {
    "sedap": ["best", "power", "mantap", "padu", "lazat", "enak"],
    "best": ["syok", "power", "mantap", "padu", "hebat", "terbaik"],
    "bagus": ["best", "power", "mantap", "solid", "padu", "terbaik"],
    "cantik": ["lawa", "cun", "gorgeous", "molek", "ayu"],
    "besar": ["gedang", "gigantic", "huge", "mega"],
    "kecil": ["comel", "mungil", "mini", "tiny"],
    "pandai": ["bijak", "cerdik", "pintar", "genius", "smart"],
    "bodoh": ["bangang", "bebal", "dungu", "bengap", "tolol"],
    "makan": ["jamu", "santap", "ngap", "ratah", "melantak"],
    "pergi": ["pegi", "gi", "gerak", "berangkat", "chow"],
    "cepat": ["laju", "pantas", "segera", "speed", "express"],
    "lambat": ["lewat", "perlahan", "slow", "lembab"],
    "gembira": ["happy", "seronok", "syok", "riang", "excited"],
    "sedih": ["pilu", "sayu", "down", "murung", "duka"],
    "marah": ["bengang", "geram", "naik angin", "triggered", "murka"],
    "takut": ["cuak", "seram", "gabra", "nervous", "gerun"],
    "suka": ["minat", "gemar", "enjoy", "fancy", "into"],
    "murah": ["cheap", "berpatutan", "affordable", "jimat"],
    "mahal": ["expensive", "pricey", "premium", "costly"],
    "senang": ["mudah", "easy", "simple", "ringkas"],
    "susah": ["payah", "hard", "tough", "mencabar"],
    "panas": ["hot", "terik", "hangat", "menyengat"],
    "sejuk": ["cold", "dingin", "cool", "beku"],
    "kawan": ["member", "bro", "geng", "buddy", "fren"],
    "teruk": ["hampeh", "hancur", "parah", "kronik", "terrible"],
    "baik": ["ok", "fine", "alright", "okay", "okey"],
    "buruk": ["teruk", "hampeh", "sampah", "useless"],
    "gila": ["crazy", "mad", "insane", "sengal", "mental"],
    "sangat": ["gila", "damn", "really", "betul2", "memang"],
    "hebat": ["power", "mantap", "solid", "padu", "legend"],
    "bosan": ["boring", "membosankan", " tedious", "sakit jiwa"],
    "penat": ["lelah", "tired", "exhausted", "letih", "pancit"],
}

# Code-switch English words commonly inserted into Malay sentences
CODE_SWITCH_WORDS = {
    "food": ["food", "taste", "flavor", "portion", "fresh", "crispy"],
    "service": ["service", "staff", "waiter", "manager", "friendly", "rude"],
    "price": ["price", "cost", "value", "expensive", "cheap", "worth"],
    "tech": ["app", "device", "feature", "update", "bug", "performance"],
    "daily_life": ["traffic", "weather", "mall", "parking", "queue"],
    "politics": ["policy", "corruption", "reform", "democracy", "election"],
    "sports": ["match", "goal", "team", "player", "coach", "champion"],
    "entertainment": ["movie", "song", "concert", "drama", "actor", "plot"],
    "education": ["exam", "assignment", "lecture", "campus", "grade"],
    "religion": ["blessing", "prayer", "faith", "community"],
}

# Malaysian particles / slang fillers
PARTICLES = [
    "lah", "la", "wei", "weh", "eh", "kan", "kot", "ni", "tu",
    "dude", "bro", "sis", "man", "leh", "meh",
]

# Emoji sets by sentiment
EMOJI_POSITIVE = ["😊", "😂", "🔥", "💪", "👍", "❤️", "🎉", "✨", "💯", "🤣", "😍", "🙌", "💐"]
EMOJI_NEGATIVE = ["😤", "😡", "😭", "💔", "😒", "🤮", "😩", "👎", "💀", "😑", "🙄", "😞"]
EMOJI_NEUTRAL = ["🤔", "😐", "📌", "ℹ️", "👀", "🗿", "🫡"]

# Aspect templates for swapping (template + topic)
ASPECT_TEMPLATES = {
    "food": {
        "food": ["{subject} sedap gila", "{subject} punya makanan power", "makanan dia {adj}"],
        "service": ["service kat {subject} hampeh", "{subject} punya staff rude gila", "layan customer mcm sampah kat {subject}"],
        "price": ["{subject} mahal gila", "harga kat {subject} tak berpatutan", "{subject} punya price okay la"],
    },
    "tech": {
        "tech": ["{subject} punya performance mantap", "{subject} ni canggih gila", "feature {subject} best"],
        "service": ["customer service {subject} teruk", "{subject} punya support lambat", "{subject} staff tak reti"],
        "price": ["{subject} overpriced", "harga {subject} berbaloi", "{subject} mahal tapi worth it"],
    },
}

# Dialect word maps
DIALECT_MAPS = {
    "kelantan": {
        "saya": "ambo", "awak": "mung", "kamu": "mung", "dia": "dio",
        "tidak": "tok", "tak": "tok", "ada": "ado", "pergi": "gi",
        "makan": "make", "nasi": "nase", "ikan": "ike", "apa": "gapo",
        "mana": "mano", "macam": "maca", "sangat": "sapa", "boleh": "buleh",
        "orang": "oghe", "rumah": "ghumoh", "besar": "beso",
    },
    "terengganu": {
        "saya": "ambo", "awak": "mung", "dia": "die",
        "tidak": "tak", "ada": "ade", "pergi": "pegi",
        "makan": "make", "apa": "ape", "mana": "mane",
        "orang": "oghang", "rumah": "ghumah", "boleh": "buleh",
    },
    "kedah": {
        "saya": "cheq", "awak": "hang", "kamu": "hang", "dia": "dia",
        "tidak": "tak", "ada": "ada", "pergi": "pi",
        "makan": "makan", "apa": "apa", "mana": "mana",
        "orang": "depa", "kami": "kamek", "korang": "hangpa",
    },
    "n9": {
        "saya": "den", "awak": "kau", "dia": "dio",
        "tidak": "tak", "ada": "ado", "pergi": "pogi",
        "apa": "apo", "mana": "mono", "orang": "ughang",
        "rumah": "ghumah", "boleh": "buleh",
    },
    "sarawak": {
        "saya": "kamek", "awak": "kitak", "dia": "nya",
        "tidak": "sik", "ada": "ada", "pergi": "pergi",
        "makan": "makan", "apa": "apa", "mana": "kitak",
        "orang": "urang", "rumah": "rumah",
    },
}

# Spelling noise patterns (common Malaysian texting misspellings)
SPELLING_NOISE = [
    (r"([a-z])\1{2,}", lambda m: m.group(1) * random.randint(1, 2)),  # reduce elongation
    (r"([a-z])$", lambda m: m.group(1) * random.randint(2, 4)),  # elongate final char
    (r"\b([a-z]{4,})\b", None),  # will apply vowel removal randomly
]

# Intensifier words
INTENSIFIERS = ["gila", "sangat", "memang", "betul2", "damn", "really", "super", "extremely"]

# Negators
NEGATORS = ["tak", "tidak", "x", "bukan", "jangan", "takde", "xde"]


# ---------------------------------------------------------------------------
# Augmentation strategies
# ---------------------------------------------------------------------------

def strategy_synonym_replacement(text: str, entry: dict) -> str:
    """Replace words with Manglish synonyms."""
    words = text.split()
    if not words:
        return text
    new_words = []
    replaced = False
    for w in words:
        lower = w.lower().strip(".,!?;:'\"")
        if lower in SYNONYMS and random.random() < 0.4:
            syn = random.choice(SYNONYMS[lower])
            # Preserve case loosely
            if w[0].isupper():
                syn = syn.capitalize()
            new_words.append(syn)
            replaced = True
        else:
            new_words.append(w)
    if not replaced and words:
        # Force at least one replacement attempt
        for i, w in enumerate(words):
            lower = w.lower().strip(".,!?;:'\"")
            if lower in SYNONYMS:
                syn = random.choice(SYNONYMS[lower])
                new_words[i] = syn
                break
    return " ".join(new_words)


def strategy_shortform_variation(text: str, entry: dict) -> str:
    """Toggle between shortforms and full words."""
    words = text.split()
    new_words = []
    for w in words:
        lower = w.lower()
        # Shortform -> full
        if lower in SHORTFORMS and random.random() < 0.5:
            new_words.append(SHORTFORMS[lower])
        # Full -> shortform
        elif lower in FULL_TO_SHORT and random.random() < 0.5:
            new_words.append(random.choice(FULL_TO_SHORT[lower]))
        else:
            new_words.append(w)
    return " ".join(new_words)


def strategy_code_switching(text: str, entry: dict) -> str:
    """Randomly insert English words into Malay text."""
    topic = entry.get("topic", "daily_life")
    cs_words = CODE_SWITCH_WORDS.get(topic, CODE_SWITCH_WORDS["daily_life"])
    words = text.split()
    if len(words) < 3:
        return text
    # Insert 1-2 English words at random positions
    n_insert = random.randint(1, 2)
    for _ in range(n_insert):
        pos = random.randint(1, len(words))
        insert_word = random.choice(cs_words)
        words.insert(pos, insert_word)
    return " ".join(words)


def strategy_slang_injection(text: str, entry: dict) -> str:
    """Add Malaysian slang particles."""
    words = text.split()
    if not words:
        return text
    n_particles = random.randint(1, 3)
    for _ in range(n_particles):
        particle = random.choice(PARTICLES)
        pos = random.randint(0, len(words))
        words.insert(pos, particle)
    return " ".join(words)


def strategy_emoji_variation(text: str, entry: dict) -> str:
    """Add/remove/swap emojis based on sentiment."""
    sentiment = entry.get("sentiment", "neutral")
    # Remove existing emojis first (simple regex)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    clean_text = emoji_pattern.sub("", text).strip()

    # Add new emojis matching sentiment
    if sentiment == "positive":
        emojis = random.sample(EMOJI_POSITIVE, k=random.randint(1, 3))
    elif sentiment == "negative":
        emojis = random.sample(EMOJI_NEGATIVE, k=random.randint(1, 3))
    else:
        emojis = random.sample(EMOJI_NEUTRAL, k=random.randint(1, 2))

    emoji_str = " ".join(emojis)
    # Randomly place at start, end, or both
    placement = random.choice(["start", "end", "both"])
    if placement == "start":
        return f"{emoji_str} {clean_text}"
    elif placement == "end":
        return f"{clean_text} {emoji_str}"
    else:
        half = len(emojis) // 2
        return f"{' '.join(emojis[:half])} {clean_text} {' '.join(emojis[half:])}"


def strategy_negation_flip(text: str, entry: dict) -> str:
    """Flip sentiment by adding/removing negation. Label will be adjusted by caller."""
    words = text.split()
    # Check if negator already present
    has_negator = any(w.lower() in NEGATORS for w in words)
    if has_negator:
        # Remove first negator
        new_words = []
        removed = False
        for w in words:
            if w.lower() in NEGATORS and not removed:
                removed = True
                continue
            new_words.append(w)
        return " ".join(new_words)
    else:
        # Add negator before adjective/verb (heuristic: after first 1-2 words)
        negator = random.choice(["tak", "x", "bukan"])
        pos = min(random.randint(1, 2), len(words))
        words.insert(pos, negator)
        return " ".join(words)


def strategy_aspect_swap(text: str, entry: dict) -> str:
    """Change the aspect being discussed (food->service->price)."""
    topic = entry.get("topic", "food")
    sentiment = entry.get("sentiment", "positive")

    # Use templates if available
    if topic in ASPECT_TEMPLATES:
        aspects = list(ASPECT_TEMPLATES[topic].keys())
        new_aspect = random.choice(aspects)
        templates = ASPECT_TEMPLATES[topic][new_aspect]
        template = random.choice(templates)
        subjects = ["kedai ni", "restoran tu", "tempat ni", "mamak tu", "kafe ni"]
        adjs_pos = ["sedap", "best", "mantap", "power", "padu"]
        adjs_neg = ["teruk", "hampeh", "sampah", "hancur", "busuk"]
        subject = random.choice(subjects)
        adj = random.choice(adjs_pos if sentiment == "positive" else adjs_neg)
        return template.format(subject=subject, adj=adj)

    # Fallback: just swap topic keywords
    return text


def strategy_length_variation(text: str, entry: dict) -> str:
    """Shorten or lengthen while keeping sentiment."""
    words = text.split()
    if len(words) < 4:
        # Too short to shorten, lengthen instead
        intensifier = random.choice(INTENSIFIERS)
        words.append(intensifier)
        return " ".join(words)

    action = random.choice(["shorten", "lengthen"])
    if action == "shorten":
        # Remove filler words and keep core message
        fillers = {"la", "lah", "wei", "weh", "eh", "kan", "ni", "tu", "laa",
                   "bro", "dude", "man", "sis", "mmg", "memang", "betul2"}
        new_words = [w for w in words if w.lower() not in fillers]
        if len(new_words) < 2:
            return " ".join(words)
        # Also randomly drop 1-2 non-essential words
        if len(new_words) > 4:
            drop_count = random.randint(1, 2)
            for _ in range(drop_count):
                idx = random.randint(1, len(new_words) - 2)
                new_words.pop(idx)
        return " ".join(new_words)
    else:
        # Lengthen: add intensifier, filler, or elaboration
        additions = random.sample(INTENSIFIERS + PARTICLES, k=random.randint(1, 3))
        for add in additions:
            pos = random.randint(0, len(words))
            words.insert(pos, add)
        return " ".join(words)


def strategy_spelling_variation(text: str, entry: dict) -> str:
    """Introduce common Malaysian texting misspellings."""
    words = text.split()
    new_words = []
    for w in words:
        if len(w) <= 3 or random.random() < 0.6:
            new_words.append(w)
            continue
        lower = w.lower()
        # Random spelling transforms
        transform = random.choice(["vowel_drop", "double_char", "swap_adjacent", "truncate"])
        if transform == "vowel_drop":
            # Remove some vowels (SMS style)
            result = re.sub(r"[aeiou]", "", lower)
            if len(result) >= 2:
                new_words.append(result)
            else:
                new_words.append(w)
        elif transform == "double_char":
            # Double a random char
            idx = random.randint(0, len(w) - 1)
            new_words.append(w[:idx] + w[idx] * 2 + w[idx + 1:])
        elif transform == "swap_adjacent":
            # Swap two adjacent characters
            if len(w) >= 4:
                idx = random.randint(1, len(w) - 3)
                new_words.append(w[:idx] + w[idx + 1] + w[idx] + w[idx + 2:])
            else:
                new_words.append(w)
        elif transform == "truncate":
            # Truncate to first few chars
            new_words.append(w[:max(3, len(w) // 2 + 1)])
        else:
            new_words.append(w)
    return " ".join(new_words)


def strategy_dialect_variation(text: str, entry: dict) -> str:
    """Convert to different Malay dialect."""
    target_dialect = random.choice(list(DIALECT_MAPS.keys()))
    dialect_map = DIALECT_MAPS[target_dialect]
    words = text.split()
    new_words = []
    for w in words:
        lower = w.lower()
        if lower in dialect_map:
            new_words.append(dialect_map[lower])
        else:
            new_words.append(w)
    return " ".join(new_words)


# ---------------------------------------------------------------------------
# Main augmentation pipeline
# ---------------------------------------------------------------------------

STRATEGIES = {
    "synonym": strategy_synonym_replacement,
    "shortform": strategy_shortform_variation,
    "codeswitch": strategy_code_switching,
    "slang": strategy_slang_injection,
    "emoji": strategy_emoji_variation,
    "negation": strategy_negation_flip,
    "aspect": strategy_aspect_swap,
    "length": strategy_length_variation,
    "spelling": strategy_spelling_variation,
    "dialect": strategy_dialect_variation,
}


def augment_entry(entry: dict, n_variants: int = 6) -> list[dict]:
    """Generate augmented variants for one labeled example."""
    text = entry["text"]
    variants = []
    seen = {text}

    # Each variant uses 1-3 strategies chained
    attempts = 0
    max_attempts = n_variants * 5

    while len(variants) < n_variants and attempts < max_attempts:
        attempts += 1
        # Pick 1-3 random strategies
        n_strats = random.randint(1, 3)
        chosen_strats = random.sample(list(STRATEGIES.keys()), k=n_strats)

        new_text = text
        current_entry = entry.copy()

        for strat_name in chosen_strats:
            strat_fn = STRATEGIES[strat_name]
            if strat_name == "negation":
                # Negation flip changes label
                new_text = strat_fn(new_text, current_entry)
                # Flip sentiment
                old_sent = current_entry.get("sentiment", "neutral")
                if old_sent == "positive":
                    current_entry = {**current_entry, "sentiment": "negative", "emotion": "angry"}
                elif old_sent == "negative":
                    current_entry = {**current_entry, "sentiment": "positive", "emotion": "happy"}
                # Flip back if double negation applied
            else:
                new_text = strat_fn(new_text, current_entry)

        # Clean up
        new_text = re.sub(r"\s+", " ", new_text).strip()

        # Skip if too short, identical to original, or already seen
        if len(new_text) < 5 or new_text in seen or new_text == text:
            continue

        seen.add(new_text)

        # Build output entry
        variant_entry = current_entry.copy()
        variant_entry["text"] = new_text
        variant_entry["augmentation_methods"] = chosen_strats
        variant_entry["source_original"] = text[:80]  # Track provenance
        variant_entry["source_type"] = entry.get("source_type", "augmented")

        # Update dialect if dialect variation was used
        if "dialect" in chosen_strats:
            target_d = random.choice(list(DIALECT_MAPS.keys()))
            variant_entry["dialect"] = target_d

        # Update code_switch flag
        if "codeswitch" in chosen_strats:
            variant_entry["is_code_switch"] = True

        variants.append(variant_entry)

    return variants


def main():
    parser = argparse.ArgumentParser(description="Augment Manglish labeled data")
    parser.add_argument("--variants", type=int, default=6,
                        help="Number of variants per example (default: 6)")
    parser.add_argument("--output", type=str,
                        default=str(DATASETS_DIR / "manglish_labeled_v3.jsonl"),
                        help="Output path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load existing data
    v1_path = DATASETS_DIR / "manglish_labeled.jsonl"
    v2_path = DATASETS_DIR / "manglish_labeled_v2.jsonl"

    all_originals = []
    if v1_path.exists():
        v1 = load_jsonl(v1_path)
        all_originals.extend(v1)
        print(f"Loaded v1: {len(v1)} examples")
    if v2_path.exists():
        v2 = load_jsonl(v2_path)
        all_originals.extend(v2)
        print(f"Loaded v2: {len(v2)} examples")

    if not all_originals:
        print("ERROR: No input data found!")
        return

    print(f"\nTotal originals: {len(all_originals)}")
    print(f"Target variants per example: {args.variants}")
    print(f"Expected output: ~{len(all_originals) * args.variants} augmented + {len(all_originals)} originals\n")

    # Augment
    all_augmented = []
    method_counts = Counter()
    skipped = 0

    for i, entry in enumerate(all_originals):
        variants = augment_entry(entry, n_variants=args.variants)
        if not variants:
            skipped += 1
            continue
        all_augmented.extend(variants)
        for v in variants:
            for m in v.get("augmentation_methods", []):
                method_counts[m] += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_originals)} examples ({len(all_augmented)} variants so far)")

    # Combine originals + augmented
    combined = all_originals + all_augmented

    # Deduplicate by text
    seen_texts = set()
    unique = []
    dupes = 0
    for item in combined:
        t = item.get("text", "").strip()
        if t and t not in seen_texts:
            seen_texts.add(t)
            unique.append(item)
        else:
            dupes += 1

    # Save
    output_path = Path(args.output)
    save_jsonl(unique, output_path)

    # Print statistics
    print("\n" + "=" * 60)
    print("AUGMENTATION STATISTICS")
    print("=" * 60)
    print(f"\nOriginal examples:    {len(all_originals)}")
    print(f"Augmented variants:   {len(all_augmented)}")
    print(f"Duplicates removed:   {dupes}")
    print(f"Skipped (no variant): {skipped}")
    print(f"Final dataset size:   {len(unique)}")
    print(f"Output file:          {output_path}")

    # Label distributions
    print("\n" + "-" * 60)
    print("LABEL DISTRIBUTIONS (final dataset)")
    print("-" * 60)

    for field in ["sentiment", "emotion", "intent", "topic", "dialect", "language"]:
        values = [item.get(field) for item in unique if item.get(field)]
        if values:
            counter = Counter(values)
            print(f"\n{field.upper()} ({len(values)} labeled):")
            for label, count in counter.most_common():
                pct = count / len(values) * 100
                bar = "#" * int(pct / 2)
                print(f"  {label:20s} {count:5d} ({pct:5.1f}%) {bar}")

    # Augmentation method usage
    print("\n" + "-" * 60)
    print("AUGMENTATION METHODS USED")
    print("-" * 60)
    for method, count in method_counts.most_common():
        print(f"  {method:20s} {count:5d} times")

    print("\nDone!")


if __name__ == "__main__":
    main()
