"""
Merge ALL Manglish NLP datasets into one clean dataset with train/test split.

Sources merged:
  - manglish_labeled.jsonl (v1 original)
  - manglish_labeled_v2.jsonl (v2 original)
  - manglish_labeled_v3.jsonl (augmented)
  - manglish_auto_labeled.jsonl (auto-labeled from scraped data)

Features:
  - Text deduplication (exact + near-duplicate via normalized text)
  - Label conflict resolution (human > auto > augmented priority)
  - Stratified 80/20 train/test split
  - Comprehensive statistics output

Usage:
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --output datasets/manglish_full.jsonl
    python scripts/merge_datasets.py --test-ratio 0.15 --seed 123
"""

import json
import re
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(filepath: Path) -> list[dict]:
    data = []
    if not filepath.exists():
        return data
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
# Deduplication
# ---------------------------------------------------------------------------

def normalize_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    t = text.lower().strip()
    # Remove emojis
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
        "]+", flags=re.UNICODE)
    t = emoji_pattern.sub("", t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Remove common particles for dedup comparison
    particles = {"lah", "la", "wei", "weh", "eh", "kan", "ni", "tu", "bro", "dude"}
    words = [w for w in t.split() if w not in particles]
    return " ".join(words)


def deduplicate(items: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate by exact text + normalized near-duplicate.

    Priority for keeping: human > auto_high_confidence > augmented > needs_review
    """
    # Priority order (lower = higher priority = keep)
    PRIORITY = {
        "human": 0,
        None: 0,  # Original data has no label_source
        "auto_high_confidence": 1,
        "augmented": 2,
        "needs_review": 3,
    }

    def get_priority(item):
        source = item.get("label_source")
        return PRIORITY.get(source, 2)

    # Sort by priority first (keep higher priority items)
    items_sorted = sorted(items, key=get_priority)

    seen_exact = set()
    seen_normalized = {}
    unique = []
    dupes = 0

    for item in items_sorted:
        text = item.get("text", "").strip()
        if not text:
            dupes += 1
            continue

        # Exact dedup
        if text in seen_exact:
            dupes += 1
            continue

        # Near-duplicate via normalized text
        norm = normalize_for_dedup(text)
        if norm in seen_normalized:
            # Keep the higher priority one (already sorted, so first seen wins)
            dupes += 1
            continue

        seen_exact.add(text)
        seen_normalized[norm] = len(unique)
        unique.append(item)

    return unique, dupes


# ---------------------------------------------------------------------------
# Label conflict resolution
# ---------------------------------------------------------------------------

def resolve_conflicts(items: list[dict]) -> list[dict]:
    """Resolve label conflicts when same text has different labels.

    Priority: human > auto_high_confidence > augmented
    For augmented data with flipped labels (negation flip), trust the augmentation.
    """
    text_map = defaultdict(list)
    for item in items:
        text = item.get("text", "").strip()
        text_map[text].append(item)

    resolved = []
    conflicts = 0

    for text, group in text_map.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue

        conflicts += 1
        # Sort by priority and keep best
        PRIORITY = {"human": 0, None: 0, "auto_high_confidence": 1, "augmented": 2}
        group.sort(key=lambda x: PRIORITY.get(x.get("label_source"), 2))
        best = group[0].copy()
        best["conflict_resolved"] = True
        best["conflict_count"] = len(group)
        resolved.append(best)

    return resolved


# ---------------------------------------------------------------------------
# Stratified train/test split
# ---------------------------------------------------------------------------

def stratified_split(items: list[dict], test_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratified split by sentiment (primary) and topic (secondary).

    Ensures proportional representation of labels in both splits.
    """
    random.seed(seed)

    # Group by (sentiment, topic) strata
    strata = defaultdict(list)
    for item in items:
        sentiment = item.get("sentiment", "neutral")
        topic = item.get("topic", "daily_life")
        strata[(sentiment, topic)].append(item)

    train = []
    test = []

    for key, group in strata.items():
        random.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio))
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    # Shuffle final sets
    random.shuffle(train)
    random.shuffle(test)

    return train, test


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(data: list[dict], label: str = ""):
    """Print comprehensive dataset statistics."""
    prefix = f" ({label})" if label else ""
    print(f"\n{'=' * 60}")
    print(f"DATASET STATISTICS{prefix}")
    print(f"{'=' * 60}")
    print(f"\nTotal examples: {len(data)}")

    # Label distributions
    for field in ["sentiment", "emotion", "intent", "topic", "dialect", "language"]:
        values = [item.get(field) for item in data if item.get(field)]
        if values:
            counter = Counter(values)
            print(f"\n{field.upper()} ({len(values)} labeled):")
            for lbl, count in counter.most_common():
                pct = count / len(values) * 100
                bar = "#" * int(pct / 2)
                print(f"  {lbl:20s} {count:5d} ({pct:5.1f}%) {bar}")

    # Source breakdown
    sources = Counter()
    for item in data:
        src = item.get("label_source", "original")
        sources[src] += 1
    print(f"\nSOURCE BREAKDOWN:")
    for src, count in sources.most_common():
        print(f"  {src:25s} {count:5d}")

    # Code-switch stats
    cs_count = sum(1 for item in data if item.get("is_code_switch"))
    print(f"\nCode-switched: {cs_count} ({cs_count/max(len(data),1)*100:.1f}%)")

    # Text length stats
    lengths = [len(item.get("text", "")) for item in data]
    if lengths:
        print(f"\nTEXT LENGTH:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Avg: {sum(lengths)//len(lengths)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge ALL Manglish NLP datasets into one clean dataset"
    )
    parser.add_argument("--output", type=str,
                        default=str(DATASETS_DIR / "manglish_full.jsonl"),
                        help="Output path for merged dataset")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split")
    args = parser.parse_args()

    output_path = Path(args.output)
    train_path = output_path.with_name(output_path.stem + "_train" + output_path.suffix)
    test_path = output_path.with_name(output_path.stem + "_test" + output_path.suffix)

    # Define all input sources with priority
    sources = {
        "v1 (original)": DATASETS_DIR / "manglish_labeled.jsonl",
        "v2 (original)": DATASETS_DIR / "manglish_labeled_v2.jsonl",
        "v3 (augmented)": DATASETS_DIR / "manglish_labeled_v3.jsonl",
        "auto-labeled": DATASETS_DIR / "manglish_auto_labeled.jsonl",
    }

    all_data = []
    source_counts = {}

    print("Loading datasets...")
    for name, path in sources.items():
        data = load_jsonl(path)
        if data:
            source_counts[name] = len(data)
            print(f"  {name:25s} {len(data):5d} examples from {path.name}")
            all_data.extend(data)
        else:
            print(f"  {name:25s} (not found or empty: {path.name})")

    if not all_data:
        print("\nERROR: No data found in any source!")
        return

    print(f"\nTotal loaded: {len(all_data)} examples")

    # Step 1: Resolve conflicts
    print("\nResolving label conflicts...")
    resolved = resolve_conflicts(all_data)
    print(f"  After conflict resolution: {len(resolved)} (resolved {len(all_data) - len(resolved)} conflicts)")

    # Step 2: Deduplicate
    print("\nDeduplicating...")
    unique, dupes = deduplicate(resolved)
    print(f"  After dedup: {len(unique)} (removed {dupes} duplicates)")

    # Step 3: Filter out very short texts
    before_filter = len(unique)
    unique = [item for item in unique if len(item.get("text", "")) >= 5]
    print(f"  After length filter: {len(unique)} (removed {before_filter - len(unique)} short texts)")

    # Step 4: Stratified train/test split
    print(f"\nSplitting (test_ratio={args.test_ratio}, seed={args.seed})...")
    train, test = stratified_split(unique, test_ratio=args.test_ratio, seed=args.seed)
    print(f"  Train: {len(train)}")
    print(f"  Test:  {len(test)}")

    # Step 5: Save
    save_jsonl(unique, output_path)
    save_jsonl(train, train_path)
    save_jsonl(test, test_path)

    print(f"\nOutput files:")
    print(f"  Full:  {output_path}")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")

    # Print stats
    print_stats(unique, "MERGED FULL")
    print_stats(train, "TRAIN")
    print_stats(test, "TEST")

    # Verify split quality
    print(f"\n{'=' * 60}")
    print("SPLIT QUALITY CHECK")
    print(f"{'=' * 60}")
    train_sent = Counter(item.get("sentiment") for item in train)
    test_sent = Counter(item.get("sentiment") for item in test)
    print(f"\nSentiment distribution comparison:")
    for label in ["positive", "negative", "neutral"]:
        tr_pct = train_sent.get(label, 0) / max(len(train), 1) * 100
        te_pct = test_sent.get(label, 0) / max(len(test), 1) * 100
        print(f"  {label:12s}  train: {tr_pct:5.1f}%  test: {te_pct:5.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
