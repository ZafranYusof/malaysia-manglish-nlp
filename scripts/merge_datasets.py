"""
Merge auto-labeled real Malaysian texts with existing augmented dataset.

Combines:
  - datasets/real_malaysian_labeled.jsonl (auto-labeled from label_data.py)
  - datasets/manglish_augmented.jsonl (existing augmented training data)

Features:
  - Exact + near-duplicate deduplication
  - Preserves all label fields from both sources
  - Outputs merged dataset for retraining

Usage:
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --output datasets/custom_merged.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file. Returns empty list if file doesn't exist."""
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
    """Save data to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    t = text.lower().strip()
    # Remove emojis
    emoji_re = re.compile(
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
    t = emoji_re.sub("", t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Remove filler particles for comparison
    particles = {"lah", "la", "wei", "weh", "eh", "kan", "ni", "tu", "bro", "dude", "ar", "lor"}
    words = [w for w in t.split() if w not in particles]
    return " ".join(words)


def deduplicate(items: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate by exact text match + normalized near-duplicate.

    Priority: human-labeled (no auto_labeled flag) > auto-labeled.
    Returns (unique_items, duplicate_count).
    """
    # Sort: human/original first, then auto-labeled
    def priority(item):
        return 1 if item.get("auto_labeled") else 0

    items_sorted = sorted(items, key=priority)

    seen_exact = set()
    seen_norm = {}
    unique = []
    dupes = 0

    for item in items_sorted:
        text = item.get("text", "").strip()
        if not text:
            dupes += 1
            continue

        # Exact match
        if text in seen_exact:
            dupes += 1
            continue

        # Near-duplicate
        norm = normalize_text(text)
        if norm in seen_norm:
            dupes += 1
            continue

        seen_exact.add(text)
        seen_norm[norm] = True
        unique.append(item)

    return unique, dupes


def merge_datasets(
    real_labeled_path: Path,
    augmented_path: Path,
) -> tuple[list[dict], dict]:
    """Merge two datasets, deduplicate, return result + stats."""
    stats = {}

    # Load
    real_data = load_jsonl(real_labeled_path)
    aug_data = load_jsonl(augmented_path)

    stats["real_labeled"] = len(real_data)
    stats["augmented"] = len(aug_data)
    stats["total_loaded"] = len(real_data) + len(aug_data)

    print(f"  real_malaysian_labeled.jsonl:  {len(real_data)} examples")
    print(f"  manglish_augmented.jsonl:      {len(aug_data)} examples")
    print(f"  Total loaded:                  {stats['total_loaded']}")

    # Combine
    combined = real_data + aug_data

    # Deduplicate
    unique, dupes = deduplicate(combined)
    stats["duplicates_removed"] = dupes
    stats["final_count"] = len(unique)

    print(f"\n  Duplicates removed: {dupes}")
    print(f"  Final count:        {len(unique)}")

    return unique, stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge auto-labeled real data with augmented dataset"
    )
    parser.add_argument(
        "--real", type=str,
        default=str(DATASETS_DIR / "real_malaysian_labeled.jsonl"),
        help="Auto-labeled real data (default: datasets/real_malaysian_labeled.jsonl)",
    )
    parser.add_argument(
        "--augmented", type=str,
        default=str(DATASETS_DIR / "manglish_augmented.jsonl"),
        help="Augmented dataset (default: datasets/manglish_augmented.jsonl)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DATASETS_DIR / "manglish_merged.jsonl"),
        help="Output merged dataset (default: datasets/manglish_merged.jsonl)",
    )
    args = parser.parse_args()

    real_path = Path(args.real)
    aug_path = Path(args.augmented)
    output_path = Path(args.output)

    # Check inputs exist
    missing = []
    if not real_path.exists():
        missing.append(str(real_path))
    if not aug_path.exists():
        missing.append(str(aug_path))

    if missing:
        print("ERROR: Missing input files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun label_data.py first to generate real_malaysian_labeled.jsonl")
        return

    print("Merging datasets...")
    print()

    merged, stats = merge_datasets(real_path, aug_path)

    # Save
    save_jsonl(merged, output_path)

    # Final summary
    print(f"\n{'=' * 50}")
    print("MERGE COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Real labeled:     {stats['real_labeled']}")
    print(f"  Augmented:        {stats['augmented']}")
    print(f"  Duplicates:       -{stats['duplicates_removed']}")
    print(f"  Final count:      {stats['final_count']}")
    print(f"  Output:           {output_path}")

    # Label distribution summary
    if merged:
        print(f"\n{'-' * 50}")
        print("SENTIMENT DISTRIBUTION")
        print(f"{'-' * 50}")
        sentiments = [it.get("sentiment", "unknown") for it in merged]
        for label, count in Counter(sentiments).most_common():
            pct = count / len(sentiments) * 100
            bar = "#" * int(pct / 2)
            print(f"  {label:15s} {count:5d} ({pct:5.1f}%) {bar}")

        # Source breakdown
        auto_count = sum(1 for it in merged if it.get("auto_labeled"))
        human_count = len(merged) - auto_count
        print(f"\nSOURCE:")
        print(f"  Human/augmented:  {human_count}")
        print(f"  Auto-labeled:     {auto_count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
