"""
Merge Manglish NLP dataset files (v1 + v2).

Loads manglish_labeled.jsonl and manglish_labeled_v2.jsonl,
deduplicates by text field, and outputs a merged file.

Usage:
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --output datasets/manglish_merged.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: Path):
    """Save a list of dicts to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_datasets(output_path: Path):
    """Merge v1 and v2 datasets, deduplicate, and save."""
    # Find dataset files
    v1_path = DATASETS_DIR / "manglish_labeled.jsonl"
    v2_path = DATASETS_DIR / "manglish_labeled_v2.jsonl"

    all_data = []
    sources = {}

    # Load v1
    if v1_path.exists():
        v1_data = load_jsonl(v1_path)
        all_data.extend(v1_data)
        sources["v1 (manglish_labeled.jsonl)"] = len(v1_data)
        print(f"Loaded v1: {len(v1_data)} examples")
    else:
        print(f"Warning: {v1_path} not found, skipping")

    # Load v2
    if v2_path.exists():
        v2_data = load_jsonl(v2_path)
        all_data.extend(v2_data)
        sources["v2 (manglish_labeled_v2.jsonl)"] = len(v2_data)
        print(f"Loaded v2: {len(v2_data)} examples")
    else:
        print(f"Warning: {v2_path} not found, skipping")

    # Also check for any other JSONL files
    for jsonl_file in DATASETS_DIR.glob("manglish_labeled*.jsonl"):
        if jsonl_file not in (v1_path, v2_path, output_path):
            extra_data = load_jsonl(jsonl_file)
            all_data.extend(extra_data)
            sources[f"extra ({jsonl_file.name})"] = len(extra_data)
            print(f"Loaded extra: {jsonl_file.name} ({len(extra_data)} examples)")

    if not all_data:
        print("Error: No data files found!")
        return

    # Deduplicate by text
    seen_texts = set()
    unique_data = []
    duplicates = 0

    for item in all_data:
        text = item.get("text", "").strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_data.append(item)
        else:
            duplicates += 1

    # Save merged file
    save_jsonl(unique_data, output_path)

    # Print stats
    print("\n" + "=" * 50)
    print("MERGE STATISTICS")
    print("=" * 50)
    print(f"\nSources:")
    for source, count in sources.items():
        print(f"  {source}: {count}")
    print(f"\nTotal loaded:     {len(all_data)}")
    print(f"Duplicates found: {duplicates}")
    print(f"Unique examples:  {len(unique_data)}")
    print(f"\nOutput: {output_path}")

    # Label distribution stats
    print("\n" + "-" * 50)
    print("LABEL DISTRIBUTIONS")
    print("-" * 50)

    label_fields = ["sentiment", "emotion", "intent", "topic", "dialect", "language"]

    for field in label_fields:
        values = [item.get(field) for item in unique_data if item.get(field)]
        if values:
            counter = Counter(values)
            print(f"\n{field.upper()} ({len(values)} labeled):")
            for label, count in counter.most_common():
                pct = count / len(values) * 100
                print(f"  {label:20s} {count:4d} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Manglish NLP dataset files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASETS_DIR / "manglish_merged.jsonl"),
        help="Output path for merged dataset (default: datasets/manglish_merged.jsonl)",
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    merge_datasets(output_path)


if __name__ == "__main__":
    main()
