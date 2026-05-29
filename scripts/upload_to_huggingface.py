"""
Upload Manglish NLP dataset and model to HuggingFace Hub.

Requirements:
    pip install huggingface_hub datasets

Authentication:
    Set HF_TOKEN environment variable or run `huggingface-cli login`

Usage:
    python scripts/upload_to_huggingface.py --dataset
    python scripts/upload_to_huggingface.py --model
    python scripts/upload_to_huggingface.py --all
"""

import os
import sys
import argparse
import json
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
    from datasets import Dataset, DatasetDict
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install huggingface_hub datasets")
    sys.exit(1)


# Configuration
DATASET_REPO = "ZafranYusof/malaysian-manglish-nlp-dataset"
MODEL_REPO = "ZafranYusof/malaysian-manglish-nlp-model"
PROJECT_ROOT = Path(__file__).parent.parent


def get_token():
    """Get HuggingFace token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        print("Set it with: export HF_TOKEN=your_token_here")
        print("Or run: huggingface-cli login")
        sys.exit(1)
    return token


def load_jsonl(filepath):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def upload_dataset(token):
    """Upload dataset to HuggingFace Hub."""
    print(f"Uploading dataset to {DATASET_REPO}...")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=token,
            exist_ok=True,
        )
        print(f"  Repository {DATASET_REPO} ready.")
    except Exception as e:
        print(f"  Warning creating repo: {e}")

    # Load dataset files
    dataset_dir = PROJECT_ROOT / "datasets"
    data_files = list(dataset_dir.glob("manglish_labeled*.jsonl"))

    if not data_files:
        print("  Error: No JSONL files found in datasets/")
        return False

    # Merge all JSONL files
    all_data = []
    for f in data_files:
        print(f"  Loading {f.name}...")
        all_data.extend(load_jsonl(f))

    # Deduplicate by text
    seen_texts = set()
    unique_data = []
    for item in all_data:
        text = item.get("text", "")
        if text not in seen_texts:
            seen_texts.add(text)
            unique_data.append(item)

    print(f"  Total examples: {len(all_data)}, unique: {len(unique_data)}")

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(unique_data)

    # Split into train/test (80/20)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict(
        {"train": split_dataset["train"], "test": split_dataset["test"]}
    )

    print(f"  Train: {len(dataset_dict['train'])}, Test: {len(dataset_dict['test'])}")

    # Push to hub
    dataset_dict.push_to_hub(DATASET_REPO, token=token)
    print(f"  Dataset uploaded to https://huggingface.co/datasets/{DATASET_REPO}")

    # Upload README (dataset card)
    readme_path = dataset_dir / "huggingface_card.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=token,
        )
        print("  Dataset card (README.md) uploaded.")

    return True


def upload_model(token):
    """Upload model to HuggingFace Hub."""
    print(f"Uploading model to {MODEL_REPO}...")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=MODEL_REPO,
            repo_type="model",
            token=token,
            exist_ok=True,
        )
        print(f"  Repository {MODEL_REPO} ready.")
    except Exception as e:
        print(f"  Warning creating repo: {e}")

    # Find model directory
    model_dir = PROJECT_ROOT / "models"
    if not model_dir.exists():
        # Try alternative locations
        model_dir = PROJECT_ROOT / "malaysian_manglish_nlp" / "models"

    if not model_dir.exists():
        print(f"  Error: Model directory not found at {model_dir}")
        print("  Train the model first or specify the correct path.")
        return False

    # Upload all model files
    model_files = list(model_dir.rglob("*"))
    model_files = [f for f in model_files if f.is_file()]

    if not model_files:
        print("  Error: No model files found.")
        return False

    print(f"  Uploading {len(model_files)} files...")

    for filepath in model_files:
        relative_path = filepath.relative_to(model_dir)
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=str(relative_path).replace("\\", "/"),
            repo_id=MODEL_REPO,
            repo_type="model",
            token=token,
        )
        print(f"    Uploaded: {relative_path}")

    # Upload model card
    model_card_path = PROJECT_ROOT / "malaysian_manglish_nlp" / "resources" / "model_card.md"
    if model_card_path.exists():
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
            repo_type="model",
            token=token,
        )
        print("  Model card (README.md) uploaded.")

    print(f"  Model uploaded to https://huggingface.co/models/{MODEL_REPO}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload Manglish NLP dataset and model to HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset", action="store_true", help="Upload dataset only"
    )
    parser.add_argument(
        "--model", action="store_true", help="Upload model only"
    )
    parser.add_argument(
        "--all", action="store_true", help="Upload both dataset and model"
    )

    args = parser.parse_args()

    if not any([args.dataset, args.model, args.all]):
        parser.print_help()
        print("\nSpecify --dataset, --model, or --all")
        sys.exit(1)

    token = get_token()

    success = True

    if args.dataset or args.all:
        if not upload_dataset(token):
            success = False

    if args.model or args.all:
        if not upload_model(token):
            success = False

    if success:
        print("\nDone! All uploads completed successfully.")
    else:
        print("\nSome uploads failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
