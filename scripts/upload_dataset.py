from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

# Create dataset repo under vexccz namespace
try:
    create_repo("vexccz/manglish-nlp-dataset", repo_type="dataset", token=token, exist_ok=True)
    print("Created/accessed dataset repo")
except Exception as e:
    print(f"Repo may already exist: {e}")

# Upload dataset files
files_to_upload = [
    "datasets/manglish_labeled.jsonl",
    "datasets/manglish_labeled_v2.jsonl",
    "datasets/manglish_labeled_v3.jsonl",
    "datasets/manglish_full.jsonl",
    "datasets/manglish_full_train.jsonl",
    "datasets/manglish_full_test.jsonl",
    "datasets/README.md",
]

for f in files_to_upload:
    if os.path.exists(f):
        try:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=os.path.basename(f),
                repo_id="vexccz/manglish-nlp-dataset",
                repo_type="dataset",
                token=token,
                commit_message=f"Upload {os.path.basename(f)}"
            )
            print(f"Uploaded: {f}")
        except Exception as e:
            print(f"FAILED {f}: {e}")
    else:
        print(f"NOT FOUND: {f}")

# Upload dataset card
if os.path.exists("datasets/huggingface_card.md"):
    try:
        api.upload_file(
            path_or_fileobj="datasets/huggingface_card.md",
            path_in_repo="README.md",
            repo_id="vexccz/manglish-nlp-dataset",
            repo_type="dataset",
            token=token,
            commit_message="Upload dataset card"
        )
        print("Uploaded dataset card as README.md")
    except Exception as e:
        print(f"FAILED dataset card: {e}")
else:
    print("No huggingface_card.md found")

print("Dataset upload complete!")
print("View at: https://huggingface.co/datasets/vexccz/manglish-nlp-dataset")
