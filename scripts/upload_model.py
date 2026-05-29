from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

# Create model repo under vexccz namespace
try:
    create_repo("vexccz/manglish-nlp-sentiment", repo_type="model", token=token, exist_ok=True)
    print("Created/accessed model repo")
except Exception as e:
    print(f"Repo may already exist: {e}")

# Upload model card
if os.path.exists("manglish_nlp/resources/model_card.md"):
    try:
        api.upload_file(
            path_or_fileobj="manglish_nlp/resources/model_card.md",
            path_in_repo="README.md",
            repo_id="vexccz/manglish-nlp-sentiment",
            repo_type="model",
            token=token,
            commit_message="Upload model card"
        )
        print("Uploaded model card")
    except Exception as e:
        print(f"FAILED model card: {e}")
else:
    print("No model_card.md found")

# Upload fine-tuned model files
model_dir = "manglish_nlp/resources/manglish_finetuned"
model_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

for f in model_files:
    path = os.path.join(model_dir, f)
    if os.path.exists(path):
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f,
                repo_id="vexccz/manglish-nlp-sentiment",
                repo_type="model",
                token=token,
                commit_message=f"Upload {f}"
            )
            print(f"Uploaded: {f}")
        except Exception as e:
            print(f"FAILED {f}: {e}")
    else:
        print(f"NOT FOUND: {path}")

# Upload Word2Vec model
if os.path.exists("manglish_nlp/resources/word2vec.model"):
    size_mb = os.path.getsize("manglish_nlp/resources/word2vec.model") / (1024*1024)
    if size_mb < 100:
        try:
            api.upload_file(
                path_or_fileobj="manglish_nlp/resources/word2vec.model",
                path_in_repo="word2vec.model",
                repo_id="vexccz/manglish-nlp-sentiment",
                repo_type="model",
                token=token,
                commit_message="Upload Word2Vec model"
            )
            print(f"Uploaded word2vec.model ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"FAILED word2vec.model: {e}")
    else:
        print(f"Skipping word2vec.model ({size_mb:.1f}MB) - too large")
else:
    print("No word2vec.model found")

print("Model upload complete!")
print("View at: https://huggingface.co/vexccz/manglish-nlp-sentiment")
