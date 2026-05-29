from huggingface_hub import HfApi
import os

api = HfApi()
token = "YOUR_HF_TOKEN_HERE"
repo_id = "vexccz/malaysian-manglish-nlp-demo"
repo_type = "space"

# Upload demo files
files = {
    "demo/app.py": "app.py",
    "demo/requirements.txt": "requirements.txt",
    "demo/README.md": "README.md",
    "demo/Dockerfile": "Dockerfile",
}

for local_path, remote_path in files.items():
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Upload {remote_path}"
        )
        print(f"Uploaded: {local_path} -> {remote_path}")
    else:
        print(f"SKIP (not found): {local_path}")

print(f"\nSpace URL: https://huggingface.co/spaces/{repo_id}")
print("Note: Space will build and start in a few minutes.")
