"""Download Qwen3-ASR-0.6B-4bit for mlx-audio using huggingface_hub.snapshot_download.
Shows progress and resumes existing blobs.
"""
from huggingface_hub import snapshot_download
import sys, os, shutil
repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(f"Downloading {repo} -> {cache_dir}")
try:
    path = snapshot_download(
        repo_id=repo,
        cache_dir=cache_dir,
        resume_download=True,
        max_workers=4,
    )
    print(f"\nDONE -> {path}")
    print("Files:")
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            print(f"  {os.path.relpath(fp, path)}  {os.path.getsize(fp)//1024}KB")
except Exception as exc:
    print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(3)
