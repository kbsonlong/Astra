#!/usr/bin/env python3
"""Reliable SenseVoice model downloader with progress + sha256 validation.
Uses `requests` (already brought in via transformers dep) instead of curl,
so it works inside the sandbox shell quoting mess.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

try:
    import requests
except Exception as exc:
    print("FATAL: no requests:", exc)
    sys.exit(2)

REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
REV = "2365baeacb507f821a0c8120fcee3d484dba7a07"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REV}"
OUT_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    (
        "model.int8.onnx",
        239_233_841,
        "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
    ),
    (
        "tokens.txt",
        315_894,
        "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
    ),
]


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, expected_size: int, expected_sha: str, *, retries: int = 3) -> Path:
    out = OUT_DIR / name
    url = f"{BASE_URL}/{name}?download=true"
    for attempt in range(1, retries + 1):
        print(f"\nDownloading {name} (attempt {attempt}/{retries}) ...")
        resume_from = out.stat().st_size if out.exists() and out.stat().st_size < expected_size else 0
        if resume_from and resume_from == expected_size:
            current_sha = sha256_file(out)
            if current_sha == expected_sha:
                print(f"  {name} already complete and hash matches. SKIP.")
                return out
            print(f"  resume byte {resume_from}...")
            headers = {"Range": f"bytes={resume_from}-"}
            mode = "ab"
        else:
            if resume_from:
                print(f"  bad existing size {resume_from}, restart from 0")
                out.unlink()
            resume_from = 0
            headers = {}
            mode = "wb"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    print(f"  HTTP {r.status_code}, retry")
                    time.sleep(2)
                    continue
                total = resume_from + int(r.headers.get("Content-Length", "0"))
                if total and total != expected_size:
                    # Just informational; HF may send varying sizes on redirections
                    pass
                start = time.perf_counter()
                so_far = resume_from
                with out.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        so_far += len(chunk)
                        pct = so_far / expected_size * 100
                        if so_far % (5 * 1024 * 1024) < len(chunk):
                            elapsed = max(0.001, time.perf_counter() - start)
                            mbps = (so_far - resume_from) / (1024 * 1024) / elapsed
                            print(f"  {so_far // (1024*1024):>5}MB / {expected_size // (1024*1024)}MB  "
                                  f"({pct:5.1f}%) @ {mbps:.2f} MB/s")
            print(f"  done transfer, verifying sha256 ...")
            actual = sha256_file(out)
            if actual != expected_sha:
                print(f"  sha256 mismatch: expected {expected_sha} got {actual}")
                out.unlink()
                continue
            print(f"  sha256 OK, size={out.stat().st_size} bytes")
            return out
        except requests.RequestException as exc:
            print(f"  network error: {exc}, retry after 2s")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {name} after {retries} retries")


def main() -> int:
    for name, sz, sha in FILES:
        try:
            path = download(name, sz, sha)
        except Exception as exc:
            print(f"FATAL {name}: {exc}")
            return 3
        print(f"  => {path} ready")
    print("\nAll models downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
