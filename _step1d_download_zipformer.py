#!/usr/bin/env python3
"""Download Zipformer Bilingual (zh+en, streaming transducer, ~80MB int8)
from k2-fsa GitHub Release tarball, verify file existence after extraction.

Zipformer bilingual zh+en model catalog entry:
  Name: sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  Release: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  Tarball: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  Required files inside tarball:
    - tokens.txt
    - encoder-epoch-99-avg-1.int8.onnx   (main model, ~67MB)
    - decoder-epoch-99-avg-1.int8.onnx
    - joiner-epoch-99-avg-1.int8.onnx
  Docs: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english
"""
from __future__ import annotations

import sys
import tarfile
import time
from pathlib import Path

try:
    import requests
except Exception as exc:
    print("FATAL: no requests:", exc)
    sys.exit(2)

TARBALL_NAME = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
MODEL_DIR_NAME = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + TARBALL_NAME
# Expected compressed tarball size (approximate, HF mirror may send different sizes,
# we do size + file list check instead of sha because tarball contents are not sha fixed)
APPROX_TARBALL_BYTES = 82_000_000

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "models" / MODEL_DIR_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARBALL_PATH = BASE_DIR / "models" / TARBALL_NAME

REQUIRED_FILES = [
    "tokens.txt",
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.int8.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
]


def download_tarball(retries: int = 3) -> Path:
    for attempt in range(1, retries + 1):
        print(f"\nDownloading {TARBALL_NAME} (attempt {attempt}/{retries}) ...")
        resume_from = TARBALL_PATH.stat().st_size if TARBALL_PATH.exists() and TARBALL_PATH.stat().st_size < APPROX_TARBALL_BYTES else 0
        if resume_from and resume_from >= APPROX_TARBALL_BYTES:
            print(f"  tarball already exists (~{resume_from // (1024*1024)}MB), skip download")
            return TARBALL_PATH
        if resume_from:
            print(f"  resume byte {resume_from}...")
            headers = {"Range": f"bytes={resume_from}-"}
            mode = "ab"
        else:
            headers = {}
            mode = "wb"
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    print(f"  HTTP {r.status_code}, retry")
                    time.sleep(3)
                    continue
                start = time.perf_counter()
                so_far = resume_from
                with TARBALL_PATH.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        so_far += len(chunk)
                        if so_far % (2 * 1024 * 1024) < len(chunk):
                            elapsed = max(0.001, time.perf_counter() - start)
                            mbps = (so_far - resume_from) / (1024 * 1024) / elapsed
                            pct = so_far / APPROX_TARBALL_BYTES * 100
                            print(f"  {so_far // (1024*1024):>4}MB / ~{APPROX_TARBALL_BYTES // (1024*1024)}MB  "
                                  f"({pct:4.1f}%) @ {mbps:.2f} MB/s")
            print(f"  download complete, size={TARBALL_PATH.stat().st_size // (1024*1024)}MB")
            return TARBALL_PATH
        except requests.RequestException as exc:
            print(f"  network error: {exc}, retry after 3s")
            time.sleep(3)
    raise RuntimeError(f"Failed to download tarball after {retries} retries")


def extract_and_verify(tb_path: Path) -> bool:
    print(f"\nExtracting {TARBALL_NAME} ...")
    # First check if required files already extracted
    all_ready = all((OUT_DIR / f).is_file() and (OUT_DIR / f).stat().st_size > 0 for f in REQUIRED_FILES)
    if all_ready:
        for f in REQUIRED_FILES:
            sz = (OUT_DIR / f).stat().st_size
            print(f"  [SKIP] {f:<45s} {sz // (1024*1024)}MB")
        print("  Already extracted and verified.")
        return True
    with tarfile.open(tb_path, "r:bz2") as tf:
        members = tf.getnames()
        # Some tarballs have a leading top-level dir with MODEL_DIR_NAME; some don't.
        # Detect the common prefix of all required files.
        def find_member(name: str) -> str | None:
            for m in members:
                if m.endswith("/" + name) or m == name:
                    return m
            return None
        found_members = {name: find_member(name) for name in REQUIRED_FILES}
        missing = [n for n, p in found_members.items() if p is None]
        if missing:
            print(f"  FATAL: missing members in tarball: {missing}\n  all members: {members[:30]}")
            return False
        for f, tarpath in found_members.items():
            out_file = OUT_DIR / f
            print(f"  extracting {tarpath} -> models/{MODEL_DIR_NAME}/{f}")
            with tf.extractfile(tarpath) as src, open(out_file, "wb") as dst:
                while True:
                    buf = src.read(1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
    for f in REQUIRED_FILES:
        sz = (OUT_DIR / f).stat().st_size
        print(f"  [OK]   {f:<45s} {sz // (1024*1024)}MB / {sz} bytes")
    return True


def main() -> int:
    try:
        tb = download_tarball()
    except Exception as exc:
        print(f"FATAL download: {exc}")
        return 3
    if not extract_and_verify(tb):
        return 4
    # Optional: delete tarball to save 80MB disk
    # tb.unlink(missing_ok=True)
    print(f"\nZipformer Bilingual model ready at {OUT_DIR}")
    # print Python import test with OnlineRecognizer
    try:
        import sherpa_onnx
        tokens = str(OUT_DIR / "tokens.txt")
        enc = str(OUT_DIR / "encoder-epoch-99-avg-1.int8.onnx")
        dec = str(OUT_DIR / "decoder-epoch-99-avg-1.int8.onnx")
        join = str(OUT_DIR / "joiner-epoch-99-avg-1.int8.onnx")
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens, encoder=enc, decoder=dec, joiner=join,
            num_threads=2, decoding_method="greedy_search", debug=False,
        )
        print(f"  OnlineRecognizer.from_transducer() smoke load OK -> {type(rec).__name__}")
        del rec
    except Exception as exc:
        print(f"  WARNING: smoke load failed (but files still OK): {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
