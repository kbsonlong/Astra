#!/usr/bin/env python3
import os
import sys
import subprocess

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("PRE-FLIGHT: dependencies + m4a info + model cache")
print("=" * 60)

ok = {}
for mod, pip_name in (
    ("mlx_audio", "mlx-audio"),
    ("sherpa_onnx", "sherpa-onnx"),
    ("numpy", "numpy"),
):
    try:
        __import__(mod)
        ok[pip_name] = "OK"
    except Exception as exc:
        ok[pip_name] = f"FAIL: {type(exc).__name__}: {exc}"
print("--- Dependencies ---")
for k, v in ok.items():
    print(f"  {k:15s}: {v}")

print("\n--- M4A recordings ---")
m4as = sorted(p for p in os.listdir(PROJECT_ROOT) if p.endswith(".m4a"))
for f in m4as:
    full = os.path.join(PROJECT_ROOT, f)
    kb = os.path.getsize(full) // 1024
    dur = "unknown"
    try:
        out = subprocess.run(["/usr/bin/afinfo", full], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "estimated duration" in line.lower():
                dur = line.strip()
                break
    except Exception as exc:
        dur = f"afinfo err: {exc}"
    print(f"  {f}: {kb} KB | {dur}")

print("\n--- Model cache checks ---")
# MLX / HF hub
hf_dir = os.path.expanduser("~/.cache/huggingface/hub")
qwen_refs = []
if os.path.isdir(hf_dir):
    qwen_refs = sorted(d for d in os.listdir(hf_dir) if "qwen3-asr" in d.lower())
print(f"  HuggingFace hub: {hf_dir}")
print(f"  Qwen3-ASR cached dirs: {qwen_refs}")
for d in qwen_refs:
    snap = os.path.join(hf_dir, d, "snapshots")
    if os.path.isdir(snap):
        for sub in os.listdir(snap):
            sdir = os.path.join(snap, sub)
            if os.path.isdir(sdir):
                files = os.listdir(sdir)
                mb = sum(os.path.getsize(os.path.join(sdir, f)) for f in files if os.path.isfile(os.path.join(sdir, f))) // (1024*1024)
                print(f"    - {d}/snapshots/{sub}: {len(files)} files, {mb} MB")

# Sherpa SenseVoice
candidates = [
    os.path.join(PROJECT_ROOT, "models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"),
    os.path.join(BACKEND, "models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"),
]
print(f"\n  Sherpa SenseVoice model candidates:")
for c in candidates:
    if os.path.isdir(c):
        files = os.listdir(c)
        model_size_mb = 0
        for f in files:
            if f.endswith(".onnx"):
                model_size_mb = max(model_size_mb, os.path.getsize(os.path.join(c, f)) // (1024*1024))
        print(f"    FOUND {c}: {files}, ONNX={model_size_mb}MB")
    else:
        print(f"    MISSING {c}")

print("\nPRE-FLIGHT DONE")
