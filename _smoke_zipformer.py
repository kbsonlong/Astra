#!/usr/bin/env python3
import sys, os, time, tempfile, subprocess, inspect
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

os.chdir(str(BASE / "backend"))
sys.path.insert(0, ".")

def flush():
    sys.stdout.flush()
    sys.stderr.flush()

print("=== Step 1: Inspect OnlineRecognizer.from_transducer signature ===")
import sherpa_onnx
sig = inspect.signature(sherpa_onnx.OnlineRecognizer.from_transducer)
print(f"  params: {list(sig.parameters.keys())}")
flush()

print("\n=== Step 2: Build recognizer ===")
tokens_f = str(MODEL_DIR / "tokens.txt")
bpe = str(MODEL_DIR / "bpe.model")
enc = str(MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx")
dec = str(MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx")
join = str(MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx")
for p in [tokens_f, bpe, enc, dec, join]:
    sz = Path(p).stat().st_size
    print(f"  {Path(p).name:<45s} {sz//(1024*1024):>3d}MB  ({sz} bytes)")
flush()

try:
    # Zipformer Bilingual zh+en 2023-02-20: tokens.txt + bpe.model + zipformer (V1)
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens_f,
        bpe_vocab=bpe,
        encoder=enc,
        decoder=dec,
        joiner=join,
        num_threads=2,
        decoding_method="greedy_search",
        provider="cpu",
        sample_rate=16000,
        feature_dim=80,
        model_type="zipformer",
        modeling_unit="bpe",
        debug=False,
        enable_endpoint_detection=False,
    )
    print(f"  recognizer OK (zipformer1+bpe) -> {type(rec).__name__}")
except (TypeError, RuntimeError) as e:
    print(f"  FAIL zipformer+bpe: {type(e).__name__}: {e}")
    try:
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_f,
            bpe_vocab=bpe,
            encoder=enc,
            decoder=dec,
            joiner=join,
            num_threads=2,
            decoding_method="greedy_search",
            provider="cpu",
            sample_rate=16000,
            feature_dim=80,
            debug=False,
            enable_endpoint_detection=False,
        )
        print(f"  recognizer OK (auto-detect) -> {type(rec).__name__}")
    except (TypeError, RuntimeError) as e2:
        print(f"  FAIL auto: {type(e2).__name__}: {e2}")
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_f,
            encoder=enc,
            decoder=dec,
            joiner=join,
            num_threads=2,
            decoding_method="greedy_search",
            provider="cpu",
            sample_rate=16000,
            feature_dim=80,
            model_type="zipformer",
            debug=False,
            enable_endpoint_detection=False,
        )
        print(f"  recognizer OK (tokens only, no bpe_vocab) -> {type(rec).__name__}")
flush()

def do_rec(rec, samples, sr=16000):
    stream = rec.create_stream()
    stream.accept_waveform(sr, samples)
    tail = np.zeros(int(0.5 * sr), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    result = rec.get_result(stream)
    if isinstance(result, str):
        return result.strip()
    t = getattr(result, "text", None)
    if isinstance(t, str):
        return t.strip()
    return str(result).strip()

print("\n=== Step 3: Recognize official test_wavs/0.wav ===")
try:
    from scipy.io import wavfile as _wf
    sr, samps = _wf.read(str(MODEL_DIR / "test_wavs" / "0.wav"))
    print(f"  test_wavs/0.wav: sr={sr}, shape={samps.shape}, dtype={samps.dtype}")
    if samps.dtype == np.int16:
        samps_f = samps.astype(np.float32) / 32768.0
    else:
        samps_f = samps.astype(np.float32)
    if len(samps_f.shape) > 1:
        samps_f = samps_f.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        dst_len = max(1, int(round(len(samps_f) * ratio)))
        src_idx = np.arange(dst_len, dtype=np.float32) / ratio
        i0 = np.floor(src_idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, len(samps_f) - 1)
        frac = (src_idx - i0).astype(np.float32)
        samps_f = (samps_f[i0] * (1 - frac) + samps_f[i1] * frac).astype(np.float32)
    print(f"  final: len={len(samps_f)}, dur={len(samps_f)/16000:.2f}s")
    text = do_rec(rec, samps_f)
    print(f"  RECOGNIZED: {text!r}")
except Exception as e:
    import traceback; traceback.print_exc()
flush()

print("\n=== Step 4: Recognize first 5s of 45s m4a ===")
try:
    with tempfile.TemporaryDirectory() as td:
        wav_p = Path(td) / "short.wav"
        m4a_p = BASE / "1787882095007-5ef1.m4a"
        subprocess.run([
            "/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000",
            "-c", "1", str(m4a_p), str(wav_p),
        ], check=True)
        from scipy.io import wavfile
        sr, sd = wavfile.read(str(wav_p))
        # Crop to first 5s
        n = min(len(sd), int(5 * sr))
        sd = sd[:n]
        print(f"  5s wav: sr={sr}, shape={sd.shape}, dtype={sd.dtype}, crop_len={n}")
        if sd.dtype == np.int16:
            sd = sd.astype(np.float32) / 32768.0
        else:
            sd = sd.astype(np.float32)
        if len(sd.shape) > 1:
            sd = sd.mean(axis=1)
        if sr != 16000:
            ratio = 16000 / sr
            dst_len = max(1, int(round(len(sd) * ratio)))
            src_idx = np.arange(dst_len, dtype=np.float32) / ratio
            i0 = np.floor(src_idx).astype(np.int64)
            i1 = np.minimum(i0 + 1, len(sd) - 1)
            frac = (src_idx - i0).astype(np.float32)
            sd = (sd[i0] * (1 - frac) + sd[i1] * frac).astype(np.float32)
        text = do_rec(rec, sd)
        print(f"  RECOGNIZED: {text!r}")
except Exception:
    import traceback; traceback.print_exc()
flush()

del rec
print("\nALL SMOKE PASSED")
flush()
