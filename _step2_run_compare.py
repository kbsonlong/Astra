#!/usr/bin/env python3
"""
Astra ASR 对比测试：
  A. mlx-audio ==> mlx-community/Qwen3-ASR-0.6B-4bit
  B. sherpa-onnx ==> SenseVoice int8 (csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17)

对项目根目录 3 个 *.m4a 进行识别，对比：
  - 延迟（decode time, RTF = decode / audio duration）
  - 字符级编辑距离 (CER-like，因无 ground truth 用两个结果互为参考)
  - 长音频截取前 N 秒 (避免超长测试)

用法：
  .venv/bin/python3 _step2_run_compare.py
      [--short-crop 44] [--long-crop 90] [--engines mlx,sherpa]
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# m4a -> wav (macOS built-in afconvert, no ffmpeg dependency required)
# ---------------------------------------------------------------------------


def m4a_to_wav(m4a: Path, *, crop_seconds: float | None, tmpdir: Path) -> tuple[Path, float]:
    """Return (wav_path, audio_duration_seconds).

    macOS ``afconvert`` writes WAVE_FORMAT_EXTENSIBLE (tag 65534) which the
    stdlib ``wave`` module rejects, so we read it with ``scipy.io.wavfile``
    (which tolerates the unknown chunks) and re-write in canonical PCM16
    mono format.
    """
    import warnings
    import numpy as np
    from scipy.io import wavfile

    full_wav = tmpdir / (m4a.stem + ".full.wav")
    args = [
        "/usr/bin/afconvert",
        "-f",
        "WAVE",
        "-d",
        "LEI16@16000",
        "-c",
        "1",
        str(m4a),
        str(full_wav),
    ]
    subprocess.run(args, check=True, capture_output=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(str(full_wav))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.floating):
        # Sherpa/Qwen both expect int16 wav bytes; convert float [-1,1] to int16
        data = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
        data = (data * 32767.0).astype(np.int16)
    elif not np.issubdtype(data.dtype, np.int16):
        data = np.asarray(data, dtype=np.int16)

    if crop_seconds is not None:
        n = max(1, int(crop_seconds * sr))
        data = data[:n]
    wav = tmpdir / (m4a.stem + ".wav")
    wavfile.write(str(wav), sr, data.astype(np.int16))
    return wav, data.shape[0] / sr


# ---------------------------------------------------------------------------
# Wrappers around Astra's ASR clients (with timing + mem deltas)
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    engine: str
    wav: str
    duration_s: float
    load_s: float
    first_inference_s: float | None
    total_transcribe_s: float
    rtf: float
    peak_mem_mb: int
    text_chars: int
    text: str
    error: str | None = None


def _peak_rss_mb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))


def _read_wav_bytes(p: Path) -> bytes:
    return p.read_bytes()


async def _load_and_run(
    engine_name: str,
    build_client,
    wav: Path,
    duration_s: float,
    *,
    label: str,
) -> RunResult:
    print(f"\n[{label}] building client for {engine_name} ...")
    t0 = time.perf_counter()
    pre_mem = _peak_rss_mb()
    client = build_client()
    # warm up is_ready once (triggers model load on most impls)
    for _ in range(300):
        if client.is_ready():
            break
        await asyncio.sleep(1.0)
    else:
        raise RuntimeError(f"{engine_name} never became ready")
    load_s = time.perf_counter() - t0
    mem_after_load = _peak_rss_mb()

    # first inference
    t1 = time.perf_counter()
    try:
        text_first = await client.transcribe(_read_wav_bytes(wav), wav.name)
        first_inference_s = time.perf_counter() - t1
    except Exception as exc:
        return RunResult(
            engine=engine_name,
            wav=str(wav.name),
            duration_s=duration_s,
            load_s=load_s,
            first_inference_s=None,
            total_transcribe_s=0.0,
            rtf=0.0,
            peak_mem_mb=mem_after_load,
            text_chars=0,
            text="",
            error=f"first_inference FAIL {type(exc).__name__}: {exc}",
        )

    # run a second time so cold-start file cache is warm, then average timing
    t2 = time.perf_counter()
    text = await client.transcribe(_read_wav_bytes(wav), wav.name)
    warm = time.perf_counter() - t2
    total = (first_inference_s + warm) / 2.0
    peak_mb = max(mem_after_load, _peak_rss_mb())
    rtf = total / duration_s if duration_s > 0 else float("nan")
    print(
        f"[{label}] {engine_name}: load={load_s:.1f}s, inf(cold+warm)/2={total:.2f}s "
        f"(audio {duration_s:.1f}s => RTF {rtf:.3f}); memΔ={peak_mb - pre_mem}MB "
        f"(peak {peak_mb}MB); chars={len(text)}"
    )
    return RunResult(
        engine=engine_name,
        wav=str(wav.name),
        duration_s=duration_s,
        load_s=load_s,
        first_inference_s=first_inference_s,
        total_transcribe_s=total,
        rtf=rtf,
        peak_mem_mb=peak_mb,
        text_chars=len(text),
        text=text,
    )


def _build_mlx():
    from app.models.asr_client import MlxAudioAsrClient

    return MlxAudioAsrClient(
        model="mlx-community/Qwen3-ASR-0.6B-4bit",
        language="Chinese",
        max_tokens=512,
        repetition_penalty=1.08,
        repetition_context_size=100,
        chunk_duration=30.0,
        long_audio_threshold=60.0,
        hotwords=(),
        system_prompt="你是一个专业的中文语音转写器。只输出音频中实际说出的内容，不要补充、解释或改写。",
    )


def _build_sherpa():
    from app.models.asr_client import SherpaSenseVoiceAsrClient

    return SherpaSenseVoiceAsrClient(
        model_dir=str(
            PROJECT_ROOT / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
        ),
        language="",
        num_threads=2,
        provider="cpu",
        auto_language=True,
        use_itn=True,
        chunk_duration=30.0,
        long_audio_threshold=60.0,
        hotwords=(),
    )


def _build_zipformer():
    from app.models.asr_client import SherpaZipformerBilingualAsrClient

    return SherpaZipformerBilingualAsrClient(
        model_dir=str(
            PROJECT_ROOT / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        ),
        language="",
        num_threads=2,
        provider="cpu",
        decoding_method="greedy_search",
        chunk_duration=30.0,
        long_audio_threshold=60.0,
        hotwords=(),
        model_type="zipformer",
        modeling_unit="bpe",
        sample_rate=16000,
        feature_dim=80,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


@dataclass
class PairwiseCompare:
    wav: str
    duration_s: float
    sherpa_vs_mlx_similarity: float  # 0..1 ratio
    sherpa_extra_chars: int
    mlx_extra_chars: int
    diff_html_excerpt: str = ""  # first 500 chars of unified diff


def compare(a: RunResult, b: RunResult) -> PairwiseCompare:
    ta, tb = a.text, b.text
    sim = difflib.SequenceMatcher(None, ta, tb).ratio()
    extras_a = sum(len(tag) for tag, *_ in difflib.SequenceMatcher(None, ta, tb).get_opcodes() if tag in {"replace", "delete"})
    extras_b = sum(len(tag) for tag, *_ in difflib.SequenceMatcher(None, ta, tb).get_opcodes() if tag in {"replace", "insert"})
    ud = list(difflib.unified_diff(ta.splitlines(), tb.splitlines(), lineterm=""))
    excerpt = "\n".join(ud[:20])
    return PairwiseCompare(
        wav=a.wav,
        duration_s=a.duration_s,
        sherpa_vs_mlx_similarity=sim,
        sherpa_extra_chars=extras_b,
        mlx_extra_chars=extras_a,
        diff_html_excerpt=excerpt,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def pick_crop(m4a: Path, short_crop: float | None, long_crop: float | None) -> float | None:
    """Apply a crop so tests don't run forever on 30-min audio."""
    try:
        import re

        out = subprocess.run(["/usr/bin/afinfo", str(m4a)], capture_output=True, text=True).stdout
        for line in out.splitlines():
            m = re.search(r"estimated duration:\s*([\d.]+)\s*sec", line)
            if m:
                dur = float(m.group(1))
                if dur <= 60:
                    return short_crop  # short audio, use user setting (or None = full)
                return long_crop
    except Exception:
        return long_crop
    return long_crop


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-crop", type=float, default=None,
                        help="Crop <=60s audio to this many seconds (None = full length)")
    parser.add_argument("--long-crop", type=float, default=None,
                        help="Crop >60s audio to this many seconds (None = full length, default for user request)")
    parser.add_argument("--engines", default="mlx,sherpa,zipformer",
                        help="Comma-separated list of engines: mlx, sherpa, zipformer")
    args = parser.parse_args()

    engines = {e.strip().lower() for e in args.engines.split(",") if e.strip()}
    m4as = sorted(PROJECT_ROOT.glob("*.m4a"))
    if not m4as:
        print("No *.m4a in project root.")
        return 2

    all_runs: list[RunResult] = []
    comps: list[PairwiseCompare] = []

    with tempfile.TemporaryDirectory(prefix="astra_asr_cmp_") as _td:
        td = Path(_td)
        for m4a in m4as:
            crop = pick_crop(m4a, args.short_crop, args.long_crop)
            print(f"\n===== {m4a.name} (crop to {crop}s) =====")
            wav, dur = m4a_to_wav(m4a, crop_seconds=crop, tmpdir=td)
            print(f"wav={wav.name}, {dur:.2f}s, {wav.stat().st_size//1024}KB")

            per_file: dict[str, RunResult] = {}
            if "mlx" in engines:
                res = await _load_and_run("mlx-0.6B-4bit", _build_mlx, wav, dur, label=m4a.name)
                all_runs.append(res)
                per_file["mlx"] = res
            if "sherpa" in engines:
                # subprocess the sherpa run into the same python interpreter, but forcefully
                # reset any onnx session state.
                res = await _load_and_run(
                    "sherpa-sensevoice-int8", _build_sherpa, wav, dur, label=m4a.name
                )
                all_runs.append(res)
                per_file["sherpa"] = res
            if "zipformer" in engines:
                res = await _load_and_run(
                    "sherpa-zipformer-bilingual-zh-en-int8",
                    _build_zipformer,
                    wav,
                    dur,
                    label=m4a.name,
                )
                all_runs.append(res)
                per_file["zipformer"] = res

            # Pairwise: compare every pair in per_file
            keys = list(per_file.keys())
            for i, ka in enumerate(keys):
                for kb in keys[i + 1 :]:
                    c = compare(per_file[ka], per_file[kb])
                    comps.append(c)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ASR BENCHMARK REPORT")
    print("=" * 80)
    print("\n--- Per-run timings ---")
    runs_dicts = [asdict(r) for r in all_runs]
    # Print compact table
    header = ["wav", "engine", "dur(s)", "load(s)", "inf(s)", "RTF", "peakMB", "chars", "error"]
    widths = [38, 24, 8, 8, 8, 7, 8, 7, 30]
    fmt = " ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    for r in runs_dicts:
        print(
            fmt.format(
                (r["wav"] or "")[:38],
                (r["engine"] or "")[:24],
                f"{r['duration_s']:.1f}",
                f"{r['load_s']:.1f}",
                f"{r['total_transcribe_s']:.2f}",
                f"{r['rtf']:.3f}",
                f"{r['peak_mem_mb']}",
                f"{r['text_chars']}",
                (r["error"] or "")[:30],
            )
        )

    print("\n--- Per-file pairwise comparison (SenseVoice vs Qwen3-0.6B) ---")
    h2 = ["wav", "dur(s)", "char_similarity", "sherpa_delta_chars", "mlx_delta_chars"]
    for h in h2:
        print(f"{h:<40}", end="")
    print()
    for c in comps:
        print(
            f"{c.wav:<40}{c.duration_s:<8.1f}{c.sherpa_vs_mlx_similarity:<17.3f}"
            f"{c.sherpa_extra_chars:<19}{c.mlx_extra_chars:<17}"
        )
        if c.diff_html_excerpt:
            print("  diff snippet:\n    " + c.diff_html_excerpt.replace("\n", "\n    "))

    # Print full text side by side (2-column)
    print("\n--- Per-file full transcription texts ---")
    for m4a in m4as:
        key = m4a.stem + ".wav"
        mlx_r = next((r for r in all_runs if r.engine.startswith("mlx") and r.wav == key), None)
        sh_r = next((r for r in all_runs if r.engine.startswith("sherpa") and r.wav == key), None)
        print(f"\n### {m4a.name}")
        if mlx_r:
            print(f"[Qwen3-ASR-0.6B-4bit | {mlx_r.total_transcribe_s:.2f}s, RTF {mlx_r.rtf:.3f}]:")
            print(mlx_r.text)
        if sh_r:
            print(f"[SenseVoice(sherpa int8) | {sh_r.total_transcribe_s:.2f}s, RTF {sh_r.rtf:.3f}]:")
            print(sh_r.text)

    # Dump machine-readable JSON for later analysis
    out = PROJECT_ROOT / "_asr_compare_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": [asdict(r) for r in all_runs],
                "compare": [asdict(c) for c in comps],
                "env": {
                    "cwd": str(PROJECT_ROOT),
                    "args": vars(args),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nMachine-readable JSON saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
