"""Benchmark the local ZipEnhancer model on CPU and/or MPS."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch

from app.core.audio_adapter import decode_audio_file
from app.core.audio_enhancement import EnhancementContext
from app.core.zipenhancer import ZipEnhancerStage


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if percentile == 0.5:
        return statistics.median(ordered)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


async def _measure(
    stage: ZipEnhancerStage,
    audio: object,
    *,
    warmups: int,
    repeats: int,
) -> list[float]:
    context = EnhancementContext(realtime=False)
    for _ in range(warmups):
        await stage.process(audio, context)  # type: ignore[arg-type]
    durations: list[float] = []
    for _ in range(repeats):
        _synchronize(stage.device)
        started = time.perf_counter()
        output, metrics = await stage.process(audio, context)  # type: ignore[arg-type]
        _synchronize(stage.device)
        durations.append(time.perf_counter() - started)
        if metrics.status != "applied":
            raise RuntimeError(f"ZipEnhancer returned {metrics.status}: {metrics.fallback_reason}")
        if len(output.samples) != len(audio.samples):  # type: ignore[attr-defined]
            raise RuntimeError("ZipEnhancer changed the sample count")
    return durations


def _device_available(device: str) -> bool:
    return device == "cpu" or (
        device == "mps" and torch.backends.mps.is_available()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/astra-zipenhancer-benchmark.json"),
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")

    model_dir = args.model_dir.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    if not model_dir.is_dir():
        parser.error(f"model directory does not exist: {model_dir}")
    if not input_path.is_file():
        parser.error(f"input audio does not exist: {input_path}")
    audio = decode_audio_file(input_path)
    duration = len(audio.samples) / audio.sample_rate

    results: list[dict[str, object]] = []
    for device in (item.strip() for item in args.devices.split(",")):
        if not device:
            continue
        if not _device_available(device):
            results.append({"device": device, "status": "skipped", "reason": "unavailable"})
            continue
        stage = ZipEnhancerStage(model_dir=model_dir, device=device)
        try:
            started = time.perf_counter()
            stage._load_backend()  # noqa: SLF001 - benchmark the backend load separately
            load_s = time.perf_counter() - started
            durations = asyncio.run(
                _measure(
                    stage,
                    audio,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            )
            results.append(
                {
                    "device": device,
                    "status": "ok",
                    "load_s": round(load_s, 3),
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "sample_rate": audio.sample_rate,
                    "duration_s": round(duration, 3),
                    "p50_s": round(_percentile(durations, 0.50), 3),
                    "p95_s": round(_percentile(durations, 0.95), 3),
                    "p50_rtf": round(_percentile(durations, 0.50) / duration, 3),
                    "p95_rtf": round(_percentile(durations, 0.95) / duration, 3),
                    "runs_s": [round(value, 3) for value in durations],
                }
            )
        except Exception as exc:
            results.append({"device": device, "status": "failed", "error": str(exc)})
        finally:
            del stage
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    payload = {
        "model_dir": str(model_dir),
        "input": str(input_path),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] != "failed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
