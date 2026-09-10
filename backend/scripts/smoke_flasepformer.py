"""Run a real local-model FLASepformer smoke test without downloading weights."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from app.core.audio_adapter import audio_buffer_to_wav_bytes, decode_audio_file
from app.core.audio_enhancement import EnhancementContext
from app.core.audio_separation import FLASepformerStage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/astra-flasepformer-smoke"),
    )
    parser.add_argument("--window-seconds", type=float, default=30.0)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_dir.is_dir():
        parser.error(f"model directory does not exist: {model_dir}")
    if not input_path.is_file():
        parser.error(f"input audio does not exist: {input_path}")
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be greater than 0")

    audio = decode_audio_file(input_path, target_sample_rate=8_000, source="upload")
    started = time.perf_counter()
    tracks, metrics = asyncio.run(
        FLASepformerStage(
            model_dir=model_dir,
            window_seconds=args.window_seconds,
        ).process(audio, EnhancementContext(realtime=False))
    )
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for index, track in enumerate(tracks):
        output_path = output_dir / f"source-{index}.wav"
        output_path.write_bytes(audio_buffer_to_wav_bytes(track))
        outputs.append(str(output_path))
    duration = len(audio.samples) / audio.sample_rate
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "input": str(input_path),
                "outputs": outputs,
                "input_sample_rate": audio.sample_rate,
                "input_samples": len(audio.samples),
                "output_count": len(tracks),
                "input_duration_s": round(duration, 3),
                "elapsed_s": round(elapsed, 3),
                "real_time_factor": round(elapsed / duration, 3),
                "metrics": metrics.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
