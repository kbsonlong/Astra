"""Run a real local-model JAEC smoke test without downloading weights."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from app.core.audio_adapter import audio_buffer_to_wav_bytes, decode_audio_file
from app.core.audio_enhancement import EnhancementContext
from app.core.jaec import JAECStage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--microphone", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/astra-jaec-smoke.wav"),
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    microphone_path = args.microphone.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    if not model_dir.is_dir():
        parser.error(f"model directory does not exist: {model_dir}")
    for path in (microphone_path, reference_path):
        if not path.is_file():
            parser.error(f"input audio does not exist: {path}")

    microphone = decode_audio_file(microphone_path, source="microphone")
    reference = decode_audio_file(reference_path, source="reference")
    started = time.perf_counter()
    enhanced, metrics = asyncio.run(
        JAECStage(model_dir=model_dir).process(
            microphone,
            EnhancementContext(reference=reference, realtime=True),
        )
    )
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(audio_buffer_to_wav_bytes(enhanced))
    duration = len(microphone.samples) / microphone.sample_rate
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "microphone": str(microphone_path),
                "reference": str(reference_path),
                "output": str(args.output),
                "input_sample_rate": microphone.sample_rate,
                "output_sample_rate": enhanced.sample_rate,
                "input_samples": len(microphone.samples),
                "output_samples": len(enhanced.samples),
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
