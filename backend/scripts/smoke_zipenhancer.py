"""Run a real local-model ZipEnhancer smoke test without downloading weights."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from app.core.audio_adapter import audio_buffer_to_wav_bytes, decode_audio_file
from app.core.audio_enhancement import EnhancementContext
from app.core.zipenhancer import ZipEnhancerStage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/astra-zipenhancer-smoke.wav"),
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        parser.error(f"model directory does not exist: {model_dir}")
    input_path = (args.input or model_dir / "examples" / "speech_with_noise.wav").expanduser()
    if not input_path.is_file():
        parser.error(f"input audio does not exist: {input_path}")

    audio = decode_audio_file(input_path)
    stage = ZipEnhancerStage(model_dir=model_dir)
    started = time.perf_counter()
    enhanced, metrics = asyncio.run(
        stage.process(audio, EnhancementContext(realtime=False))
    )
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(audio_buffer_to_wav_bytes(enhanced))
    duration = len(audio.samples) / audio.sample_rate
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "input": str(input_path),
                "output": str(args.output),
                "input_sample_rate": audio.sample_rate,
                "output_sample_rate": enhanced.sample_rate,
                "input_channels": audio.channels,
                "output_channels": enhanced.channels,
                "input_samples": len(audio.samples),
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
