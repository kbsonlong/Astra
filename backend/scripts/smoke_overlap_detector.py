#!/usr/bin/env python3
"""Run the optional local pyannote overlap detector without downloading models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.core.audio_adapter import decode_audio_file
from app.core.audio_separation import PyannoteOverlapDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    audio = decode_audio_file(args.input, target_sample_rate=16_000)
    detector = PyannoteOverlapDetector(model_dir=args.model_dir)
    started = time.perf_counter()
    detection = detector.detect(audio)
    elapsed_s = time.perf_counter() - started
    print(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "input": str(args.input),
                "input_duration_s": round(len(audio.samples) / audio.sample_rate, 3),
                "elapsed_s": round(elapsed_s, 3),
                "detection": detection.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
