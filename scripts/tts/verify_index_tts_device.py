"""Run a local Piper + IndexTTS hardware smoke test without storing user audio."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import resource
import statistics
import tempfile
import time
import wave
from pathlib import Path

from app.config import Settings
from app.core.tts_voice_store import TtsVoiceSnapshot
from app.models.tts_client import IndexTTS25MlxClient, PiperSdkTtsClient


async def verify() -> dict[str, object]:
    settings = Settings.from_env()
    reference_audio = await PiperSdkTtsClient(settings.tts_model_path).synthesize(
        "你好，这是用于 IndexTTS 连续合成稳定性验证的临时参考音频。请保持自然清晰的语速。"
    )
    with wave.open(io.BytesIO(reference_audio), "rb") as wav:
        reference_duration = wav.getnframes() / wav.getframerate()
    if not 5 <= reference_duration <= 15:
        raise RuntimeError(
            f"temporary reference duration {reference_duration:.3f}s is outside 5–15s"
        )

    with tempfile.TemporaryDirectory(prefix="astra-indextts-device-check-") as directory:
        reference_path = Path(directory) / "reference.wav"
        reference_path.write_bytes(reference_audio)
        snapshot = TtsVoiceSnapshot(
            voice_id="device-validation",
            voice_revision=1,
            backend_family="indextts25_mlx",
            language="zh",
            reference_path=reference_path,
            reference_sha256=hashlib.sha256(reference_audio).hexdigest(),
            default_params={"greedy": True},
        )
        client = IndexTTS25MlxClient(
            settings.tts_indextts_model_dir,
            model_revision=settings.tts_indextts_model_revision,
        )
        latencies: list[float] = []
        try:
            for index in range(10):
                text = f"连续合成第 {index + 1} 轮，检查 IndexTTS 输出稳定性。"
                started = time.perf_counter()
                result = await client.synthesize_snapshot(text, snapshot)
                elapsed = time.perf_counter() - started
                with wave.open(io.BytesIO(result), "rb") as wav:
                    if wav.getframerate() != 22_050 or wav.getnchannels() != 1:
                        raise RuntimeError(f"invalid WAV parameters in round {index + 1}")
                    if wav.getnframes() <= 0:
                        raise RuntimeError(f"empty audio in round {index + 1}")
                latencies.append(elapsed)
                print(
                    f"round={index + 1} latency_s={elapsed:.3f} bytes={len(result)}",
                    flush=True,
                )
        finally:
            await client.aclose()

    return {
        "rounds": len(latencies),
        "reference_duration_s": round(reference_duration, 3),
        "min_s": round(min(latencies), 3),
        "median_s": round(statistics.median(latencies), 3),
        "max_s": round(max(latencies), 3),
        "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_revision": settings.tts_indextts_model_revision,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False), flush=True)
