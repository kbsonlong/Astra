"""Optional ModelScope JAEC 16 kHz realtime AEC stage."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

from .audio_adapter import audio_buffer_to_wav_bytes
from .audio_enhancement import (
    AudioBuffer,
    AudioEnhancementPipeline,
    EnhancementContext,
    EnhancementMetrics,
)

JAEC_MODEL_ID = "iic/speech_jaec_aec_16k"
JAEC_FRAME_SAMPLES = 160
JAEC_ALGORITHMIC_DELAY_SAMPLES = 352
JAEC_ALGORITHMIC_DELAY_MS = 22.0


class JAECStage:
    """Run JAEC with an explicit far-end reference buffer."""

    name = "jaec_16k"
    input_sample_rate = 16_000
    output_sample_rate = 16_000
    realtime = True

    def __init__(
        self,
        *,
        model_id: str = JAEC_MODEL_ID,
        model_dir: str | Path | None = None,
        backend: object | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = str(Path(model_dir).expanduser()) if model_dir else ""
        self._backend = backend

    def _load_backend(self) -> object:
        if self._backend is not None:
            return self._backend
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "JAEC requires the ModelScope native-code audio backend; "
                "install the optional JAEC dependencies before enabling AUDIO_AEC_MODEL"
            ) from exc
        if not self.model_dir:
            raise RuntimeError(
                "JAEC requires a prepared local model directory; "
                "set AUDIO_ENHANCEMENT_MODEL_DIR before enabling AUDIO_AEC_MODEL"
            )
        model_path = Path(self.model_dir)
        if not model_path.is_dir():
            raise RuntimeError(f"JAEC model directory does not exist: {model_path}")
        self._backend = pipeline(
            Tasks.acoustic_echo_cancellation,
            model=str(model_path),
            device="cpu",
            trust_native_code=True,
        )
        return self._backend

    def _infer(self, microphone_wav: bytes, reference_wav: bytes) -> np.ndarray:
        backend = self._load_backend()
        result = backend(  # type: ignore[operator]
            {"nearend_mic": microphone_wav, "farend_speech": reference_wav}
        )
        if not isinstance(result, dict) or result.get("output_pcm") is None:
            raise RuntimeError("JAEC returned no output_pcm")
        output = result["output_pcm"]
        if not isinstance(output, (bytes, bytearray, memoryview)):
            raise RuntimeError("JAEC output_pcm must be PCM16 bytes")
        raw = bytes(output)
        if len(raw) % 2:
            raise RuntimeError("JAEC returned unaligned PCM16 bytes")
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[AudioBuffer, EnhancementMetrics]:
        started = time.perf_counter()
        reference = context.reference
        if reference is None:
            return audio, EnhancementMetrics(
                stage_name=self.name,
                status="not_applicable",
                input_sample_rate=audio.sample_rate,
                output_sample_rate=audio.sample_rate,
                latency_ms=0.0,
                reference_present=False,
                fallback_reason="JAEC requires a far-end reference signal",
            )
        if audio.sample_rate != self.input_sample_rate or audio.channels != 1:
            raise ValueError("JAEC requires mono 16 kHz microphone input")
        if reference.sample_rate != self.input_sample_rate or reference.channels != 1:
            raise ValueError("JAEC requires mono 16 kHz far-end reference input")
        microphone_samples = np.asarray(audio.samples)
        reference_samples = np.asarray(reference.samples)
        if len(microphone_samples) != len(reference_samples):
            raise ValueError("JAEC microphone and reference inputs must have equal length")

        output = await asyncio.to_thread(
            self._infer,
            audio_buffer_to_wav_bytes(audio),
            audio_buffer_to_wav_bytes(reference),
        )
        expected_samples = len(microphone_samples)
        if len(output) < expected_samples:
            output = np.pad(output, (0, expected_samples - len(output)))
        else:
            output = output[:expected_samples]
        enhanced = AudioBuffer(
            samples=np.asarray(np.clip(output, -1.0, 1.0), dtype=np.float32),
            sample_rate=self.output_sample_rate,
            channels=1,
            start_time=audio.start_time,
            source=audio.source,
        )
        return enhanced, EnhancementMetrics(
            stage_name=self.name,
            status="applied",
            input_sample_rate=audio.sample_rate,
            output_sample_rate=enhanced.sample_rate,
            latency_ms=(time.perf_counter() - started) * 1000,
            reference_present=True,
            details={
                "backend": "modelscope_native",
                "model_id": self.model_id,
                "processing_frame_samples": JAEC_FRAME_SAMPLES,
                "algorithmic_delay_samples": JAEC_ALGORITHMIC_DELAY_SAMPLES,
                "algorithmic_delay_ms": JAEC_ALGORITHMIC_DELAY_MS,
                "tde_lp_intermediate_outputs": False,
                "output_samples": len(output),
            },
        )


def build_audio_aec_pipeline(
    *,
    enabled: bool,
    aec_model: str,
    model_dir: str = "",
) -> AudioEnhancementPipeline | None:
    """Build the explicitly configured realtime AEC pipeline."""
    if not enabled or aec_model in {"", "none"}:
        return None
    if aec_model != "jaec_16k":
        raise ValueError(f"unsupported audio AEC model: {aec_model}")
    return AudioEnhancementPipeline([JAECStage(model_dir=model_dir or None)])
