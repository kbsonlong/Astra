"""Optional ModelScope backend for the ZipEnhancer 16 kHz ANS model."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

from .audio_adapter import audio_buffer_to_wav_bytes, decode_audio_bytes
from .audio_enhancement import (
    AudioBuffer,
    AudioEnhancementPipeline,
    EnhancementContext,
    EnhancementMetrics,
)

ZIPENHANCER_MODEL_ID = "iic/speech_zipenhancer_ans_multiloss_16k_base"


class ZipEnhancerStage:
    """Run ZipEnhancer through the optional ModelScope pipeline API."""

    name = "zipenhancer_16k"
    input_sample_rate = 16_000
    output_sample_rate = 16_000
    realtime = False

    def __init__(
        self,
        *,
        model_id: str = ZIPENHANCER_MODEL_ID,
        model_dir: str | Path | None = None,
        device: str = "auto",
        backend: object | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = str(Path(model_dir).expanduser()) if model_dir else ""
        self.device = device
        self._backend = backend

    def _load_backend(self) -> object:
        if self._backend is not None:
            return self._backend
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "ZipEnhancer requires the ModelScope audio dependencies; "
                "install the backend extras before enabling AUDIO_ANS_MODEL"
            ) from exc
        if not self.model_dir:
            raise RuntimeError(
                "ZipEnhancer requires a prepared local model directory; "
                "set AUDIO_ENHANCEMENT_MODEL_DIR before enabling AUDIO_ANS_MODEL"
            )
        model_path = Path(self.model_dir)
        if not model_path.is_dir():
            raise RuntimeError(f"ZipEnhancer model directory does not exist: {model_path}")
        if self.device == "mps":
            # ModelScope 1.39.1 rejects ``device="mps"`` in its pipeline
            # validator. Load through its CPU path, then move the PyTorch
            # model and pipeline input device to MPS explicitly.
            import torch

            if not torch.backends.mps.is_available():
                raise RuntimeError("ZipEnhancer MPS requested but MPS is unavailable")
            pipeline_options = {"device": "cpu"}
        else:
            pipeline_options = {} if self.device == "auto" else {"device": self.device}
        self._backend = pipeline(
            Tasks.acoustic_noise_suppression, model=str(model_path), **pipeline_options
        )
        if self.device == "mps":
            self._backend.device = torch.device("mps")  # type: ignore[attr-defined]
            self._backend.model.to(torch.device("mps"))  # type: ignore[attr-defined]
        return self._backend

    @staticmethod
    def _result_samples(result: object) -> np.ndarray:
        if isinstance(result, dict):
            result = result.get("output_pcm")
        if result is None:
            raise RuntimeError("ZipEnhancer returned no output_pcm")
        if isinstance(result, (bytes, bytearray, memoryview)):
            raw = bytes(result)
            if raw[:4] == b"RIFF":
                return np.asarray(
                    decode_audio_bytes(raw, filename="enhanced.wav").samples,
                    dtype=np.float32,
                )
            if len(raw) % 2:
                raise RuntimeError("ZipEnhancer returned unaligned PCM16 bytes")
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

        samples = np.asarray(result)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if samples.ndim != 1:
            raise RuntimeError("ZipEnhancer output must be one-dimensional audio")
        if np.issubdtype(samples.dtype, np.integer):
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        else:
            samples = samples.astype(np.float32, copy=False)
        return samples

    def _infer(self, wav_bytes: bytes) -> np.ndarray:
        backend = self._load_backend()
        result = backend(wav_bytes)  # type: ignore[operator]
        return self._result_samples(result)

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[AudioBuffer, EnhancementMetrics]:
        started = time.perf_counter()
        reference_present = context.reference is not None
        if context.realtime:
            return audio, EnhancementMetrics(
                stage_name=self.name,
                status="not_applicable",
                input_sample_rate=audio.sample_rate,
                output_sample_rate=audio.sample_rate,
                latency_ms=0.0,
                reference_present=reference_present,
                fallback_reason="ZipEnhancer is configured for offline Phase B use",
            )
        if audio.sample_rate != self.input_sample_rate or audio.channels != 1:
            raise ValueError("ZipEnhancer requires mono 16 kHz AudioBuffer input")

        output = await asyncio.to_thread(self._infer, audio_buffer_to_wav_bytes(audio))
        expected_samples = len(np.asarray(audio.samples))
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
            reference_present=reference_present,
            details={
                "backend": "modelscope",
                "model_id": self.model_id,
                "device": self.device,
                "output_samples": len(output),
            },
        )


def build_audio_enhancement_pipeline(
    *,
    enabled: bool,
    ans_model: str,
    model_dir: str = "",
    device: str = "auto",
) -> AudioEnhancementPipeline | None:
    """Build only the explicitly configured offline enhancement pipeline."""
    if not enabled or ans_model in {"", "none"}:
        return None
    if ans_model != "zipenhancer_16k":
        raise ValueError(f"unsupported audio ANS model: {ans_model}")
    return AudioEnhancementPipeline(
        [ZipEnhancerStage(model_dir=model_dir or None, device=device)]
    )
