"""Optional offline FLASepformer speech-separation stage."""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .audio_adapter import audio_buffer_to_wav_bytes
from .audio_enhancement import AudioBuffer, EnhancementContext

FLASEPFORMER_MODEL_ID = "iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
FLASEPFORMER_SAMPLE_RATE = 8_000
FLASEPFORMER_SPEAKERS = 2
FLASEPFORMER_WINDOW_SECONDS = 30.0
SeparationStatus = Literal["applied", "not_applicable", "failed", "disabled"]


@dataclass(frozen=True)
class SeparationMetrics:
    stage_name: str
    status: SeparationStatus
    input_sample_rate: int
    output_sample_rate: int
    latency_ms: float
    output_count: int
    fallback_reason: str | None = None
    details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "input_sample_rate": self.input_sample_rate,
            "output_sample_rate": self.output_sample_rate,
            "latency_ms": self.latency_ms,
            "output_count": self.output_count,
            "fallback_reason": self.fallback_reason,
            "details": dict(self.details or {}),
        }


class AudioSeparationStage(Protocol):
    name: str
    input_sample_rate: int
    output_sample_rate: int
    realtime: bool

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[list[AudioBuffer], SeparationMetrics]: ...


class FLASepformerStage:
    """Separate one 8 kHz mono mix into two bounded offline tracks."""

    name = "flasepformer_8k"
    input_sample_rate = FLASEPFORMER_SAMPLE_RATE
    output_sample_rate = FLASEPFORMER_SAMPLE_RATE
    realtime = False

    def __init__(
        self,
        *,
        model_id: str = FLASEPFORMER_MODEL_ID,
        model_dir: str | Path | None = None,
        window_seconds: float = FLASEPFORMER_WINDOW_SECONDS,
        backend: object | None = None,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("FLASepformer window_seconds must be greater than 0")
        self.model_id = model_id
        self.model_dir = str(Path(model_dir).expanduser()) if model_dir else ""
        self.window_seconds = window_seconds
        self._backend = backend

    def _load_backend(self) -> object:
        if self._backend is not None:
            return self._backend
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "FLASepformer requires the optional ModelScope speech-separation backend"
            ) from exc
        if not self.model_dir:
            raise RuntimeError(
                "FLASepformer requires a prepared local model directory; "
                "set AUDIO_ENHANCEMENT_MODEL_DIR before enabling separation"
            )
        model_path = Path(self.model_dir)
        if not model_path.is_dir():
            raise RuntimeError(f"FLASepformer model directory does not exist: {model_path}")
        self._backend = pipeline(
            Tasks.speech_separation,
            model=str(model_path),
            device="cpu",
        )
        return self._backend

    def _infer(self, wav_bytes: bytes) -> list[np.ndarray]:
        backend = self._load_backend()
        with tempfile.TemporaryDirectory(prefix="astra_flasep_") as directory:
            wav_path = Path(directory) / "mix.wav"
            wav_path.write_bytes(wav_bytes)
            result = backend(str(wav_path))  # type: ignore[operator]
        if not isinstance(result, dict):
            raise RuntimeError("FLASepformer returned an invalid result")
        outputs = result.get("output_pcm_list")
        if outputs is None:
            try:
                from modelscope.outputs import OutputKeys

                outputs = result.get(OutputKeys.OUTPUT_PCM_LIST)
            except (ImportError, AttributeError):
                outputs = None
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise RuntimeError("FLASepformer returned no separated PCM tracks")
        tracks: list[np.ndarray] = []
        for output in outputs:
            if not isinstance(output, (bytes, bytearray, memoryview)):
                raise RuntimeError("FLASepformer output tracks must be PCM16 bytes")
            raw = bytes(output)
            if len(raw) % 2:
                raise RuntimeError("FLASepformer returned unaligned PCM16 bytes")
            tracks.append(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
        return tracks

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[list[AudioBuffer], SeparationMetrics]:
        started = time.perf_counter()
        if context.realtime:
            return [], SeparationMetrics(
                stage_name=self.name,
                status="not_applicable",
                input_sample_rate=audio.sample_rate,
                output_sample_rate=audio.sample_rate,
                latency_ms=0.0,
                output_count=0,
                fallback_reason="FLASepformer is an offline bounded-window stage",
            )
        if audio.sample_rate != self.input_sample_rate or audio.channels != 1:
            raise ValueError("FLASepformer requires mono 8 kHz input")

        samples = np.asarray(audio.samples, dtype=np.float32)
        window_samples = max(1, round(self.window_seconds * self.input_sample_rate))
        separated: list[list[np.ndarray]] = []
        window_count = 0
        for start in range(0, len(samples), window_samples):
            window = samples[start : start + window_samples]
            if len(window) == 0:
                continue
            window_audio = AudioBuffer(
                samples=window,
                sample_rate=self.input_sample_rate,
                channels=1,
                start_time=audio.start_time + start / self.input_sample_rate,
                source=audio.source,
            )
            tracks = await asyncio.to_thread(
                self._infer, audio_buffer_to_wav_bytes(window_audio)
            )
            if len(separated) == 0:
                separated = [[] for _ in range(len(tracks))]
            if len(tracks) != len(separated):
                raise RuntimeError("FLASepformer changed the number of output tracks")
            for index, track in enumerate(tracks):
                if len(track) < len(window):
                    track = np.pad(track, (0, len(window) - len(track)))
                separated[index].append(track[: len(window)])
            window_count += 1

        outputs = [
            AudioBuffer(
                samples=np.asarray(np.clip(np.concatenate(chunks), -1.0, 1.0), dtype=np.float32),
                sample_rate=self.output_sample_rate,
                channels=1,
                start_time=audio.start_time,
                source="separated",
            )
            for chunks in separated
        ]
        return outputs, SeparationMetrics(
            stage_name=self.name,
            status="applied",
            input_sample_rate=audio.sample_rate,
            output_sample_rate=self.output_sample_rate,
            latency_ms=(time.perf_counter() - started) * 1000,
            output_count=len(outputs),
            details={
                "backend": "modelscope",
                "model_id": self.model_id,
                "window_seconds": self.window_seconds,
                "window_samples": window_samples,
                "window_count": window_count,
                "source_count": len(outputs),
                "input_samples": len(samples),
            },
        )


def build_audio_separation_stage(
    *,
    enabled: bool,
    separation_model: str,
    model_dir: str = "",
    window_seconds: float = FLASEPFORMER_WINDOW_SECONDS,
) -> AudioSeparationStage | None:
    if not enabled or separation_model in {"", "none"}:
        return None
    if separation_model != "flasepformer_8k":
        raise ValueError(f"unsupported audio separation model: {separation_model}")
    return FLASepformerStage(model_dir=model_dir or None, window_seconds=window_seconds)
