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
class OverlapDetection:
    suspected: bool
    score: float
    active_frames: int
    candidate_frames: int
    threshold: float
    details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "suspected": self.suspected,
            "score": self.score,
            "active_frames": self.active_frames,
            "candidate_frames": self.candidate_frames,
            "threshold": self.threshold,
            "details": dict(self.details or {}),
        }


class OverlapDetector(Protocol):
    def detect(self, audio: AudioBuffer) -> OverlapDetection: ...


class SpectralOverlapDetector:
    """Conservative mono overlap heuristic for deciding whether to separate.

    It counts short-time frames with two prominent low-frequency peaks. This is
    intentionally a trigger heuristic, not a speaker-count or diarization
    result; false negatives fall back to the mixed ASR path without changing
    the default manual/always behavior.
    """

    def __init__(
        self,
        *,
        frame_seconds: float = 0.050,
        hop_seconds: float = 0.010,
        min_rms: float = 0.01,
        peak_ratio: float = 0.22,
        min_score: float = 0.45,
        min_active_frames: int = 3,
        max_spectral_flatness: float = 0.65,
        harmonic_tolerance: float = 0.08,
        min_candidate_run_frames: int = 3,
    ) -> None:
        if frame_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("overlap frame and hop durations must be positive")
        if not 0 < peak_ratio < 1 or not 0 < min_score <= 1:
            raise ValueError("overlap ratios must be between 0 and 1")
        if not 0 < max_spectral_flatness <= 1 or not 0 < harmonic_tolerance < 1:
            raise ValueError("overlap spectral thresholds must be between 0 and 1")
        if min_candidate_run_frames <= 0:
            raise ValueError("overlap candidate run length must be positive")
        self.frame_seconds = frame_seconds
        self.hop_seconds = hop_seconds
        self.min_rms = min_rms
        self.peak_ratio = peak_ratio
        self.min_score = min_score
        self.min_active_frames = min_active_frames
        self.max_spectral_flatness = max_spectral_flatness
        self.harmonic_tolerance = harmonic_tolerance
        self.min_candidate_run_frames = min_candidate_run_frames

    def detect(self, audio: AudioBuffer) -> OverlapDetection:
        samples = np.asarray(audio.samples, dtype=np.float32)
        if audio.channels != 1:
            raise ValueError("overlap detection currently requires mono audio")
        if samples.ndim != 1:
            raise ValueError("overlap detection requires one-dimensional samples")
        frame_size = max(16, round(audio.sample_rate * self.frame_seconds))
        hop_size = max(1, round(audio.sample_rate * self.hop_seconds))
        if len(samples) < frame_size:
            samples = np.pad(samples, (0, frame_size - len(samples)))
        window = np.hanning(frame_size).astype(np.float32)
        frequencies = np.fft.rfftfreq(frame_size, 1.0 / audio.sample_rate)
        band = (frequencies >= 80.0) & (frequencies <= 350.0)
        active_frames = 0
        eligible_frames = 0
        candidate_frames = 0
        noise_rejections = 0
        harmonic_rejections = 0
        longest_candidate_run = 0
        candidate_run = 0
        min_peak_distance = max(1, round(40.0 * frame_size / audio.sample_rate))

        for start in range(0, len(samples) - frame_size + 1, hop_size):
            frame = samples[start : start + frame_size]
            rms = float(np.sqrt(np.mean(frame * frame)))
            if rms < self.min_rms:
                candidate_run = 0
                continue
            active_frames += 1
            spectrum = np.abs(np.fft.rfft(frame * window))
            band_spectrum = spectrum[band]
            if not len(band_spectrum):
                candidate_run = 0
                continue
            positive_spectrum = band_spectrum + 1e-9
            spectral_flatness = float(
                np.exp(np.mean(np.log(positive_spectrum)))
                / np.mean(positive_spectrum)
            )
            if spectral_flatness > self.max_spectral_flatness:
                noise_rejections += 1
                candidate_run = 0
                continue
            eligible_frames += 1
            peak_limit = float(band_spectrum.max()) * self.peak_ratio
            peaks = self._find_peaks(
                band_spectrum,
                height=peak_limit,
                distance=min_peak_distance,
            )
            peak_frequencies = frequencies[band][peaks]
            if len(peaks) < 2:
                candidate_run = 0
                continue
            if not self._has_non_harmonic_pair(peak_frequencies):
                harmonic_rejections += 1
                candidate_run = 0
                continue
            candidate_frames += 1
            candidate_run += 1
            longest_candidate_run = max(longest_candidate_run, candidate_run)

        score = candidate_frames / eligible_frames if eligible_frames else 0.0
        return OverlapDetection(
            suspected=(
                active_frames >= self.min_active_frames
                and score >= self.min_score
                and longest_candidate_run >= self.min_candidate_run_frames
            ),
            score=round(score, 4),
            active_frames=active_frames,
            candidate_frames=candidate_frames,
            threshold=self.min_score,
            details={
                "sample_rate": audio.sample_rate,
                "frame_seconds": self.frame_seconds,
                "hop_seconds": self.hop_seconds,
                "frequency_band_hz": [80, 350],
                "peak_ratio": self.peak_ratio,
                "eligible_frames": eligible_frames,
                "max_spectral_flatness": self.max_spectral_flatness,
                "harmonic_tolerance": self.harmonic_tolerance,
                "noise_rejections": noise_rejections,
                "harmonic_rejections": harmonic_rejections,
                "longest_candidate_run_frames": longest_candidate_run,
                "min_candidate_run_frames": self.min_candidate_run_frames,
            },
        )

    def _has_non_harmonic_pair(self, frequencies: np.ndarray) -> bool:
        if len(frequencies) < 2:
            return False
        base = float(np.min(frequencies))
        ratios = frequencies / max(base, 1e-6)
        harmonic_series = all(
            abs(float(ratio) - round(float(ratio))) <= self.harmonic_tolerance
            for ratio in ratios
        )
        return not harmonic_series

    @staticmethod
    def _find_peaks(
        values: np.ndarray, *, height: float, distance: int
    ) -> np.ndarray:
        """Small dependency-free peak finder for the bounded detector band."""
        candidates = np.flatnonzero(
            (values[1:-1] >= values[:-2])
            & (values[1:-1] >= values[2:])
            & (values[1:-1] >= height)
        ) + 1
        if len(candidates) <= 1:
            return candidates
        order = candidates[np.argsort(values[candidates])[::-1]]
        selected: list[int] = []
        for candidate in order:
            if all(abs(int(candidate) - item) >= distance for item in selected):
                selected.append(int(candidate))
        return np.asarray(selected, dtype=np.int64)


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
