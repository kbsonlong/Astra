"""Audio enhancement stage contracts and the default no-op implementation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import numpy as np


AudioSource = Literal["microphone", "upload", "reference", "separated"]
EnhancementStatus = Literal["applied", "not_applicable", "failed", "disabled"]


@dataclass(frozen=True)
class AudioBuffer:
    """A decoded audio buffer exchanged between enhancement stages."""

    samples: object
    sample_rate: int
    channels: int
    start_time: float = 0.0
    source: AudioSource = "upload"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if self.channels <= 0:
            raise ValueError("channels must be greater than 0")
        if self.start_time < 0:
            raise ValueError("start_time must not be negative")


@dataclass(frozen=True)
class EnhancementContext:
    reference: AudioBuffer | None = None
    session_id: str = ""
    task_id: str | None = None
    realtime: bool = False
    model_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EnhancementMetrics:
    stage_name: str
    status: EnhancementStatus
    input_sample_rate: int
    output_sample_rate: int
    latency_ms: float
    reference_present: bool
    fallback_reason: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "input_sample_rate": self.input_sample_rate,
            "output_sample_rate": self.output_sample_rate,
            "latency_ms": self.latency_ms,
            "reference_present": self.reference_present,
            "fallback_reason": self.fallback_reason,
            "details": dict(self.details),
        }


class AudioEnhancementStage(Protocol):
    name: str
    input_sample_rate: int
    output_sample_rate: int
    realtime: bool

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[AudioBuffer, EnhancementMetrics]: ...


class PassthroughStage:
    """No-op stage used until an enhancement model is explicitly enabled."""

    name = "passthrough"
    realtime = True

    def __init__(self, sample_rate: int | None = None) -> None:
        self._sample_rate = sample_rate

    @property
    def input_sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("passthrough sample rate is determined per input")
        return self._sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self.input_sample_rate

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[AudioBuffer, EnhancementMetrics]:
        started = time.perf_counter()
        metrics = EnhancementMetrics(
            stage_name=self.name,
            status="disabled",
            input_sample_rate=audio.sample_rate,
            output_sample_rate=audio.sample_rate,
            latency_ms=(time.perf_counter() - started) * 1000,
            reference_present=context.reference is not None,
            details={"reason": "enhancement disabled"},
        )
        return audio, metrics


class AudioEnhancementPipeline:
    """Run configured stages while preserving the last usable audio on failure."""

    def __init__(self, stages: list[AudioEnhancementStage] | None = None) -> None:
        self.stages = list(stages) if stages is not None else [PassthroughStage()]

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[AudioBuffer, list[EnhancementMetrics]]:
        current = audio
        metrics: list[EnhancementMetrics] = []
        for stage in self.stages:
            try:
                current, stage_metrics = await stage.process(current, context)
            except Exception as exc:  # pragma: no cover - exercised by stage integration tests
                metrics.append(
                    EnhancementMetrics(
                        stage_name=stage.name,
                        status="failed",
                        input_sample_rate=current.sample_rate,
                        output_sample_rate=current.sample_rate,
                        latency_ms=0.0,
                        reference_present=context.reference is not None,
                        fallback_reason=str(exc),
                    )
                )
                continue
            metrics.append(stage_metrics)
        return current, metrics

    async def process_realtime(
        self,
        audio: AudioBuffer,
        context: EnhancementContext,
        *,
        frame_samples: int = 160,
        max_queue: int = 2,
        frame_timeout_ms: float = 80.0,
    ) -> tuple[AudioBuffer, list[EnhancementMetrics]]:
        """Process an utterance through bounded realtime-sized frames.

        The current WebSocket protocol still submits an utterance at
        ``speech_end``. This method keeps model execution at the realtime frame
        boundary so a future PCM frame transport can reuse the same contract.
        Slow frames are bypassed after completion; native model threads are not
        cancelled because cancellation cannot safely stop ModelScope inference.
        """
        if frame_samples <= 0:
            raise ValueError("frame_samples must be greater than 0")
        if max_queue <= 0:
            raise ValueError("max_queue must be greater than 0")
        if frame_timeout_ms <= 0:
            raise ValueError("frame_timeout_ms must be greater than 0")
        if audio.channels != 1:
            raise ValueError("realtime enhancement currently requires mono audio")

        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("realtime enhancement requires one-dimensional audio")
        reference_samples: np.ndarray | None = None
        reference_rate = audio.sample_rate
        if context.reference is not None:
            if context.reference.channels != 1:
                raise ValueError("realtime enhancement reference must be mono")
            reference_samples = np.asarray(context.reference.samples, dtype=np.float32)
            if reference_samples.ndim != 1 or len(reference_samples) != len(samples):
                raise ValueError("realtime enhancement inputs must have equal length")
            reference_rate = context.reference.sample_rate

        queue: asyncio.Queue[tuple[AudioBuffer, AudioBuffer | None] | None] = asyncio.Queue(
            maxsize=max_queue
        )

        async def produce() -> None:
            for start in range(0, len(samples), frame_samples):
                end = min(start + frame_samples, len(samples))
                frame = samples[start:end]
                if len(frame) < frame_samples:
                    frame = np.pad(frame, (0, frame_samples - len(frame)))
                reference = None
                if reference_samples is not None:
                    ref_frame = reference_samples[start:end]
                    if len(ref_frame) < frame_samples:
                        ref_frame = np.pad(ref_frame, (0, frame_samples - len(ref_frame)))
                    reference = AudioBuffer(
                        samples=ref_frame,
                        sample_rate=reference_rate,
                        channels=1,
                        start_time=audio.start_time + start / audio.sample_rate,
                        source="reference",
                    )
                await queue.put(
                    (
                        AudioBuffer(
                            samples=frame,
                            sample_rate=audio.sample_rate,
                            channels=1,
                            start_time=audio.start_time + start / audio.sample_rate,
                            source=audio.source,
                        ),
                        reference,
                    )
                )
            await queue.put(None)

        producer = asyncio.create_task(produce())
        output_frames: list[np.ndarray] = []
        frame_metrics: list[list[EnhancementMetrics]] = []
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                frame_audio, frame_reference = item
                started = time.perf_counter()
                output, metrics = await self.process(
                    frame_audio,
                    replace(context, reference=frame_reference, realtime=True),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                if elapsed_ms > frame_timeout_ms:
                    reason = (
                        f"frame processing exceeded {frame_timeout_ms:g}ms "
                        f"({elapsed_ms:.1f}ms)"
                    )
                    output = frame_audio
                    metrics = [
                        replace(
                            metric,
                            status="failed",
                            fallback_reason=reason,
                            details={
                                **metric.details,
                                "frame_bypassed": True,
                                "frame_latency_ms": round(elapsed_ms, 3),
                            },
                        )
                        for metric in metrics
                    ]
                output_frames.append(np.asarray(output.samples, dtype=np.float32))
                frame_metrics.append(metrics)
        except BaseException:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            raise
        else:
            await producer

        if not frame_metrics:
            return audio, []

        aggregated: list[EnhancementMetrics] = []
        for stage_index in range(len(self.stages)):
            metrics_for_stage = [items[stage_index] for items in frame_metrics]
            first = metrics_for_stage[0]
            applied = sum(item.status == "applied" for item in metrics_for_stage)
            failed = sum(item.status == "failed" for item in metrics_for_stage)
            not_applicable = sum(
                item.status == "not_applicable" for item in metrics_for_stage
            )
            if failed:
                status: EnhancementStatus = "failed"
            elif applied:
                status = "applied"
            elif not_applicable:
                status = "not_applicable"
            else:
                status = "disabled"
            fallback_reason = next(
                (item.fallback_reason for item in metrics_for_stage if item.fallback_reason),
                None,
            )
            aggregated.append(
                EnhancementMetrics(
                    stage_name=first.stage_name,
                    status=status,
                    input_sample_rate=first.input_sample_rate,
                    output_sample_rate=first.output_sample_rate,
                    latency_ms=sum(item.latency_ms for item in metrics_for_stage),
                    reference_present=any(
                        item.reference_present for item in metrics_for_stage
                    ),
                    fallback_reason=fallback_reason,
                    details={
                        **first.details,
                        "frame_count": len(metrics_for_stage),
                        "applied_frames": applied,
                        "failed_frames": failed,
                        "not_applicable_frames": not_applicable,
                        "max_frame_latency_ms": round(
                            max(
                                item.details.get("frame_latency_ms", item.latency_ms)
                                for item in metrics_for_stage
                            ),
                            3,
                        ),
                        "max_queue": max_queue,
                        "frame_timeout_ms": frame_timeout_ms,
                    },
                )
            )

        output_samples = np.concatenate(output_frames)[: len(samples)]
        return (
            AudioBuffer(
                samples=np.asarray(output_samples, dtype=np.float32),
                sample_rate=audio.sample_rate,
                channels=1,
                start_time=audio.start_time,
                source=audio.source,
            ),
            aggregated,
        )
