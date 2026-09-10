"""Audio enhancement stage contracts and the default no-op implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Protocol


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
