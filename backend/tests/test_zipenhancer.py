import asyncio

import numpy as np
import pytest

from app.core.audio_enhancement import AudioBuffer, EnhancementContext
from app.core.audio_adapter import audio_buffer_to_wav_bytes, decode_audio_file
from app.core.audio_enhancement import AudioEnhancementPipeline, EnhancementMetrics
from app.core.meeting import MeetingPipeline
from app.core.zipenhancer import (
    ZipEnhancerStage,
    build_audio_enhancement_pipeline,
)


class FakeZipBackend:
    def __init__(self) -> None:
        self.received: list[bytes] = []

    def __call__(self, audio: bytes) -> dict[str, bytes]:
        self.received.append(audio)
        return {"output_pcm": (np.zeros(160, dtype="<i2")).tobytes()}


def test_zipenhancer_stage_uses_modelscope_contract_and_preserves_length() -> None:
    backend = FakeZipBackend()
    audio = AudioBuffer(
        samples=np.linspace(-0.5, 0.5, 320, dtype=np.float32),
        sample_rate=16_000,
        channels=1,
    )

    output, metrics = asyncio.run(
        ZipEnhancerStage(backend=backend).process(audio, EnhancementContext())
    )

    assert len(backend.received) == 1
    assert backend.received[0][:4] == b"RIFF"
    assert output.sample_rate == 16_000
    assert output.channels == 1
    assert len(output.samples) == 320
    assert metrics.status == "applied"
    assert metrics.details["backend"] == "modelscope"


def test_zipenhancer_is_not_applicable_to_realtime_context() -> None:
    audio = AudioBuffer(samples=np.zeros(160, dtype=np.float32), sample_rate=16_000, channels=1)

    output, metrics = asyncio.run(
        ZipEnhancerStage(backend=FakeZipBackend()).process(audio, EnhancementContext(realtime=True))
    )

    assert output is audio
    assert metrics.status == "not_applicable"
    assert "offline" in (metrics.fallback_reason or "")


def test_zipenhancer_rejects_non_16k_input() -> None:
    audio = AudioBuffer(samples=np.zeros(80), sample_rate=8_000, channels=1)

    with pytest.raises(ValueError, match="16 kHz"):
        asyncio.run(ZipEnhancerStage(backend=FakeZipBackend()).process(audio, EnhancementContext()))


def test_build_zipenhancer_pipeline_is_disabled_by_default() -> None:
    assert build_audio_enhancement_pipeline(enabled=False, ans_model="zipenhancer_16k") is None
    pipeline = build_audio_enhancement_pipeline(enabled=True, ans_model="zipenhancer_16k")
    assert pipeline is not None
    assert pipeline.stages[0].name == "zipenhancer_16k"


def test_meeting_pipeline_consumes_enhanced_audio_and_records_metrics() -> None:
    class FakeStage:
        name = "fake_ans"
        input_sample_rate = 16_000
        output_sample_rate = 16_000
        realtime = False

        async def process(
            self, audio: AudioBuffer, context: EnhancementContext
        ) -> tuple[AudioBuffer, EnhancementMetrics]:
            return (
                AudioBuffer(
                    samples=np.zeros_like(audio.samples),
                    sample_rate=audio.sample_rate,
                    channels=audio.channels,
                    start_time=audio.start_time,
                    source=audio.source,
                ),
                EnhancementMetrics(
                    stage_name=self.name,
                    status="applied",
                    input_sample_rate=audio.sample_rate,
                    output_sample_rate=audio.sample_rate,
                    latency_ms=1.0,
                    reference_present=context.reference is not None,
                ),
            )

    pipeline = MeetingPipeline(
        asr=None,
        diarization=None,
        enhancement=AudioEnhancementPipeline([FakeStage()]),
    )
    observed: dict[str, bool] = {}

    async def fake_transcribe(wav: str, *, progress=None):
        enhanced = decode_audio_file(wav)
        observed["is_silent"] = bool(np.max(np.abs(enhanced.samples)) == 0)
        return "zh", []

    pipeline.transcribe = fake_transcribe
    source = AudioBuffer(
        samples=np.full(320, 0.5, dtype=np.float32),
        sample_rate=16_000,
        channels=1,
    )

    result = asyncio.run(
        pipeline.process(
            audio_buffer_to_wav_bytes(source),
            filename="meeting.wav",
            summarize=False,
            do_translate=False,
        )
    )

    assert observed["is_silent"] is True
    assert result.enhancement_metrics[0]["stage_name"] == "fake_ans"
