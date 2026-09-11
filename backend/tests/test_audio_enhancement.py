import asyncio

import numpy as np
import pytest

from app.core.audio_enhancement import (
    AudioBuffer,
    AudioEnhancementPipeline,
    EnhancementContext,
    PassthroughStage,
)


def test_audio_buffer_validates_format_metadata() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        AudioBuffer(samples=b"audio", sample_rate=0, channels=1)
    with pytest.raises(ValueError, match="channels"):
        AudioBuffer(samples=b"audio", sample_rate=16_000, channels=0)
    with pytest.raises(ValueError, match="start_time"):
        AudioBuffer(samples=b"audio", sample_rate=16_000, channels=1, start_time=-1)


def test_passthrough_stage_preserves_audio_and_emits_disabled_metrics() -> None:
    audio = AudioBuffer(samples=b"audio", sample_rate=16_000, channels=1)

    output, metrics = asyncio.run(
        PassthroughStage().process(
            audio,
            EnhancementContext(session_id="session-1", realtime=True),
        )
    )

    assert output is audio
    assert metrics.status == "disabled"
    assert metrics.input_sample_rate == 16_000
    assert metrics.output_sample_rate == 16_000
    assert metrics.reference_present is False
    assert metrics.to_dict()["details"] == {"reason": "enhancement disabled"}


def test_default_enhancement_pipeline_is_observable_but_has_no_audio_effect() -> None:
    audio = AudioBuffer(samples=b"audio", sample_rate=16_000, channels=1)

    output, metrics = asyncio.run(AudioEnhancementPipeline().process(audio, EnhancementContext()))

    assert output is audio
    assert [item.status for item in metrics] == ["disabled"]


def test_pipeline_keeps_previous_audio_when_a_stage_fails() -> None:
    class BrokenStage:
        name = "broken"
        input_sample_rate = 16_000
        output_sample_rate = 16_000
        realtime = False

        async def process(self, audio, context):
            raise RuntimeError("model unavailable")

    audio = AudioBuffer(samples=b"audio", sample_rate=16_000, channels=1)
    output, metrics = asyncio.run(
        AudioEnhancementPipeline([BrokenStage()]).process(
            audio, EnhancementContext(task_id="task-1")
        )
    )

    assert output is audio
    assert len(metrics) == 1
    assert metrics[0].status == "failed"
    assert metrics[0].fallback_reason == "model unavailable"


def test_realtime_pipeline_processes_10ms_frames_and_aggregates_metrics() -> None:
    class FrameStage:
        name = "fake_realtime"
        input_sample_rate = 16_000
        output_sample_rate = 16_000
        realtime = True

        async def process(self, audio, context):
            from app.core.audio_enhancement import EnhancementMetrics

            return audio, EnhancementMetrics(
                stage_name=self.name,
                status="applied",
                input_sample_rate=audio.sample_rate,
                output_sample_rate=audio.sample_rate,
                latency_ms=1.0,
                reference_present=context.reference is not None,
            )

    audio = AudioBuffer(
        samples=np.ones(320, dtype=np.float32), sample_rate=16_000, channels=1
    )
    output, metrics = asyncio.run(
        AudioEnhancementPipeline([FrameStage()]).process_realtime(
            audio, EnhancementContext(realtime=True), frame_samples=160
        )
    )

    assert len(output.samples) == 320
    assert metrics[0].status == "applied"
    assert metrics[0].details["frame_count"] == 2
    assert metrics[0].details["applied_frames"] == 2
