import asyncio

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
