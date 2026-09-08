from collections.abc import Sequence
from pathlib import Path

import pytest

from app.core.speaker_registry import SpeakerMatch
from app.core.workflow import (
    AudioWorkflow,
    ResemblyzerDiarizationStage,
    Segment,
    SpeechChunk,
    clean_repeated_punctuation,
)


class FakeVAD:
    async def detect(self, wav: str | Path) -> Sequence[SpeechChunk]:
        events.append("vad")
        return [SpeechChunk(1.25, 2.0, b"one"), SpeechChunk(3.0, 4.5, b"two")]


class FakeASR:
    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        events.append(f"asr:{audio.decode()}")
        return audio.decode()


class FakePunctuation:
    async def restore(self, text: str, *, language: str = "zh") -> str:
        events.append(f"punct:{text}")
        return f"{text}。"


class FakeSD:
    async def assign(self, wav: str | Path, segments: list[Segment]) -> None:
        events.append("sd")
        for index, segment in enumerate(segments, start=1):
            segment.speaker = f"S{index}"


events: list[str] = []


@pytest.mark.anyio
async def test_audio_workflow_runs_stages_in_order_and_preserves_boundaries() -> None:
    events.clear()
    workflow = AudioWorkflow(FakeVAD(), FakeASR(), FakePunctuation(), FakeSD())

    result = await workflow.run("meeting.wav", filename="meeting.m4a")

    assert events == ["vad", "asr:one", "punct:one", "asr:two", "punct:two", "sd"]
    assert [(s.start, s.end, s.text, s.speaker) for s in result.segments] == [
        (1.25, 2.0, "one。", "S1"),
        (3.0, 4.5, "two。", "S2"),
    ]


@pytest.mark.anyio
async def test_audio_workflow_keeps_asr_text_when_punctuation_fails() -> None:
    class BrokenPunctuation:
        async def restore(self, text: str, *, language: str = "zh") -> str:
            raise RuntimeError("model unavailable")

    workflow = AudioWorkflow(FakeVAD(), FakeASR(), BrokenPunctuation())
    result = await workflow.run("meeting.wav")

    assert [segment.text for segment in result.segments] == ["one", "two"]


@pytest.mark.anyio
async def test_diarization_propagates_registered_speaker_identity(monkeypatch) -> None:
    match = SpeakerMatch("speaker-uuid", "忠思", 0.91, "high")
    stage = ResemblyzerDiarizationStage()
    monkeypatch.setattr(
        stage,
        "_assign_blocking",
        lambda wav, segments: [("S1", match) for _ in segments],
    )
    segments = [Segment(0.0, 1.0, "测试")]

    await stage.assign("meeting.wav", segments)

    assert segments[0].speaker == "S1"
    assert segments[0].speaker_id == "speaker-uuid"
    assert segments[0].speaker_name == "忠思"
    assert segments[0].speaker_similarity == 0.91
    assert segments[0].speaker_confidence == "high"


@pytest.mark.parametrize(
    ("source", "expected"),
    [("原始文本。。", "原始文本。"), ("原始文本。，下一句", "原始文本。下一句")],
)
def test_workflow_punctuation_cleanup_is_available_at_final_text_boundary(
    source: str, expected: str
) -> None:
    assert clean_repeated_punctuation(source) == expected
