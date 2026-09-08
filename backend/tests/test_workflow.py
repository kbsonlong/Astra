from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from app.core.speaker_registry import SpeakerMatch
from app.core.workflow import (
    AudioWorkflow,
    CorrectionStage,
    ResemblyzerDiarizationStage,
    Segment,
    SpeechChunk,
    WorkflowContext,
    WorkflowBuilder,
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


class FakeCleanup:
    name = "correction"

    async def run(self, context: WorkflowContext) -> None:
        events.append("cleanup")
        for segment in context.segments:
            segment.text = f"清洗:{segment.text}"


events: list[str] = []


@pytest.mark.anyio
async def test_audio_workflow_runs_stages_in_order_and_preserves_boundaries() -> None:
    events.clear()
    workflow = AudioWorkflow(FakeVAD(), FakeASR(), FakePunctuation(), FakeSD())

    result = await workflow.run("meeting.wav", filename="meeting.m4a")

    assert events == ["vad", "asr:one", "asr:two", "punct:one", "punct:two", "sd"]
    assert [(s.start, s.end, s.text, s.speaker) for s in result.segments] == [
        (1.25, 2.0, "one。", "S1"),
        (3.0, 4.5, "two。", "S2"),
    ]


@pytest.mark.anyio
async def test_workflow_builder_runs_only_registered_stages_in_order() -> None:
    events.clear()

    class SeedStage:
        name = "seed"

        async def run(self, context) -> None:
            events.append("seed")
            context.metadata["source"] = "custom"

    class FinishStage:
        name = "finish"

        async def run(self, context) -> None:
            events.append("finish")
            context.segments.append(Segment(2.0, 3.0, "自定义阶段"))

    engine = (
        WorkflowBuilder(language="zh")
        .use(SeedStage())
        .use(FinishStage())
        .build()
    )
    result = await engine.run("meeting.wav", filename="meeting.m4a")

    assert events == ["seed", "finish"]
    assert result.metadata == {"source": "custom"}
    assert result.segments[0].text == "自定义阶段"


def test_workflow_builder_rejects_empty_workflow() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        WorkflowBuilder().build()


def test_meeting_pipeline_accepts_custom_workflow() -> None:
    from app.core.meeting import MeetingPipeline

    custom = WorkflowBuilder().use(SeedStageForTest()).build()
    pipeline = MeetingPipeline(asr=FakeASR(), diarization=None, workflow=custom)

    assert pipeline.workflow is custom


class SeedStageForTest:
    name = "custom"

    async def run(self, context) -> None:
        context.segments.append(Segment(0.0, 1.0, "自定义"))


@pytest.mark.anyio
async def test_audio_workflow_keeps_asr_text_when_punctuation_fails() -> None:
    class BrokenPunctuation:
        async def restore(self, text: str, *, language: str = "zh") -> str:
            raise RuntimeError("model unavailable")

    workflow = AudioWorkflow(FakeVAD(), FakeASR(), BrokenPunctuation())
    result = await workflow.run("meeting.wav")

    assert [segment.text for segment in result.segments] == ["one", "two"]


@pytest.mark.anyio
async def test_audio_workflow_places_correction_after_punctuation_and_sd() -> None:
    events.clear()
    workflow = AudioWorkflow(
        FakeVAD(), FakeASR(), FakePunctuation(), FakeSD(), correction=FakeCleanup()
    )

    result = await workflow.run("meeting.wav", filename="meeting.m4a")

    assert events == [
        "vad",
        "asr:one",
        "asr:two",
        "punct:one",
        "punct:two",
        "sd",
        "cleanup",
    ]
    assert [segment.text for segment in result.segments] == ["清洗:one。", "清洗:two。"]


@pytest.mark.anyio
async def test_correction_stage_applies_rules_and_restricts_llm_candidates() -> None:
    rules_context = WorkflowContext(
        wav="meeting.wav",
        filename="meeting.wav",
        language="zh",
        segments=[Segment(0.0, 1.0, "宗师使用后视网络模式")],
    )
    await CorrectionStage().run(rules_context)
    assert rules_context.segments[0].text == "忠思使用host 网络模式"

    class FakeLLM:
        model = "local-test"

        async def stream_chat(self, messages, **kwargs):
            assert "宗师->忠思" in messages[0]["content"]
            assert messages[1]["content"] == "宗师"
            assert kwargs["max_tokens"] == 32
            yield "忠思"

    context = WorkflowContext(
        wav="meeting.wav",
        filename="meeting.wav",
        language="zh",
        segments=[Segment(0.0, 1.0, "宗师")],
    )
    stage = CorrectionStage(
        FakeLLM(),
        rules_enabled=False,
        llm_enabled=True,
        candidate_rules={"宗师": "忠思"},
        system_prompt="只清洗",
        max_tokens=32,
    )
    await stage.run(context)
    assert context.segments[0].text == "忠思"

    class UnsafeLLM(FakeLLM):
        async def stream_chat(self, messages, **kwargs):
            yield "忠思并新增内容"

    unsafe_context = WorkflowContext(
        wav="meeting.wav",
        filename="meeting.wav",
        language="zh",
        segments=[Segment(0.0, 1.0, "宗师")],
    )
    await CorrectionStage(
        UnsafeLLM(),
        rules_enabled=False,
        llm_enabled=True,
        candidate_rules={"宗师": "忠思"},
        system_prompt="只清洗",
        max_tokens=32,
    ).run(unsafe_context)
    assert unsafe_context.segments[0].text == "宗师"

    class BrokenLLM(FakeLLM):
        async def stream_chat(self, messages, **kwargs):
            raise RuntimeError("LLM unavailable")
            yield "never"

    fallback_context = WorkflowContext(
        wav="meeting.wav",
        filename="meeting.wav",
        language="zh",
        segments=[Segment(0.0, 1.0, "原始")],
    )
    await CorrectionStage(
        BrokenLLM(),
        rules_enabled=False,
        llm_enabled=True,
        candidate_rules={"宗师": "忠思"},
        system_prompt="只清洗",
        max_tokens=32,
    ).run(fallback_context)
    assert fallback_context.segments[0].text == "原始"


@pytest.mark.anyio
async def test_correction_stage_keeps_unconfirmed_cluster_machine_term() -> None:
    context = WorkflowContext(
        wav="meeting.wav",
        filename="meeting.wav",
        language="zh",
        segments=[Segment(0.0, 1.0, "集群机单机变成集群")],
    )

    await CorrectionStage().run(context)

    assert context.segments[0].text == "集群机单机变成集群"


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


@pytest.mark.anyio
async def test_meeting_summary_and_translation_stages_are_replaceable() -> None:
    from app.core.meeting import MeetingPipeline

    events: list[str] = []

    class CustomSummary:
        name = "summary"

        async def run(self, context) -> None:
            events.append(f"summary:{context.meeting_topic}")
            context.summary = "自定义纪要"

    class CustomTranslation:
        name = "translation"

        async def run(self, context) -> None:
            events.append(f"translation:{context.target_language}")
            context.translation = f"translated:{context.summary}"

    pipeline = MeetingPipeline(
        asr=FakeASR(),
        diarization=None,
        summary_stage=CustomSummary(),
        translation_stage=CustomTranslation(),
    )
    summary, translation = await pipeline.summarize(
        [Segment(0.0, 1.0, "原文")],
        meeting_topic="测试会议",
        target_language="English",
    )

    assert events == ["summary:测试会议", "translation:English"]
    assert (summary, translation) == ("自定义纪要", "translated:自定义纪要")


def test_diarization_auto_clusters_more_than_two_speakers() -> None:
    stage = ResemblyzerDiarizationStage(cluster_distance_threshold=0.35)
    vectors = np.zeros((6, 256), dtype=np.float32)
    vectors[0:2, 0] = 1.0
    vectors[2:4, 1] = 1.0
    vectors[4:6, 2] = 1.0

    raw = stage._cluster_embeddings(vectors)

    assert len(set(raw)) == 3


@pytest.mark.parametrize(
    ("source", "expected"),
    [("原始文本。。", "原始文本。"), ("原始文本。，下一句", "原始文本。下一句")],
)
def test_workflow_punctuation_cleanup_is_available_at_final_text_boundary(
    source: str, expected: str
) -> None:
    assert clean_repeated_punctuation(source) == expected
