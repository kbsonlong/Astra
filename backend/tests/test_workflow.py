from collections.abc import Sequence
import json
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
    assert [s.audio for s in result.segments] == [b"one", b"two"]
    assert [s.raw_text for s in result.segments] == ["one", "two"]


def test_meeting_worker_exports_reviewable_qwen_jsonl(tmp_path) -> None:
    from app.core.meeting import MeetingResult
    from app.core.meeting_cli import _write_training_artifacts

    result = MeetingResult(
        filename="meeting.wav",
        duration_s=2.0,
        language="zh",
        segments=[Segment(0.0, 1.0, "云资源中心。", speaker="S1", audio=b"RIFF", raw_text="冷资源中心")],
    )

    artifacts = _write_training_artifacts(tmp_path, result)
    assert artifacts["candidate_segments"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "qwen3-asr-candidates.jsonl").read_text().splitlines()
    ]
    assert rows[0]["audio"].endswith("asr_clips/seg-000001.wav")
    assert rows[0]["text"] == "云资源中心。"
    assert rows[0]["review_status"] == "pending"
    assert (tmp_path / "asr_clips/seg-000001.wav").read_bytes() == b"RIFF"


def test_training_export_merges_adjacent_same_speaker_segments(tmp_path) -> None:
    import io
    import soundfile as sf

    from app.core.meeting import MeetingResult
    from app.core.meeting_cli import _write_training_artifacts

    def wav_bytes(value: float) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, np.full(1600, value, dtype=np.float32), 16000, format="WAV")
        return buffer.getvalue()

    result = MeetingResult(
        filename="meeting.wav",
        duration_s=4.0,
        language="zh",
        segments=[
            Segment(0.0, 1.0, "第一句", speaker="S1", audio=wav_bytes(0.1), raw_text="第一句"),
            Segment(1.4, 2.4, "第二句", speaker="S1", audio=wav_bytes(0.2), raw_text="第二句"),
            Segment(2.5, 3.5, "第三句", speaker="S2", audio=wav_bytes(0.3), raw_text="第三句"),
        ],
    )

    artifacts = _write_training_artifacts(tmp_path, result)
    assert artifacts["candidate_segments"] == 2
    rows = [json.loads(line) for line in (tmp_path / "qwen3-asr-candidates.jsonl").read_text().splitlines()]
    assert rows[0]["text"] == "第一句第二句"
    assert rows[1]["text"] == "第三句"
    detailed = [json.loads(line) for line in (tmp_path / "transcript_segments.jsonl").read_text().splitlines()]
    assert detailed[0]["start"] == 0.0
    assert detailed[0]["end"] == 2.4


def test_training_loader_explains_pending_meeting_samples(tmp_path) -> None:
    import importlib.util

    script_path = Path(__file__).resolve().parents[2] / "scripts/training/train_qwen3_asr.py"
    spec = importlib.util.spec_from_file_location("train_qwen3_asr", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_path = tmp_path / "transcript_segments.jsonl"
    dataset_path.write_text(
        json.dumps({"audio": "missing.wav", "corrected_text": "待审核", "review_status": "pending"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no approved samples.*pending"):
        module.load_dataset(str(dataset_path))


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


@pytest.mark.anyio
async def test_meeting_summary_uses_selected_prompt_template(tmp_path) -> None:
    from app.core.meeting import MeetingPipeline
    from app.core.meeting_prompts import MeetingPromptTemplateStore

    prompts: list[str] = []
    store = MeetingPromptTemplateStore(tmp_path / "templates.json")
    custom = store.create(
        name="测试模板",
        description="测试",
        chunk_system_prompt="你是自定义分段助手。只提取明确决策。",
        merge_system_prompt="你是自定义合并助手。只合并明确决策。",
    )

    class PromptLLM:
        model = "local-test"

        async def stream_chat(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            yield "## 会议决策\n无"

    pipeline = MeetingPipeline(
        llm=PromptLLM(),
        asr=FakeASR(),
        diarization=None,
        prompt_templates_path=tmp_path / "templates.json",
    )
    await pipeline.summarize(
        [Segment(0.0, 1.0, "确定周五发布", speaker="S1")],
        translate=False,
        prompt_template_id=custom.id,
    )

    assert len(prompts) == 1
    assert "自定义分段助手" in prompts[0]
    assert "明确决策" in prompts[0]


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
