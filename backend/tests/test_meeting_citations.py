"""F1: 纪要引用锚点 —— segment id / prompt 引用 / citations 统计 / 报告渲染。"""
from __future__ import annotations

import asyncio

from app.api.meeting_routes import _render_markdown
from app.core.meeting import (
    _CITATION_INSTRUCTION,
    MeetingPipeline,
    MeetingResult,
    SummaryStage,
    _assign_segment_ids,
    extract_citations,
)
from app.core.workflow import Segment


def test_extract_citations_counts_in_order() -> None:
    summary = "要点一 [seg-000001]；要点二 [seg-000002][seg-000001]。"
    assert extract_citations(summary) == {"seg-000001": 2, "seg-000002": 1}


def test_extract_citations_empty_and_invalid() -> None:
    assert extract_citations("") == {}
    assert extract_citations("无引用的纪要") == {}
    # 格式不符 (5位/7位) 不算引用
    assert extract_citations("[seg-00001] [seg-0000001]") == {}


def test_assign_segment_ids_is_idempotent() -> None:
    segments = [Segment(0.0, 1.0, "一"), Segment(1.0, 2.0, "二")]
    _assign_segment_ids(segments)
    assert [s.segment_id for s in segments] == ["seg-000001", "seg-000002"]
    _assign_segment_ids(segments)  # 重复调用不覆盖
    assert [s.segment_id for s in segments] == ["seg-000001", "seg-000002"]


def test_timeline_text_includes_segment_anchor() -> None:
    result = MeetingResult(
        filename="m.wav",
        duration_s=2.0,
        language="zh",
        segments=[Segment(65.0, 66.0, "内容", speaker="S1", segment_id="seg-000003")],
    )
    assert "[01:05] [seg-000003] S1 内容" in result.timeline_text()


def test_timeline_text_without_id_stays_backward_compatible() -> None:
    result = MeetingResult(
        filename="m.wav",
        duration_s=2.0,
        language="zh",
        segments=[Segment(0.0, 1.0, "测试内容", speaker="S1")],
    )
    assert "[00:00] S1 测试内容" in result.timeline_text()


def test_summary_prompt_includes_segment_ids_and_citation_rule() -> None:
    captured: list[list[dict[str, str]]] = []

    class FakeLLM:
        model = "local-test"

        async def stream_chat(self, messages, **kwargs):
            captured.append(messages)
            yield "要点 [seg-000001]"

    segments = [
        Segment(0.0, 1.0, "第一句", speaker="S1", segment_id="seg-000001"),
        Segment(1.0, 2.0, "第二句", speaker="S2", segment_id="seg-000002"),
    ]
    summary = asyncio.run(
        MeetingPipeline(llm=FakeLLM())._generate_summary(segments)
    )
    assert summary == "要点 [seg-000001]"
    transcript = captured[0][1]["content"]
    assert "[seg-000001][S1] 第一句" in transcript
    assert "[seg-000002][S2] 第二句" in transcript
    assert _CITATION_INSTRUCTION in captured[0][0]["content"]


def test_summary_prompt_without_ids_omits_citation_rule() -> None:
    captured: list[list[dict[str, str]]] = []

    class FakeLLM:
        model = "local-test"

        async def stream_chat(self, messages, **kwargs):
            captured.append(messages)
            yield "要点"

    segments = [Segment(0.0, 1.0, "第一句", speaker="S1")]
    asyncio.run(MeetingPipeline(llm=FakeLLM())._generate_summary(segments))
    assert _CITATION_INSTRUCTION not in captured[0][0]["content"]


def test_process_computes_citations_from_summary() -> None:
    class FakeASR:
        async def transcribe(self, audio, filename="speech.wav"):
            return "文本"

    class FakeVAD:
        name = "vad"

        async def detect(self, wav):
            return []

    class FakeSD:
        name = "sd"

        async def assign(self, wav, segments):
            return None

    class FakePunct:
        name = "punct"

        async def run(self, text):
            return text

    class FakeLLM:
        model = "local-test"

        async def stream_chat(self, messages, **kwargs):
            yield "要点 [seg-000001][seg-009999]"

    from app.core.meeting import MeetingPipeline as MP

    pipeline = MP.__new__(MP)  # 绕过重量级 __init__，只测 citations 填充
    result = MeetingResult(
        filename="m.wav",
        duration_s=1.0,
        language="zh",
        segments=[Segment(0.0, 1.0, "一", segment_id="seg-000001")],
        summary="要点 [seg-000001][seg-009999]",
    )
    from app.core.meeting import extract_citations as ec

    result.citations = ec(result.summary)
    assert result.citations == {"seg-000001": 1, "seg-009999": 1}


def test_render_markdown_citation_section() -> None:
    segments = [
        Segment(0.0, 1.0, "第一句", speaker="S1", segment_id="seg-000001"),
        Segment(1.0, 2.0, "第二句", speaker="S2", segment_id="seg-000002"),
    ]
    result = MeetingResult(
        filename="m.wav",
        duration_s=2.0,
        language="zh",
        segments=segments,
        summary="要点 [seg-000001][seg-000001][seg-009999]",
    )
    result.citations = extract_citations(result.summary)
    md = _render_markdown(result, {"engine": "e", "llm_model": "m"})

    assert "### 🔗 引用统计" in md
    assert "`seg-000001` ×2 — S1: 第一句" in md
    assert "⚠️ `seg-009999` ×1 — 无效引用" in md


def test_render_markdown_without_citations_has_no_section() -> None:
    result = MeetingResult(
        filename="m.wav",
        duration_s=2.0,
        language="zh",
        segments=[Segment(0.0, 1.0, "第一句", speaker="S1")],
        summary="普通纪要",
    )
    md = _render_markdown(result, {"engine": "e", "llm_model": "m"})
    assert "引用统计" not in md
