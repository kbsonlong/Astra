"""F3: 纪要任务 fan-out —— 并发/超时隔离/失败隔离/进度回调/渲染。"""
from __future__ import annotations

import asyncio
import json
import time

from app.api.meeting_routes import _render_markdown
from app.config import Settings
from app.core.meeting import (
    _ACTION_ITEM_PROMPT,
    _DECISION_PROMPT,
    MeetingPipeline,
    MeetingResult,
    Segment,
)
from app.core.meeting_cli import _note_task_status_updater


class RoutedLLM:
    """按内置 prompt 前缀路由的假 LLM: (延迟秒, 输出或异常)。"""

    model = "local-test"

    def __init__(
        self,
        actions=(0.0, "ok"),
        decisions=(0.0, "ok"),
        summary=(0.0, "ok"),
        translation=(0.0, "ok"),
    ) -> None:
        self.routes = [
            (_ACTION_ITEM_PROMPT, actions),
            (_DECISION_PROMPT, decisions),
        ]
        self._summary = summary
        self._translation = translation

    async def stream_chat(self, messages, **kwargs):
        content = messages[0]["content"]
        if content.startswith("你是专业翻译"):
            delay, outcome = self._translation
        else:
            for prefix, (delay, outcome) in self.routes:
                if content.startswith(prefix):
                    break
            else:
                delay, outcome = self._summary
        await asyncio.sleep(delay)
        if isinstance(outcome, Exception):
            raise outcome
        yield str(outcome)


def _pipeline(llm, timeout: float = 300.0) -> MeetingPipeline:
    class FakeASR:
        async def transcribe(self, audio, filename="speech.wav"):
            return "文本"

    return MeetingPipeline(
        llm=llm,
        asr=FakeASR(),
        diarization=None,
        note_task_timeout_seconds=timeout,
    )


_SEGMENTS = [Segment(0.0, 1.0, "确定周五发布", speaker="S1", segment_id="seg-000001")]


def test_fan_out_runs_concurrently_and_reports_first_done_first() -> None:
    # summary 慢 0.15s, actions/decisions 即回 —— 总耗时接近最慢者而非求和
    llm = RoutedLLM(summary=(0.15, "纪要正文"))
    pipeline = _pipeline(llm)
    events: list[tuple[str, str]] = []

    def on_note_task(task: str, entry: dict) -> None:
        events.append((task, str(entry["status"])))

    started = time.monotonic()
    summary, translation = asyncio.run(
        pipeline.summarize(_SEGMENTS, translate=False, on_note_task=on_note_task)
    )
    wall = time.monotonic() - started

    assert summary == "纪要正文"
    assert translation == ""
    assert wall < 0.4, f"fan-out looks sequential: {wall:.2f}s"
    done_order = [task for task, status in events if status == "done"]
    assert done_order.index("actions") < done_order.index("summary")
    summary_statuses = [status for task, status in events if task == "summary"]
    assert summary_statuses == ["processing", "done"]


def test_timeout_isolated_single_task() -> None:
    llm = RoutedLLM(decisions=(0.5, "决策内容"))
    pipeline = _pipeline(llm, timeout=0.05)
    note_tasks: dict = {}

    summary, _ = asyncio.run(
        pipeline.summarize(_SEGMENTS, translate=False, note_tasks=note_tasks)
    )

    assert summary != ""
    assert note_tasks["summary"]["status"] == "done"
    assert note_tasks["actions"]["status"] == "done"
    assert note_tasks["decisions"]["status"] == "timeout"
    assert "timed out" in str(note_tasks["decisions"]["error"])


def test_failure_isolated_single_task() -> None:
    llm = RoutedLLM(actions=(0.0, RuntimeError("LLM exploded")))
    pipeline = _pipeline(llm)
    note_tasks: dict = {}

    summary, _ = asyncio.run(
        pipeline.summarize(_SEGMENTS, translate=False, note_tasks=note_tasks)
    )

    assert note_tasks["summary"]["status"] == "done"
    assert note_tasks["decisions"]["status"] == "done"
    assert note_tasks["actions"]["status"] == "failed"
    assert "LLM exploded" in str(note_tasks["actions"]["error"])


def test_translation_runs_after_summary_with_own_entry() -> None:
    llm = RoutedLLM(translation=(0.01, "translated text"))
    pipeline = _pipeline(llm)
    order: list[str] = []
    note_tasks: dict = {}

    def on_note_task(task: str, entry: dict) -> None:
        if entry["status"] == "done":
            order.append(task)

    _, translation = asyncio.run(
        pipeline.summarize(
            _SEGMENTS, translate=True, on_note_task=on_note_task, note_tasks=note_tasks
        )
    )

    assert translation == "translated text"
    assert note_tasks["translation"]["status"] == "done"
    assert order.index("summary") < order.index("translation")


def test_render_markdown_note_task_sections() -> None:
    segments = [Segment(0.0, 1.0, "原文", speaker="S1", segment_id="seg-000001")]
    result = MeetingResult(
        filename="m.wav",
        duration_s=1.0,
        language="zh",
        segments=segments,
        summary="纪要正文",
        note_tasks={
            "summary": {"status": "done", "text": "纪要正文", "elapsed_s": 1.0},
            "actions": {"status": "done", "text": "- [ ] 发版 — 张三", "elapsed_s": 1.1},
            "decisions": {
                "status": "timeout",
                "text": "",
                "elapsed_s": 300.0,
                "error": "note task timed out after 300s",
            },
        },
    )
    md = _render_markdown(result, {"engine": "e", "llm_model": "m"})

    assert "## ✅ 待办行动项" in md
    assert "- [ ] 发版 — 张三" in md
    assert "## 🧭 关键决策" in md
    assert "子任务超时" in md


def test_render_markdown_without_note_tasks_has_no_sections() -> None:
    result = MeetingResult(
        filename="m.wav", duration_s=1.0, language="zh", summary="纪要"
    )
    md = _render_markdown(result, {"engine": "e", "llm_model": "m"})
    assert "待办行动项" not in md
    assert "关键决策" not in md


def test_cli_status_updater_writes_note_tasks(tmp_path) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "processing"}), encoding="utf-8"
    )
    updater = _note_task_status_updater(tmp_path)

    updater("actions", {"status": "processing", "text": "", "elapsed_s": 0.0})
    updater("actions", {"status": "done", "text": "- [ ] 发版", "elapsed_s": 1.2})
    updater("summary", {"status": "processing", "text": "", "elapsed_s": 0.0})

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "processing"
    assert payload["note_tasks"]["actions"]["status"] == "done"
    assert payload["note_tasks"]["summary"]["status"] == "processing"


def test_cli_status_updater_bootstraps_missing_status_file(tmp_path) -> None:
    updater = _note_task_status_updater(tmp_path)
    updater("summary", {"status": "done", "text": "x", "elapsed_s": 0.5})

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "processing"
    assert payload["note_tasks"]["summary"]["text"] == "x"


def test_settings_note_task_timeout_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MEETING_NOTE_TASK_TIMEOUT_SECONDS", "42")
    assert Settings.from_env().meeting_note_task_timeout_seconds == 42.0
    monkeypatch.delenv("MEETING_NOTE_TASK_TIMEOUT_SECONDS")
    assert Settings.from_env().meeting_note_task_timeout_seconds == 300.0
