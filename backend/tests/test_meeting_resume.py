"""F2: 断点续跑 —— segments.json checkpoint 读写 / resume 跳过 ASR / routes 透传。"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.core.meeting import (
    CheckpointError,
    MeetingPipeline,
    Segment,
    SummaryStage,
    TranslationStage,
    load_segments_checkpoint,
    segments_from_checkpoint,
    write_segments_checkpoint,
)


def _sample_segments() -> list[Segment]:
    return [
        Segment(0.0, 1.0, "第一句", speaker="S1", speaker_id="c1",
                speaker_name="张三", raw_text="第一句", segment_id="seg-000001"),
        Segment(1.5, 2.5, "第二句", speaker="S2", speaker_id="c2",
                speaker_name="李四", raw_text="第二句", segment_id="seg-000002"),
    ]


def test_checkpoint_round_trip(tmp_path) -> None:
    path = write_segments_checkpoint(
        tmp_path,
        filename="input.m4a",
        duration_s=2.5,
        language="zh",
        segments=_sample_segments(),
    )
    assert path.name == "segments.json"

    payload = load_segments_checkpoint(tmp_path)
    assert payload["version"] == 1
    assert payload["duration_s"] == 2.5
    segments = segments_from_checkpoint(payload)
    assert [(s.segment_id, s.start, s.end, s.text, s.speaker_name) for s in segments] == [
        ("seg-000001", 0.0, 1.0, "第一句", "张三"),
        ("seg-000002", 1.5, 2.5, "第二句", "李四"),
    ]
    assert all(s.audio is None for s in segments)


def test_checkpoint_write_is_atomic_no_tmp_leftover(tmp_path) -> None:
    write_segments_checkpoint(
        tmp_path, filename="m.wav", duration_s=1.0, language="zh",
        segments=_sample_segments(),
    )
    assert [p.name for p in tmp_path.iterdir()] == ["segments.json"]


def test_checkpoint_missing_raises(tmp_path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_segments_checkpoint(tmp_path)


def test_checkpoint_corrupt_raises(tmp_path) -> None:
    (tmp_path / "segments.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="corrupt"):
        load_segments_checkpoint(tmp_path)


def test_checkpoint_bad_version_raises(tmp_path) -> None:
    (tmp_path / "segments.json").write_text(
        json.dumps({"version": 99, "segments": []}), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="version"):
        load_segments_checkpoint(tmp_path)


def test_checkpoint_bad_row_raises(tmp_path) -> None:
    (tmp_path / "segments.json").write_text(
        json.dumps({"version": 1, "segments": ["not-a-dict"]}), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="segment row"):
        segments_from_checkpoint(load_segments_checkpoint(tmp_path))


def _resumable_pipeline(calls: dict[str, int]) -> MeetingPipeline:
    """轻量 pipeline：resume 路径只需 summary/translation 两个 stage。"""

    async def fake_generate(segments, **kwargs):
        calls["summary"] += 1
        return "纪要 [seg-000001]"

    async def fake_translate(summary, **kwargs):
        calls["translate"] += 1
        return "translation"

    pipeline = MeetingPipeline.__new__(MeetingPipeline)
    pipeline.note_task_timeout_seconds = 300.0
    pipeline._stages_default = False  # resume 测试走注入的 stage(旧串行语义)
    pipeline.summary_stage = SummaryStage(fake_generate)
    pipeline.translation_stage = TranslationStage(fake_translate)
    return pipeline


def test_process_resume_skips_audio_and_runs_summary(tmp_path) -> None:
    write_segments_checkpoint(
        tmp_path, filename="meeting.m4a", duration_s=2.5, language="zh",
        segments=_sample_segments(),
    )
    calls = {"summary": 0, "translate": 0}
    pipeline = _resumable_pipeline(calls)

    result = asyncio.run(
        pipeline.process(
            b"should-never-be-touched",
            filename="meeting.m4a",
            checkpoint_dir=tmp_path,
            resume=True,
        )
    )

    assert result.resumed is True
    assert result.summary == "纪要 [seg-000001]"
    assert result.translation == "translation"
    assert result.citations == {"seg-000001": 1}
    assert [s.segment_id for s in result.segments] == ["seg-000001", "seg-000002"]
    assert calls == {"summary": 1, "translate": 1}


def test_process_resume_without_checkpoint_raises(tmp_path) -> None:
    pipeline = _resumable_pipeline({"summary": 0, "translate": 0})
    with pytest.raises(CheckpointError):
        asyncio.run(
            pipeline.process(
                b"audio", filename="meeting.m4a", checkpoint_dir=tmp_path, resume=True
            )
        )


def test_routes_resume_of_missing_checkpoint_returns_400(tmp_path) -> None:
    from app.main import create_app

    app = create_app(
        settings=Settings_for_test(tmp_path),
        enable_pipeline=False,
        enable_meeting=False,
    )
    client = TestClient(app)
    response = client.post(
        "/api/meeting/process",
        data={"resume_of": "20260905-123456-1a2b3c4d"},
    )
    assert response.status_code == 400
    assert "no resume checkpoint" in response.json()["detail"]
    # 早期校验失败，不创建任何 job 目录（meetings 目录也尚未建立）
    assert list(tmp_path.iterdir()) == []


def test_routes_resume_copies_checkpoint_and_passes_flag(tmp_path, monkeypatch) -> None:
    from app.api import meeting_routes
    from app.main import create_app

    meetings = tmp_path / "meetings"
    old_task = "20260905-123456-1a2b3c4d"
    write_segments_checkpoint(
        meetings / old_task, filename="input.m4a", duration_s=1.0,
        language="zh", segments=_sample_segments(),
    )

    settings = Settings_for_test(tmp_path, task_store=tmp_path / "tasks.sqlite3")
    app = create_app(settings=settings, enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)
    observed: list[object] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        observed.extend(args)

        class FakeProc:
            pid = 4321

            async def wait(self):
                return 0

        return FakeProc()

    monkeypatch.setattr(meeting_routes.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    response = client.post(
        "/api/meeting/process",
        data={"resume_of": old_task},
    )

    assert response.status_code == 200, response.text
    new_task = response.json()["task_id"]
    assert new_task != old_task
    assert "--resume" in observed
    # checkpoint 被复制进新 job 目录
    assert (meetings / new_task / "segments.json").is_file()
    # resume 请求不要求上传音频
    assert response.json()["status"] == "processing"


def Settings_for_test(tmp_path, task_store=None):
    from app.config import Settings

    kwargs = {"meeting_output_dir": str(tmp_path / "meetings")}
    if task_store is not None:
        kwargs["task_store_path"] = str(task_store)
    return Settings(**kwargs)
