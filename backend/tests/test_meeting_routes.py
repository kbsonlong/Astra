"""meeting_routes 单元测试: task_id / job_dir / 渲染 / WebSocket 状态事件。"""
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.meeting_routes import (
    ENGINE_LABEL,
    _job_dir,
    _new_task_id,
    _render_markdown,
)
from app.config import Settings
from app.core.meeting import MeetingResult, Segment


def test_new_task_id_unique_and_formatted() -> None:
    ids = {_new_task_id() for _ in range(50)}
    assert len(ids) == 50  # uuid 后缀保证唯一
    for tid in ids:
        assert len(tid) == 8 + 1 + 6 + 1 + 8  # 20260905-123456-1a2b3c4d
        assert tid[8] == "-" and tid[15] == "-"


def test_job_dir_rejects_traversal_and_bad_format(tmp_path) -> None:
    base = tmp_path / "meetings"
    base.mkdir()

    good = _job_dir(base, "20260905-123456-1a2b3c4d")
    assert good == base / "20260905-123456-1a2b3c4d"

    with pytest.raises(Exception):  # HTTPException(400)
        _job_dir(base, "../../etc/passwd")
    with pytest.raises(Exception):
        _job_dir(base, "20260905-123456")
    with pytest.raises(Exception):
        _job_dir(base, "abc")


def test_render_markdown_uses_real_engine_label() -> None:
    seg = Segment(0.0, 5.0, "测试内容", speaker="S1")
    result = MeetingResult(
        filename="meet.m4a",
        duration_s=300.0,
        language="zh",
        segments=[seg],
        summary="# 摘要\n要点。",
        translation="",
    )
    md = _render_markdown(result, {"topic": "成本优化", "engine": ENGINE_LABEL, "llm_model": "Qwen3-8B"})

    assert ENGINE_LABEL in md                      # 真实引擎链
    assert "whisper-large-v3-turbo" not in md       # 已弃用标注不应出现
    assert "spectralcluster" not in md
    assert "成本优化" in md
    assert "[00:00] S1 测试内容" in md


def test_meeting_events_rejects_missing_or_invalid_task(tmp_path) -> None:
    settings = Settings(meeting_output_dir=str(tmp_path))
    from app.main import create_app

    app = create_app(settings=settings, enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)

    with client.websocket_connect("/api/meeting/abc/events") as websocket:
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["message"] == "invalid task_id format"
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    with client.websocket_connect("/api/meeting/20260905-123456-1a2b3c4d/events") as websocket:
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["message"] == "task not found"
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()


def test_meeting_events_pushes_done_payload(tmp_path) -> None:
    task_id = "20260905-123456-1a2b3c4d"
    out_dir = tmp_path / task_id
    out_dir.mkdir()
    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "done",
                "duration_s": 65.5,
                "segments": 2,
                "speakers": ["S1"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text("# 摘要\n完成", encoding="utf-8")
    (out_dir / "transcript.txt").write_text("[00:00] S1 测试\n[00:02] S1 完成", encoding="utf-8")

    settings = Settings(meeting_output_dir=str(tmp_path))
    from app.main import create_app

    app = create_app(settings=settings, enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)

    with client.websocket_connect(f"/api/meeting/{task_id}/events") as websocket:
        event = websocket.receive_json()
        assert event["type"] == "meeting_status"
        assert event["status"] == "done"
        assert event["duration_s"] == 65.5
        assert event["segments"] == 2
        assert event["speakers"] == ["S1"]
        assert event["summary_preview"] == "# 摘要\n完成"
        assert event["transcript_preview"] == "[00:00] S1 测试\n[00:02] S1 完成"
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()


def test_training_review_updates_detail_and_candidate_jsonl(tmp_path) -> None:
    task_id = "20260905-123456-1a2b3c4d"
    out_dir = tmp_path / task_id
    clips_dir = out_dir / "asr_clips"
    clips_dir.mkdir(parents=True)
    row = {
        "segment_id": "seg-000001",
        "start": 0.0,
        "end": 1.0,
        "raw_text": "冷资源中心",
        "corrected_text": "冷资源中心",
        "text": "冷资源中心",
        "review_status": "pending",
        "audio": str((clips_dir / "seg-000001.wav").resolve()),
    }
    for name in ("transcript_segments.jsonl", "qwen3-asr-candidates.jsonl"):
        (out_dir / name).write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (clips_dir / "seg-000001.wav").write_bytes(b"RIFF")

    from app.main import create_app

    app = create_app(settings=Settings(meeting_output_dir=str(tmp_path)), enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)

    review = client.get(f"/api/meeting/{task_id}/training-data")
    assert review.status_code == 200
    assert review.json()["counts"] == {"pending": 1, "approved": 0, "rejected": 0}
    assert review.json()["items"][0]["audio_url"].endswith("/seg-000001/audio")

    response = client.patch(
        f"/api/meeting/{task_id}/training-data/seg-000001",
        json={"corrected_text": "云资源中心", "review_status": "approved"},
    )
    assert response.status_code == 200
    assert response.json()["item"]["review_status"] == "approved"
    for name in ("transcript_segments.jsonl", "qwen3-asr-candidates.jsonl"):
        saved = json.loads((out_dir / name).read_text(encoding="utf-8"))
        assert saved["text"] == "云资源中心"
        assert saved["review_status"] == "approved"

    audio = client.get(f"/api/meeting/{task_id}/training-data/seg-000001/audio")
    assert audio.status_code == 200
    assert audio.content == b"RIFF"
