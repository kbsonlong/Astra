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


def test_prompt_templates_endpoint_and_invalid_selection(tmp_path) -> None:
    from app.main import create_app

    app = create_app(
        settings=Settings(meeting_output_dir=str(tmp_path)),
        enable_pipeline=False,
        enable_meeting=False,
    )
    client = TestClient(app)

    response = client.get("/api/meeting/prompt-templates")
    assert response.status_code == 200, response.text
    assert response.json()["default"] == "standard"

    invalid = client.post(
        "/api/meeting/process",
        files={"file": ("meeting.wav", b"audio", "audio/wav")},
        data={"prompt_template": "unknown"},
    )
    assert invalid.status_code == 400
    assert "unknown meeting prompt template" in invalid.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_custom_prompt_template_store_round_trip(tmp_path) -> None:
    from app.core.meeting_prompts import MeetingPromptTemplateStore

    store = MeetingPromptTemplateStore(tmp_path / "templates.json")
    created = store.create(
        name="产品评审",
        description="记录产品评审结果",
        chunk_system_prompt="提取明确的评审结论。",
        merge_system_prompt="合并评审结论并去重。",
    )
    assert created.id.startswith("custom-")
    assert store.get(created.id).name == "产品评审"

    updated = store.update(
        created.id,
        name="产品评审更新",
        description="记录最终评审结果",
        chunk_system_prompt="只提取明确的评审结论。",
        merge_system_prompt="合并最终评审结论并去重。",
    )
    assert store.get(created.id) == updated
    assert any(item.id == created.id for item in store.list_templates())

    store.delete(created.id)
    with pytest.raises(ValueError, match="unknown meeting prompt template"):
        store.get(created.id)


def test_process_rejects_oversized_upload_without_creating_task(tmp_path) -> None:
    from app.main import create_app

    app = create_app(
        settings=Settings(
            meeting_output_dir=str(tmp_path), meeting_max_upload_bytes=3
        ),
        enable_pipeline=False,
        enable_meeting=False,
    )
    response = TestClient(app).post(
        "/api/meeting/process",
        files={"file": ("meeting.wav", b"four", "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "file too large (max 3 bytes)"
    assert list(tmp_path.iterdir()) == []


def test_custom_prompt_template_api_crud_and_builtin_is_read_only(tmp_path) -> None:
    from app.main import create_app

    settings = Settings(
        meeting_output_dir=str(tmp_path / "meetings"),
        meeting_prompt_templates_path=str(tmp_path / "templates.json"),
    )
    app = create_app(settings=settings, enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)
    body = {
        "name": "产品评审",
        "description": "只记录评审结果",
        "chunk_system_prompt": "提取明确的评审结论。",
        "merge_system_prompt": "合并评审结论并去重。",
    }

    created = client.post("/api/meeting/prompt-templates", json=body)
    assert created.status_code == 200
    item = created.json()["item"]
    assert item["id"].startswith("custom-")
    assert item["editable"] is True

    body["name"] = "产品评审更新"
    updated = client.put(f"/api/meeting/prompt-templates/{item['id']}", json=body)
    assert updated.status_code == 200
    assert updated.json()["item"]["name"] == "产品评审更新"

    listed = client.get("/api/meeting/prompt-templates")
    assert any(template["name"] == "产品评审更新" for template in listed.json()["items"])

    builtin = client.put("/api/meeting/prompt-templates/standard", json=body)
    assert builtin.status_code == 400
    assert "read-only" in builtin.json()["detail"]

    deleted = client.delete(f"/api/meeting/prompt-templates/{item['id']}")
    assert deleted.status_code == 200


def test_process_forwards_custom_prompt_template_to_worker(tmp_path, monkeypatch) -> None:
    from app.api import meeting_routes
    from app.main import create_app

    settings = Settings(
        meeting_output_dir=str(tmp_path / "meetings"),
        meeting_prompt_templates_path=str(tmp_path / "templates.json"),
    )
    app = create_app(settings=settings, enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)
    template = client.post(
        "/api/meeting/prompt-templates",
        json={
            "name": "周会",
            "description": "测试",
            "chunk_system_prompt": "提取周会结论。",
            "merge_system_prompt": "合并周会结论。",
        },
    ).json()["item"]
    observed: list[object] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        observed.extend(args)
        return object()

    monkeypatch.setattr(meeting_routes.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    response = client.post(
        "/api/meeting/process",
        files={"file": ("meeting.wav", b"audio", "audio/wav")},
        data={"prompt_template": template["id"]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["prompt_template"] == template["id"]
    assert "--prompt-template" in observed
    assert template["id"] in observed
    assert "--prompt-templates-path" in observed
    assert str(tmp_path / "templates.json") in observed


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
    md = _render_markdown(
        result,
        {
            "topic": "成本优化",
            "engine": ENGINE_LABEL,
            "llm_model": "Qwen3-8B",
            "prompt_template_name": "标准纪要",
        },
    )

    assert ENGINE_LABEL in md                      # 真实引擎链
    assert "成本优化" in md
    assert "提示词模板: 标准纪要" in md
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


def test_training_review_paginates_and_batch_updates_segments(tmp_path) -> None:
    task_id = "20260905-123456-1a2b3c4d"
    out_dir = tmp_path / task_id
    out_dir.mkdir()
    rows = [
        {
            "segment_id": f"seg-{index:06d}",
            "start": float(index),
            "end": float(index + 1),
            "raw_text": f"原始 {index}",
            "corrected_text": f"原始 {index}",
            "text": f"原始 {index}",
            "review_status": "pending",
        }
        for index in range(1, 4)
    ]
    for name in ("transcript_segments.jsonl", "qwen3-asr-candidates.jsonl"):
        (out_dir / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    from app.main import create_app

    app = create_app(settings=Settings(meeting_output_dir=str(tmp_path)), enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)

    page = client.get(f"/api/meeting/{task_id}/training-data?page=2&page_size=2")
    assert page.status_code == 200
    assert page.json()["pagination"] == {
        "page": 2,
        "page_size": 2,
        "total": 3,
        "total_pages": 2,
        "has_next": False,
        "has_prev": True,
    }
    assert [item["segment_id"] for item in page.json()["items"]] == ["seg-000003"]

    batch = client.post(
        f"/api/meeting/{task_id}/training-data/batch-review",
        json={
            "review_status": "approved",
            "items": [
                {"segment_id": "seg-000001", "corrected_text": "确认一"},
                {"segment_id": "seg-000002", "corrected_text": "确认二"},
            ],
        },
    )
    assert batch.status_code == 200
    assert batch.json()["updated_count"] == 2
    assert batch.json()["counts"] == {"pending": 1, "approved": 2, "rejected": 0}

    for name in ("transcript_segments.jsonl", "qwen3-asr-candidates.jsonl"):
        saved = [json.loads(line) for line in (out_dir / name).read_text(encoding="utf-8").splitlines()]
        assert saved[0]["text"] == "确认一"
        assert saved[1]["text"] == "确认二"
        assert all(row["review_status"] == ("approved" if row["segment_id"] != "seg-000003" else "pending") for row in saved)
