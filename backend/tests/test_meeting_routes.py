"""meeting_routes 单元测试: task_id / job_dir / 渲染 / 状态查询。"""
import pytest
from fastapi.testclient import TestClient

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


def test_status_endpoint_404_and_400(tmp_path) -> None:
    settings = Settings(meeting_output_dir=str(tmp_path))
    from app.main import create_app

    app = create_app(settings=settings, enable_pipeline=False, enable_meeting=False)
    client = TestClient(app)

    # 格式非法 -> 400
    r = client.get("/api/meeting/abc")
    assert r.status_code == 400
    # 格式合法但不存在 -> 404
    r = client.get("/api/meeting/20260905-123456-1a2b3c4d")
    assert r.status_code == 404
