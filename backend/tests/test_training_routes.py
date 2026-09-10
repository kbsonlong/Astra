import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.core.task_store import TaskStore
from app.main import create_app


def test_training_datasets_returns_only_meetings_with_approved_rows(tmp_path) -> None:
    approved_dir = tmp_path / "20260909-120000-a1b2c3d4"
    approved_dir.mkdir()
    dataset = approved_dir / "qwen3-asr-candidates.jsonl"
    dataset.write_text(
        "\n".join([
            json.dumps({"segment_id": "seg-000001", "review_status": "approved"}),
            json.dumps({"segment_id": "seg-000002", "review_status": "pending"}),
        ]) + "\n",
        encoding="utf-8",
    )
    pending_dir = tmp_path / "20260909-120001-a1b2c3d4"
    pending_dir.mkdir()
    (pending_dir / "qwen3-asr-candidates.jsonl").write_text(
        json.dumps({"segment_id": "seg-000001", "review_status": "pending"}) + "\n",
        encoding="utf-8",
    )

    app = create_app(settings=Settings(meeting_output_dir=str(tmp_path)), enable_pipeline=False, enable_meeting=False)
    response = TestClient(app).get("/api/training/datasets")

    assert response.status_code == 200
    assert response.json()["items"] == [{
        "id": "20260909-120000-a1b2c3d4",
        "task_id": "20260909-120000-a1b2c3d4",
        "path": str(dataset.resolve()),
        "approved_count": 1,
        "total_count": 2,
        "updated_at": response.json()["items"][0]["updated_at"],
    }]


def test_training_start_rejects_persisted_concurrency_limit(tmp_path) -> None:
    task_store_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(task_store_path)
    store.reserve(
        task_id="existing-training", kind="training", max_concurrent=1,
        output_dir=str(tmp_path / "model"),
    )
    app = create_app(
        settings=Settings(
            task_store_path=str(task_store_path),
            training_max_concurrent_jobs=1,
        ),
        enable_pipeline=False,
        enable_meeting=False,
    )

    response = TestClient(app).post("/api/training/start")

    assert response.status_code == 429
    assert response.json()["detail"] == "training concurrency limit reached"
