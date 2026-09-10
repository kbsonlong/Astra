from datetime import datetime, timezone
from pathlib import Path

from app.core.artifact_retention import clean_meeting_artifacts
from app.core.task_store import TaskStore


def _register(
    store: TaskStore,
    *,
    task_id: str,
    directory: Path,
    status: str = "done",
) -> None:
    store.register(
        task_id=task_id,
        kind="meeting",
        status=status,
        pid=None,
        pgid=None,
        output_dir=str(directory),
        log_path=str(directory / "job.log"),
    )


def test_retention_removes_expired_terminal_artifacts_and_audits(tmp_path) -> None:
    base = tmp_path / "meetings"
    expired = base / "expired"
    running = base / "running"
    unregistered = base / "unregistered"
    for directory in (expired, running, unregistered):
        directory.mkdir(parents=True)
        (directory / "audio.bin").write_bytes(b"x" * 12)

    store = TaskStore(tmp_path / "tasks.sqlite3")
    _register(store, task_id="expired", directory=expired)
    _register(store, task_id="running", directory=running, status="processing")
    with store._connect() as connection:
        connection.execute(
            "UPDATE tasks SET finished_at = '2000-01-01 00:00:00' WHERE task_id = 'expired'"
        )

    result = clean_meeting_artifacts(
        base_dir=base,
        task_store=store,
        retention_days=30,
        max_bytes=10_000,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert result.removed_task_ids == ("expired",)
    assert result.bytes_reclaimed == 12
    assert not expired.exists()
    assert running.is_dir()
    assert unregistered.is_dir()
    assert store.artifact_cleanup_events() == [{
        "task_id": "expired",
        "output_dir": str(expired.resolve()),
        "bytes_reclaimed": 12,
        "reason": "retention_period_expired",
        "created_at": store.artifact_cleanup_events()[0]["created_at"],
    }]


def test_retention_removes_oldest_terminal_artifacts_for_storage_budget(tmp_path) -> None:
    base = tmp_path / "meetings"
    old = base / "old"
    new = base / "new"
    for directory in (old, new):
        directory.mkdir(parents=True)
        (directory / "audio.bin").write_bytes(b"x" * 10)
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _register(store, task_id="old", directory=old)
    _register(store, task_id="new", directory=new)
    with store._connect() as connection:
        connection.execute("UPDATE tasks SET finished_at = '2000-01-01 00:00:00' WHERE task_id = 'old'")
        connection.execute("UPDATE tasks SET finished_at = '2025-12-31 00:00:00' WHERE task_id = 'new'")

    result = clean_meeting_artifacts(
        base_dir=base,
        task_store=store,
        retention_days=10_000,
        max_bytes=12,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert result.removed_task_ids == ("old",)
    assert not old.exists()
    assert new.is_dir()
    assert store.artifact_cleanup_events()[0]["reason"] == "storage_budget_exceeded"


def test_retention_never_deletes_registered_path_outside_meeting_root(tmp_path) -> None:
    base = tmp_path / "meetings"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "audio.bin").write_bytes(b"x" * 10)
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _register(store, task_id="outside", directory=outside)
    with store._connect() as connection:
        connection.execute("UPDATE tasks SET finished_at = '2000-01-01 00:00:00' WHERE task_id = 'outside'")

    result = clean_meeting_artifacts(
        base_dir=base,
        task_store=store,
        retention_days=1,
        max_bytes=1,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert result.removed_task_ids == ()
    assert outside.is_dir()
    assert store.artifact_cleanup_events() == []


def test_lifespan_runs_meeting_artifact_retention(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    base = tmp_path / "meetings"
    expired = base / "expired"
    expired.mkdir(parents=True)
    (expired / "audio.bin").write_bytes(b"x")
    task_store_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(task_store_path)
    _register(store, task_id="expired", directory=expired)
    with store._connect() as connection:
        connection.execute(
            "UPDATE tasks SET finished_at = '2000-01-01 00:00:00' WHERE task_id = 'expired'"
        )
    app = create_app(
        settings=Settings(
            meeting_output_dir=str(base),
            task_store_path=str(task_store_path),
            meeting_artifact_retention_days=1,
            meeting_artifact_max_bytes=1_000,
        ),
        enable_pipeline=False,
        enable_meeting=False,
    )

    with TestClient(app):
        pass

    assert not expired.exists()
    assert store.artifact_cleanup_events()[0]["task_id"] == "expired"
