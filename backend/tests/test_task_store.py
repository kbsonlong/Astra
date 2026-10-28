import asyncio
import os
import signal
import time

import pytest
from fastapi.testclient import TestClient

from app.api.meeting_routes import _status_payload
from app.config import Qwen3TrainingConfig, Settings
from app.core.task_store import TaskStore
from app.core.training import TrainingManager
from app.main import create_app


def test_task_store_reconciles_dead_meeting_from_status_json(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    out_dir = tmp_path / "meeting"
    out_dir.mkdir()
    (out_dir / "status.json").write_text('{"status":"done","segments":2}', encoding="utf-8")
    store.register(
        task_id="meeting-1", kind="meeting", status="processing", pid=123,
        pgid=123, output_dir=str(out_dir), log_path=str(out_dir / "job.log"),
    )
    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(lambda pid: False))

    record = store.reconcile_task("meeting-1")

    assert record.status == "done"
    assert record.detail["status_json"]["segments"] == 2


def test_task_store_marks_dead_training_failed(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.register(
        task_id="training-1", kind="training", status="processing", pid=123,
        pgid=None, output_dir=str(tmp_path / "model"), log_path="training.log",
    )
    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(lambda pid: False))

    record = store.reconcile_task("training-1")

    assert record.status == "failed"
    assert record.detail["error"] == "process exited without a final status"


def test_task_store_recovers_dead_training_from_worker_status(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "training-status.json").write_text(
        '{"status":"done","checkpoint":"final"}', encoding="utf-8"
    )
    store.register(
        task_id="training-done", kind="training", status="processing", pid=123,
        pgid=None, output_dir=str(output_dir), log_path="training.log",
    )
    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(lambda pid: False))

    record = store.reconcile_task("training-done")

    assert record.status == "done"
    assert record.detail["status_json"]["checkpoint"] == "final"


def test_task_store_rejects_reused_pid_on_reconcile(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.register(
        task_id="meeting-reused", kind="meeting", status="processing", pid=123,
        pgid=123, process_identity="old-start", output_dir=str(tmp_path),
    )
    monkeypatch.setattr(TaskStore, "process_identity", staticmethod(lambda pid: "new-start"))
    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(lambda pid: True))

    record = store.reconcile_task("meeting-reused")

    assert record.status == "failed"
    assert record.detail["identity_mismatch"] is True


def test_task_store_does_not_signal_reused_pid_on_stop(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.register(
        task_id="training-reused", kind="training", status="processing", pid=123,
        pgid=None, process_identity="old-start", output_dir=str(tmp_path),
    )
    monkeypatch.setattr(TaskStore, "process_identity", staticmethod(lambda pid: "new-start"))
    calls: list[int] = []
    monkeypatch.setattr("app.core.task_store.os.kill", lambda pid, sig: calls.append(pid))

    record = store.terminate_and_wait(
        "training-reused", reason="cancelled by administrator", grace_seconds=0, force_seconds=0
    )

    assert calls == []
    assert record.status == "stopped"
    assert record.detail["identity_mismatch"] is True


def test_meeting_status_converges_dead_processing_task(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task_id = "20260905-123456-1a2b3c4d"
    out_dir = tmp_path / task_id
    out_dir.mkdir()
    (out_dir / "status.json").write_text('{"status":"processing"}', encoding="utf-8")
    store.register(
        task_id=task_id, kind="meeting", status="processing", pid=123,
        pgid=123, output_dir=str(out_dir), log_path=str(out_dir / "job.log"),
    )
    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(lambda pid: False))

    payload = _status_payload(tmp_path, task_id, store)

    assert payload["status"] == "failed"
    assert "process exited" in str(payload["error"])
    assert '"status": "failed"' in (out_dir / "status.json").read_text(encoding="utf-8")


def test_lifespan_restores_live_training_pid_view(tmp_path) -> None:
    store_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(store_path)
    store.register(
        task_id="training-live", kind="training", status="processing", pid=os.getpid(),
        pgid=None, output_dir=str(tmp_path / "model"), log_path="training.log",
    )
    app = create_app(
        settings=Settings(task_store_path=str(store_path)),
        enable_pipeline=False,
        enable_meeting=False,
    )

    with TestClient(app) as client:
        status = client.get("/api/training/status")

    assert status.status_code == 200
    assert status.json()["task_id"] == "training-live"
    assert status.json()["status"] == "processing"
    assert status.json()["pid"] == os.getpid()


@pytest.mark.anyio
async def test_training_manager_persists_start_reap_and_stop(tmp_path, monkeypatch) -> None:
    import app.core.training as training

    class FakeProcess:
        pid = 4321
        exitcode: int | None = None

        def start(self) -> None:
            return None

        def join(self) -> None:
            self.exitcode = 0

    class FakeContext:
        def Process(self, **kwargs):
            return FakeProcess()

    store = TaskStore(tmp_path / "tasks.sqlite3")
    manager = TrainingManager(store)
    monkeypatch.setattr(training, "get_context", lambda method: FakeContext())
    monkeypatch.setattr(training.os, "kill", lambda pid, sig: None)

    started = await manager.start(
        Qwen3TrainingConfig(output_dir=str(tmp_path / "model")).validate()
    )
    task_id = str(started["task_id"])
    assert store.get(task_id).status == "processing"

    await asyncio.sleep(0.02)
    assert store.get(task_id).status == "done"

    # Restore a live PID view to exercise stop without a multiprocessing handle.
    alive = True

    def fake_alive(pid: int | None) -> bool:
        return alive and pid == 4321

    def fake_kill(pid: int, sig: int) -> None:
        nonlocal alive
        alive = False

    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(fake_alive))
    monkeypatch.setattr(training.os, "kill", fake_kill)
    store.register(
        task_id="training-stop", kind="training", status="processing", pid=4321,
        pgid=None, output_dir=str(tmp_path / "model"), log_path="training-stop.log",
    )
    manager.restore(store.get("training-stop"))
    stopped = await manager.stop()
    assert stopped["status"] == "stopped"
    assert store.get("training-stop").status == "stopped"


def test_task_store_reserves_kind_slots_atomically(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")

    assert store.reserve(
        task_id="meeting-1", kind="meeting", max_concurrent=1,
        output_dir=str(tmp_path / "meeting-1"),
    )
    assert not store.reserve(
        task_id="meeting-2", kind="meeting", max_concurrent=1,
        output_dir=str(tmp_path / "meeting-2"),
    )
    assert store.reserve(
        task_id="training-1", kind="training", max_concurrent=1,
        output_dir=str(tmp_path / "training-1"),
    )


def test_task_store_supervisor_lease_allows_one_worker_and_failover(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")

    assert store.try_acquire_supervisor("worker-a", lease_seconds=30)
    assert not store.try_acquire_supervisor("worker-b", lease_seconds=30)
    assert store.try_acquire_supervisor("worker-a", lease_seconds=30)

    store.release_supervisor("worker-a")

    assert store.try_acquire_supervisor("worker-b", lease_seconds=30)


def test_task_store_supervisor_lease_expires_for_dead_worker(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")

    assert store.try_acquire_supervisor("worker-a", lease_seconds=0.01)
    time.sleep(0.02)

    assert store.try_acquire_supervisor("worker-b", lease_seconds=30)


def test_task_store_terminates_meeting_process_group(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.register(
        task_id="meeting-1", kind="meeting", status="processing", pid=123,
        pgid=456, output_dir=str(tmp_path), log_path="job.log",
    )
    calls: list[tuple[int, object]] = []
    monkeypatch.setattr("app.core.task_store.os.killpg", lambda pgid, sig: calls.append((pgid, sig)))

    record = store.terminate("meeting-1", reason="cancelled by administrator")

    assert calls and calls[0][0] == 456
    assert record.status == "stopped"
    assert record.detail["error"] == "cancelled by administrator"


def test_task_store_force_kills_when_graceful_stop_does_not_exit(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.register(
        task_id="training-1", kind="training", status="processing", pid=123,
        pgid=None, output_dir=str(tmp_path), log_path="training.log",
    )
    alive = True
    calls: list[int] = []

    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(lambda pid: alive))

    def fake_kill(pid: int, sig: int) -> None:
        nonlocal alive
        calls.append(sig)
        if sig == signal.SIGKILL:
            alive = False

    monkeypatch.setattr("app.core.task_store.os.kill", fake_kill)

    record = store.terminate_and_wait(
        "training-1", reason="cancelled by administrator", grace_seconds=0, force_seconds=0
    )

    assert calls == [signal.SIGTERM, signal.SIGKILL]
    assert record.status == "stopped"
    assert record.detail["forced"] is True


def test_task_store_times_out_live_task(tmp_path, monkeypatch) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.register(
        task_id="meeting-1", kind="meeting", status="processing", pid=123,
        pgid=456, output_dir=str(tmp_path), log_path="job.log",
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE tasks SET started_at = '2000-01-01 00:00:00' WHERE task_id = ?",
            ("meeting-1",),
        )
    monkeypatch.setattr(TaskStore, "pid_alive", staticmethod(lambda pid: True))
    calls: list[int] = []
    monkeypatch.setattr("app.core.task_store.os.killpg", lambda pgid, sig: calls.append(pgid))

    record = store.reconcile_task("meeting-1", timeout_seconds=1)

    assert calls == [456]
    assert record.status == "failed"
    assert record.detail["error"] == "task timed out after 1s"
