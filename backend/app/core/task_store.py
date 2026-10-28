"""会议与训练子进程的持久化任务登记。"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    kind: str
    status: str
    pid: int | None
    pgid: int | None
    process_identity: str | None
    output_dir: str
    log_path: str
    started_at: str
    finished_at: str | None
    detail: dict[str, Any]


class TaskStore:
    """SQLite-backed child-process registry resilient to API restarts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('meeting', 'training')),
                status TEXT NOT NULL CHECK(status IN ('processing', 'done', 'failed', 'stopped')),
                pid INTEGER,
                pgid INTEGER,
                process_identity TEXT,
                output_dir TEXT NOT NULL,
                log_path TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                detail TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_kind_status
                ON tasks(kind, status);
            CREATE TABLE IF NOT EXISTS artifact_cleanup_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                bytes_reclaimed INTEGER NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_artifact_cleanup_events_created
                ON artifact_cleanup_events(created_at DESC);
            CREATE TABLE IF NOT EXISTS task_supervisor_lease (
                lease_name TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                expires_at REAL NOT NULL
            );
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "process_identity" not in columns:
            connection.execute("ALTER TABLE tasks ADD COLUMN process_identity TEXT")
        return connection

    def processing_tasks(self) -> list[TaskRecord]:
        """Read active tasks without changing their state."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE status = 'processing' ORDER BY started_at, rowid"
            ).fetchall()
        return [self._record(row) for row in rows]

    def try_acquire_supervisor(
        self, owner_id: str, *, lease_name: str = "default", lease_seconds: float = 30.0
    ) -> bool:
        """Acquire or renew the single database-backed supervisor lease."""
        if not owner_id or lease_seconds <= 0:
            raise ValueError("owner_id and lease_seconds must be valid")
        now = time.time()
        expires_at = now + lease_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_id, expires_at FROM task_supervisor_lease WHERE lease_name = ?",
                (lease_name,),
            ).fetchone()
            if row is not None and row["owner_id"] != owner_id and float(row["expires_at"]) > now:
                return False
            connection.execute(
                "INSERT INTO task_supervisor_lease(lease_name, owner_id, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(lease_name) DO UPDATE SET owner_id=excluded.owner_id, expires_at=excluded.expires_at",
                (lease_name, owner_id, expires_at),
            )
        return True

    def release_supervisor(self, owner_id: str, *, lease_name: str = "default") -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM task_supervisor_lease WHERE lease_name = ? AND owner_id = ?",
                (lease_name, owner_id),
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> TaskRecord:
        try:
            detail = json.loads(row["detail"] or "{}")
        except json.JSONDecodeError:
            detail = {"error": "stored task detail is invalid"}
        return TaskRecord(
            task_id=row["task_id"],
            kind=row["kind"],
            status=row["status"],
            pid=row["pid"],
            pgid=row["pgid"],
            process_identity=row["process_identity"],
            output_dir=row["output_dir"],
            log_path=row["log_path"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            detail=detail if isinstance(detail, dict) else {},
        )

    def register(
        self,
        *,
        task_id: str,
        kind: str,
        status: str,
        pid: int | None,
        pgid: int | None,
        process_identity: str | None = None,
        output_dir: str,
        log_path: str = "",
        detail: dict[str, Any] | None = None,
    ) -> TaskRecord:
        if kind not in {"meeting", "training"}:
            raise ValueError("unsupported task kind")
        if status not in {"processing", "done", "failed", "stopped"}:
            raise ValueError("unsupported task status")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks(task_id, kind, status, pid, pgid, process_identity, output_dir, log_path, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    kind=excluded.kind, status=excluded.status, pid=excluded.pid,
                    pgid=excluded.pgid, process_identity=excluded.process_identity,
                    output_dir=excluded.output_dir,
                    log_path=excluded.log_path, detail=excluded.detail,
                    finished_at=CASE WHEN excluded.status = 'processing' THEN NULL ELSE CURRENT_TIMESTAMP END
                """,
                (
                    task_id, kind, status, pid, pgid, process_identity, output_dir, log_path,
                    json.dumps(detail or {}, ensure_ascii=False),
                ),
            )
        return self.get(task_id)  # type: ignore[return-value]

    def reserve(
        self,
        *,
        task_id: str,
        kind: str,
        max_concurrent: int,
        output_dir: str,
        log_path: str = "",
        detail: dict[str, Any] | None = None,
    ) -> bool:
        """Atomically reserve one processing slot before spawning a child."""
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if kind not in {"meeting", "training"}:
            raise ValueError("unsupported task kind")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE kind = ? AND status = 'processing'",
                (kind,),
            ).fetchone()[0]
            if active >= max_concurrent:
                return False
            connection.execute(
                "INSERT INTO tasks(task_id, kind, status, output_dir, log_path, detail) "
                "VALUES (?, ?, 'processing', ?, ?, ?)",
                (task_id, kind, output_dir, log_path, json.dumps(detail or {}, ensure_ascii=False)),
            )
        return True

    def active_count(self, kind: str) -> int:
        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE kind = ? AND status = 'processing'",
                (kind,),
            ).fetchone()[0])

    def update(
        self,
        task_id: str,
        *,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> TaskRecord:
        if status not in {"processing", "done", "failed", "stopped"}:
            raise ValueError("unsupported task status")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT detail FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(task_id)
            merged = json.loads(existing["detail"] or "{}")
            if not isinstance(merged, dict):
                merged = {}
            if detail:
                merged.update(detail)
            connection.execute(
                "UPDATE tasks SET status = ?, detail = ?, "
                "finished_at = CASE WHEN ? = 'processing' THEN NULL ELSE CURRENT_TIMESTAMP END "
                "WHERE task_id = ?",
                (status, json.dumps(merged, ensure_ascii=False), status, task_id),
            )
        return self.get(task_id)  # type: ignore[return-value]

    def get(self, task_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._record(row) if row is not None else None

    def latest(self, kind: str, *, statuses: tuple[str, ...] | None = None) -> TaskRecord | None:
        query = "SELECT * FROM tasks WHERE kind = ?"
        params: list[object] = [kind]
        if statuses:
            query += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
            params.extend(statuses)
        query += " ORDER BY started_at DESC, rowid DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._record(row) if row is not None else None

    def terminal_tasks(self, kind: str) -> list[TaskRecord]:
        if kind not in {"meeting", "training"}:
            raise ValueError("unsupported task kind")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE kind = ? "
                "AND status IN ('done', 'failed', 'stopped') "
                "ORDER BY COALESCE(finished_at, started_at), rowid",
                (kind,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def record_artifact_cleanup(
        self, *, task_id: str, output_dir: str, bytes_reclaimed: int, reason: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO artifact_cleanup_events "
                "(task_id, output_dir, bytes_reclaimed, reason) VALUES (?, ?, ?, ?)",
                (task_id, output_dir, bytes_reclaimed, reason),
            )

    def artifact_cleanup_events(self, *, limit: int = 100) -> list[dict[str, object]]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id, output_dir, bytes_reclaimed, reason, created_at "
                "FROM artifact_cleanup_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def process_identity(pid: int | None) -> str | None:
        """Return a stable process-start token where the host exposes one."""
        if not pid:
            return None
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="],
                check=True,
                capture_output=True,
                text=True,
                timeout=0.5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        token = result.stdout.strip()
        return token or None

    @staticmethod
    def _elapsed_seconds(started_at: str) -> float:
        try:
            started = datetime.fromisoformat(started_at).replace(tzinfo=timezone.utc)
        except ValueError:
            return 0.0
        return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())

    def terminate(self, task_id: str, *, status: str = "stopped", reason: str) -> TaskRecord:
        record = self.get(task_id)
        if record is None:
            raise KeyError(task_id)
        if record.status != "processing":
            return record
        try:
            if record.kind == "meeting" and record.pgid:
                os.killpg(record.pgid, signal.SIGTERM)
            elif record.pid:
                os.kill(record.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return self.update(task_id, status=status, detail={"error": reason})

    def terminate_and_wait(
        self,
        task_id: str,
        *,
        reason: str,
        grace_seconds: float = 5.0,
        force_seconds: float = 2.0,
    ) -> TaskRecord:
        """Stop a child, confirm exit, and only then publish its terminal state."""
        if grace_seconds < 0 or force_seconds < 0:
            raise ValueError("termination timeouts must not be negative")
        record = self.get(task_id)
        if record is None:
            raise KeyError(task_id)
        if record.status != "processing":
            return record
        if not record.pid:
            return self.update(task_id, status="stopped", detail={"error": reason})
        if record.process_identity:
            current_identity = self.process_identity(record.pid)
            if current_identity is None and self.pid_alive(record.pid):
                return self.update(
                    task_id,
                    status="failed",
                    detail={"error": "cannot verify original process identity", "identity_unverified": True},
                )
            if current_identity is not None and current_identity != record.process_identity:
                return self.update(
                    task_id,
                    status="stopped",
                    detail={"error": "original process is no longer present", "identity_mismatch": True},
                )

        try:
            if record.kind == "meeting" and record.pgid:
                os.killpg(record.pgid, signal.SIGTERM)
            else:
                os.kill(record.pid, signal.SIGTERM)
        except ProcessLookupError:
            return self.update(task_id, status="stopped", detail={"error": reason})

        deadline = time.monotonic() + grace_seconds
        while self.pid_alive(record.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        forced = False
        if self.pid_alive(record.pid):
            forced = True
            try:
                if record.kind == "meeting" and record.pgid:
                    os.killpg(record.pgid, signal.SIGKILL)
                else:
                    os.kill(record.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + force_seconds
            while self.pid_alive(record.pid) and time.monotonic() < deadline:
                time.sleep(0.05)

        if self.pid_alive(record.pid):
            return self.update(
                task_id,
                status="failed",
                detail={"error": f"unable to stop process: {reason}", "forced": forced},
            )
        return self.update(
            task_id,
            status="stopped",
            detail={"error": reason, "forced": forced},
        )

    def reconcile(self, timeouts: dict[str, float] | None = None) -> list[TaskRecord]:
        """Converge stale processing entries after API restart or child exit."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE status = 'processing'"
            ).fetchall()
        return [
            self.reconcile_task(row["task_id"], timeout_seconds=(timeouts or {}).get(row["kind"]))
            for row in rows
        ]

    def reconcile_task(
        self, task_id: str, *, timeout_seconds: float | None = None
    ) -> TaskRecord:
        record = self.get(task_id)
        if record is None:
            raise KeyError(task_id)
        if record.status != "processing":
            return record
        if (
            timeout_seconds is not None
            and self._elapsed_seconds(record.started_at) >= timeout_seconds
        ):
            return self.terminate(
                task_id,
                status="failed",
                reason=f"task timed out after {timeout_seconds:g}s",
            )
        # reserve() creates the processing row before the child is spawned.
        # Give that short hand-off window a chance to register its PID.
        if record.pid is None:
            return record
        if record.process_identity:
            current_identity = self.process_identity(record.pid)
            if current_identity is not None and current_identity != record.process_identity:
                status, detail = self._terminal_status(record)
                return self.update(record.task_id, status=status, detail={**detail, "identity_mismatch": True})
        if self.pid_alive(record.pid):
            return record
        status, detail = self._terminal_status(record)
        return self.update(record.task_id, status=status, detail=detail)

    def _terminal_status(self, record: TaskRecord) -> tuple[str, dict[str, Any]]:
        if record.kind == "meeting":
            status_path = Path(record.output_dir) / "status.json"
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            final_status = payload.get("status")
            if final_status in {"done", "failed"}:
                return str(final_status), {"status_json": payload}
        if record.kind == "training":
            status_path = Path(record.output_dir) / "training-status.json"
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            final_status = payload.get("status")
            if final_status in {"done", "failed"}:
                return str(final_status), {"status_json": payload}
        return "failed", {"error": "process exited without a final status"}
