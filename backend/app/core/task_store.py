"""会议与训练子进程的持久化任务登记。"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
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
            """
        )
        return connection

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
                INSERT INTO tasks(task_id, kind, status, pid, pgid, output_dir, log_path, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    kind=excluded.kind, status=excluded.status, pid=excluded.pid,
                    pgid=excluded.pgid, output_dir=excluded.output_dir,
                    log_path=excluded.log_path, detail=excluded.detail,
                    finished_at=CASE WHEN excluded.status = 'processing' THEN NULL ELSE CURRENT_TIMESTAMP END
                """,
                (
                    task_id, kind, status, pid, pgid, output_dir, log_path,
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
            and record.pid is not None
            and self._elapsed_seconds(record.started_at) >= timeout_seconds
        ):
            return self.terminate(
                task_id,
                status="failed",
                reason=f"task timed out after {timeout_seconds:g}s",
            )
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
        return "failed", {"error": "process exited without a final status"}
