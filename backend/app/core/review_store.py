"""Transactional persistence for reviewed transcript segments."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ReviewConflictError(RuntimeError):
    pass


class ReviewStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS review_segments ("
            "task_id TEXT NOT NULL, segment_id TEXT NOT NULL, payload TEXT NOT NULL, "
            "review_status TEXT NOT NULL CHECK(review_status IN ('pending','approved','rejected')), "
            "version INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "PRIMARY KEY(task_id, segment_id))"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_segments_task_status "
            "ON review_segments(task_id, review_status)"
        )
        return connection

    def import_if_empty(self, task_id: str, rows: list[dict[str, object]]) -> None:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM review_segments WHERE task_id = ? LIMIT 1", (task_id,)
            ).fetchone()
            if existing is not None:
                return
            connection.executemany(
                "INSERT INTO review_segments(task_id, segment_id, payload, review_status) VALUES (?, ?, ?, ?)",
                [
                    (task_id, str(row["segment_id"]), json.dumps(row, ensure_ascii=False),
                     str(row.get("review_status", "pending")))
                    for row in rows
                ],
            )

    def has_task(self, task_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM review_segments WHERE task_id = ? LIMIT 1", (task_id,)
            ).fetchone() is not None

    def get(self, task_id: str, segment_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, version FROM review_segments "
                "WHERE task_id = ? AND segment_id = ?",
                (task_id, segment_id),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        payload["version"] = row["version"]
        return payload

    def page(self, task_id: str, *, page: int, page_size: int) -> tuple[list[dict[str, object]], dict[str, int]]:
        with self._connect() as connection:
            total = int(connection.execute(
                "SELECT COUNT(*) FROM review_segments WHERE task_id = ?", (task_id,)
            ).fetchone()[0])
            counts = {status: int(connection.execute(
                "SELECT COUNT(*) FROM review_segments WHERE task_id = ? AND review_status = ?", (task_id, status)
            ).fetchone()[0]) for status in ("pending", "approved", "rejected")}
            rows = connection.execute(
                "SELECT payload, version FROM review_segments WHERE task_id = ? "
                "ORDER BY segment_id LIMIT ? OFFSET ?", (task_id, page_size, (page - 1) * page_size)
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["version"] = row["version"]
            result.append(payload)
        return result, {"total": total, **counts}

    def update_many(self, task_id: str, updates: list[dict[str, object]], review_status: str) -> None:
        if review_status not in {"pending", "approved", "rejected"}:
            raise ValueError("unsupported review status")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for item in updates:
                row = connection.execute(
                    "SELECT payload, version FROM review_segments WHERE task_id = ? AND segment_id = ?",
                    (task_id, item["segment_id"]),
                ).fetchone()
                if row is None:
                    raise KeyError(str(item["segment_id"]))
                if row["version"] != item["version"]:
                    raise ReviewConflictError(str(item["segment_id"]))
                payload = json.loads(row["payload"])
                corrected = str(item["corrected_text"]).strip()
                payload.update({"text": corrected, "corrected_text": corrected, "review_status": review_status, "correction_source": "human"})
                connection.execute(
                    "UPDATE review_segments SET payload = ?, review_status = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP "
                    "WHERE task_id = ? AND segment_id = ? AND version = ?",
                    (json.dumps(payload, ensure_ascii=False), review_status, task_id, item["segment_id"], item["version"]),
                )

    def export_rows(self, task_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM review_segments WHERE task_id = ? ORDER BY segment_id", (task_id,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def export_jsonl(self, task_id: str, path: str | Path) -> int:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.export_rows(task_id)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            + ("\n" if rows else ""),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return len(rows)
