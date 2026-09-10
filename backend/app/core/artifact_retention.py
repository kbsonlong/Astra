"""Safe retention enforcement for completed meeting artifacts."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .task_store import TaskRecord, TaskStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupResult:
    removed_task_ids: tuple[str, ...]
    bytes_reclaimed: int


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _safe_task_directory(base: Path, record: TaskRecord) -> Path | None:
    try:
        directory = Path(record.output_dir).expanduser().resolve()
    except OSError:
        return None
    if base not in directory.parents or not directory.is_dir():
        return None
    return directory


def clean_meeting_artifacts(
    *,
    base_dir: str | Path,
    task_store: TaskStore,
    retention_days: int,
    max_bytes: int,
    now: datetime | None = None,
) -> CleanupResult:
    """Remove only registered terminal meeting directories.

    Age-based removal happens first; remaining oldest terminal directories are
    removed only while the complete meeting root exceeds ``max_bytes``.
    ``processing`` tasks, unregistered directories, and paths outside the
    configured root are never candidates.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1")
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        return CleanupResult((), 0)

    current_time = now or datetime.now(timezone.utc)
    expiry = current_time - timedelta(days=retention_days)
    candidates: list[tuple[TaskRecord, Path, datetime]] = []
    for record in task_store.terminal_tasks("meeting"):
        directory = _safe_task_directory(base, record)
        if directory is not None:
            candidates.append((record, directory, _parse_timestamp(record.finished_at or record.started_at)))
    candidates.sort(key=lambda candidate: candidate[2])

    removed: list[str] = []
    reclaimed = 0

    def remove(record: TaskRecord, directory: Path, reason: str) -> bool:
        nonlocal reclaimed
        # Re-read immediately before deletion to defend against an unexpected
        # state transition by another API process.
        latest = task_store.get(record.task_id)
        if latest is None or latest.status == "processing":
            return False
        checked = _safe_task_directory(base, latest)
        if checked is None or checked != directory:
            return False
        size = _tree_size(directory)
        try:
            shutil.rmtree(directory)
        except OSError:
            return False
        task_store.record_artifact_cleanup(
            task_id=latest.task_id,
            output_dir=str(directory),
            bytes_reclaimed=size,
            reason=reason,
        )
        logger.info(
            "removed meeting artifacts task_id=%s reason=%s bytes_reclaimed=%s",
            latest.task_id,
            reason,
            size,
        )
        removed.append(latest.task_id)
        reclaimed += size
        return True

    for record, directory, finished_at in candidates:
        if finished_at < expiry:
            remove(record, directory, "retention_period_expired")

    total = _tree_size(base)
    for record, directory, _ in candidates:
        if total <= max_bytes:
            break
        if record.task_id in removed:
            continue
        if remove(record, directory, "storage_budget_exceeded"):
            total = _tree_size(base)

    return CleanupResult(tuple(removed), reclaimed)
