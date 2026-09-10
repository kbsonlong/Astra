"""后台 Qwen3-ASR 训练任务管理。"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from ..config import Qwen3TrainingConfig
from .task_store import TaskRecord, TaskStore


@dataclass
class TrainingJob:
    task_id: str
    process: object | None
    pid: int | None
    output_dir: str
    log_path: str
    started_at: str


def _training_worker(config_data: dict[str, object], log_path: str) -> None:
    """子进程入口；训练本身通过 mlx-tune SDK 调用。"""
    with Path(log_path).open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            repo_root = Path(__file__).resolve().parents[3]
            sys.path.insert(0, str(repo_root))
            sys.path.insert(0, str(repo_root / "backend"))
            from scripts.training.train_qwen3_asr import run_training

            run_training(Qwen3TrainingConfig.from_mapping(config_data))


class TrainingManager:
    """每个 API 进程最多运行一个训练子进程，并持久化其控制视图。"""

    def __init__(self, task_store: TaskStore | None = None) -> None:
        self._job: TrainingJob | None = None
        self._task_store = task_store
        self._lock = asyncio.Lock()

    @staticmethod
    def _is_alive(pid: int | None) -> bool:
        return TaskStore.pid_alive(pid)

    def restore(self, record: TaskRecord) -> None:
        """Restore a running process view after API restart (without Process handle)."""
        if record.kind != "training" or record.status != "processing":
            return
        if self._is_alive(record.pid):
            self._job = TrainingJob(
                task_id=record.task_id,
                process=None,
                pid=record.pid,
                output_dir=record.output_dir,
                log_path=record.log_path,
                started_at=record.started_at,
            )

    def _running(self) -> bool:
        if self._job is None:
            return False
        if self._job.process is not None:
            return self._job.process.exitcode is None  # type: ignore[attr-defined]
        return self._is_alive(self._job.pid)

    async def start(self, config: Qwen3TrainingConfig) -> dict[str, object]:
        async with self._lock:
            if self._running() and self._job:
                raise RuntimeError(f"训练任务已在运行: {self._job.task_id}")

            output_dir = Path(config.output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            task_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            log_path = output_dir / f"training-{task_id}.log"
            process = get_context("spawn").Process(
                target=_training_worker,
                args=(config.to_dict(), str(log_path)),
            )
            process.start()
            pid = process.pid
            self._job = TrainingJob(
                task_id=task_id,
                process=process,
                pid=pid,
                output_dir=str(output_dir),
                log_path=str(log_path),
                started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            if self._task_store is not None:
                self._task_store.register(
                    task_id=task_id, kind="training", status="processing", pid=pid,
                    pgid=None, output_dir=str(output_dir), log_path=str(log_path),
                )
            asyncio.create_task(self._reap(process, task_id))
            return self.status()

    async def _reap(self, process: object, task_id: str) -> None:
        await asyncio.to_thread(process.join)  # type: ignore[attr-defined]
        code = process.exitcode  # type: ignore[attr-defined]
        if self._task_store is not None:
            existing = self._task_store.get(task_id)
            if existing is not None and existing.status == "stopped":
                return
            self._task_store.update(
                task_id,
                status="done" if code == 0 else "failed",
                detail={"returncode": code},
            )

    async def stop(self) -> dict[str, object]:
        async with self._lock:
            if not self._running() or not self._job or not self._job.pid:
                raise RuntimeError("当前没有运行中的训练任务")
            os.kill(self._job.pid, signal.SIGTERM)
            if self._task_store is not None:
                self._task_store.update(self._job.task_id, status="stopped")
            return self.status()

    def status(self) -> dict[str, object]:
        if not self._job:
            return {"status": "idle"}
        record = self._task_store.get(self._job.task_id) if self._task_store else None
        process = self._job.process
        code = process.exitcode if process is not None else None  # type: ignore[attr-defined]
        state = record.status if record is not None else (
            "processing" if self._running() else ("done" if code == 0 else "failed")
        )
        return {
            "task_id": self._job.task_id,
            "status": state,
            "returncode": code if record is None else record.detail.get("returncode", code),
            "pid": self._job.pid,
            "started_at": self._job.started_at,
            "output_dir": self._job.output_dir,
            "log_path": self._job.log_path,
        }
