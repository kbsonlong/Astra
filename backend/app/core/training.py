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


@dataclass
class TrainingJob:
    task_id: str
    process: object
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
    """每个 API 进程最多运行一个训练子进程。"""

    def __init__(self) -> None:
        self._job: TrainingJob | None = None
        self._lock = asyncio.Lock()

    async def start(self, config: Qwen3TrainingConfig) -> dict[str, object]:
        async with self._lock:
            if self._job and self._job.process.returncode is None:
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
            self._job = TrainingJob(
                task_id=task_id,
                process=process,
                output_dir=str(output_dir),
                log_path=str(log_path),
                started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            asyncio.create_task(self._reap(process))
            return self.status()

    async def _reap(self, process: object) -> None:
        await asyncio.to_thread(process.join)  # type: ignore[attr-defined]

    async def stop(self) -> dict[str, object]:
        async with self._lock:
            if not self._job or self._job.process.exitcode is not None:  # type: ignore[attr-defined]
                raise RuntimeError("当前没有运行中的训练任务")
            os.kill(self._job.process.pid, signal.SIGTERM)  # type: ignore[attr-defined]
            return self.status()

    def status(self) -> dict[str, object]:
        if not self._job:
            return {"status": "idle"}
        code = self._job.process.exitcode  # type: ignore[attr-defined]
        state = "processing" if code is None else ("done" if code == 0 else "failed")
        return {
            "task_id": self._job.task_id,
            "status": state,
            "returncode": code,
            "started_at": self._job.started_at,
            "output_dir": self._job.output_dir,
            "log_path": self._job.log_path,
        }
