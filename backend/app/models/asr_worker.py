"""固定线程的实时 ASR worker，隔离 MLX 同步推理与 API 事件循环。"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from .asr_client import ASRClientError

logger = logging.getLogger(__name__)


class ASRTranscriber(Protocol):
    def is_ready(self) -> bool: ...

    def transcribe(self, audio: bytes, filename: str = "speech.wav") -> Awaitable[str]: ...


@dataclass
class _Job:
    audio: bytes
    filename: str
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[str]


class AsrWorkerClient:
    """Run all realtime ASR calls on one stable OS thread.

    MLX streams and its model cache are thread-local. A worker thread therefore
    preserves the cache while the FastAPI event loop awaits a Future instead of
    running model code itself. A timeout cannot preempt a running MLX call; it
    only drops queued work or prevents a late result from being delivered.
    """

    def __init__(
        self,
        client: ASRTranscriber,
        *,
        max_queue: int = 8,
        request_timeout_seconds: float = 120.0,
        shutdown_timeout_seconds: float = 5.0,
        name: str = "astra-asr-worker",
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be at least 1")
        if request_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("worker timeouts must be greater than 0")
        self._client = client
        self._queue: queue.Queue[_Job | None] = queue.Queue(maxsize=max_queue)
        self._max_queue = max_queue
        self._request_timeout_seconds = request_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._name = name
        self._thread: threading.Thread | None = None
        self._current: _Job | None = None
        self._stopping = False
        self._failed: BaseException | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._stopping:
                raise ASRClientError("asr worker is stopping")
            if self._failed is not None:
                raise ASRClientError("asr worker has failed") from self._failed
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=True,
            )
            self._thread.start()

    def is_ready(self) -> bool:
        if self._stopping or self._failed is not None:
            return False
        ready = getattr(self._client, "is_ready", None)
        return bool(ready and ready())

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def max_queue(self) -> int:
        return self._max_queue

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        if not audio:
            raise ASRClientError("audio must not be empty")
        self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        job = _Job(audio, filename, loop, future)
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            raise ASRClientError("asr worker queue is full") from exc
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self._request_timeout_seconds
            )
        except TimeoutError as exc:
            future.cancel()
            raise ASRClientError(
                f"asr worker request timed out after {self._request_timeout_seconds:g}s"
            ) from exc

    async def aclose(self) -> None:
        """Reject queued calls and request shutdown without preempting MLX."""
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
            thread = self._thread
            queued: list[_Job] = []
            while True:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    queued.append(job)
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # defensive: draining above should make room
                pass
            current = self._current
        for job in queued:
            self._deliver_error(job, ASRClientError("asr worker stopped"))
        if current is not None:
            self._deliver_error(current, ASRClientError("asr worker stopped"))
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, self._shutdown_timeout_seconds)

    def _run(self) -> None:
        try:
            while True:
                job = self._queue.get()
                if job is None:
                    return
                if job.future.cancelled():
                    continue
                self._current = job
                try:
                    result = asyncio.run(self._client.transcribe(job.audio, job.filename))
                except Exception as exc:
                    self._deliver_error(job, exc)
                else:
                    self._deliver_result(job, result)
                finally:
                    self._current = None
        except BaseException as exc:  # pragma: no cover - defensive thread guard
            logger.exception("ASR worker thread crashed")
            self._failed = exc
            self._fail_pending(ASRClientError("asr worker crashed"))

    def _fail_pending(self, error: Exception) -> None:
        current = self._current
        if current is not None:
            self._deliver_error(current, error)
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            if job is not None:
                self._deliver_error(job, error)

    @staticmethod
    def _deliver_result(job: _Job, result: str) -> None:
        def deliver() -> None:
            if not job.future.done():
                job.future.set_result(result)

        try:
            job.loop.call_soon_threadsafe(deliver)
        except RuntimeError:
            pass  # the request loop has already shut down

    @staticmethod
    def _deliver_error(job: _Job, error: Exception) -> None:
        def deliver() -> None:
            if not job.future.done():
                job.future.set_exception(error)

        try:
            job.loop.call_soon_threadsafe(deliver)
        except RuntimeError:
            pass
