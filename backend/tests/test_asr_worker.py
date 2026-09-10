import asyncio
import threading

import pytest

from app.models.asr_client import ASRClientError
from app.models.asr_worker import AsrWorkerClient


class ThreadRecordingASR:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def is_ready(self) -> bool:
        return True

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        self.thread_ids.append(threading.get_ident())
        await asyncio.sleep(0.01)
        return f"{filename}:{audio.decode()}"


@pytest.mark.anyio
async def test_worker_runs_multiple_requests_on_one_fixed_thread() -> None:
    client = ThreadRecordingASR()
    worker = AsrWorkerClient(client, max_queue=4, request_timeout_seconds=1)
    try:
        results = await asyncio.gather(
            worker.transcribe(b"one", "one.wav"),
            worker.transcribe(b"two", "two.wav"),
        )
    finally:
        await worker.aclose()

    assert results == ["one.wav:one", "two.wav:two"]
    assert len(client.thread_ids) == 2
    assert len(set(client.thread_ids)) == 1
    assert client.thread_ids[0] != threading.get_ident()


@pytest.mark.anyio
async def test_worker_rejects_requests_when_queue_is_full() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingASR:
        def is_ready(self) -> bool:
            return True

        async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
            started.set()
            await asyncio.to_thread(release.wait)
            return audio.decode()

    worker = AsrWorkerClient(BlockingASR(), max_queue=1, request_timeout_seconds=1)
    first = asyncio.create_task(worker.transcribe(b"one"))
    await asyncio.to_thread(started.wait, 1)
    second = asyncio.create_task(worker.transcribe(b"two"))
    await asyncio.sleep(0.02)
    try:
        with pytest.raises(ASRClientError, match="queue is full"):
            await worker.transcribe(b"three")
    finally:
        release.set()
        assert await first == "one"
        assert await second == "two"
        await worker.aclose()


@pytest.mark.anyio
async def test_worker_timeout_drops_late_result_without_stopping_worker() -> None:
    release = threading.Event()

    class SlowASR:
        def is_ready(self) -> bool:
            return True

        async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
            await asyncio.to_thread(release.wait)
            return "late"

    worker = AsrWorkerClient(SlowASR(), request_timeout_seconds=0.02)
    try:
        with pytest.raises(ASRClientError, match="request timed out"):
            await worker.transcribe(b"audio")
        release.set()
        await asyncio.sleep(0.02)
        assert worker.is_ready() is True
    finally:
        release.set()
        await worker.aclose()


@pytest.mark.anyio
async def test_worker_propagates_asr_errors() -> None:
    class BrokenASR:
        def is_ready(self) -> bool:
            return True

        async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
            raise ASRClientError("model unavailable")

    worker = AsrWorkerClient(BrokenASR())
    try:
        with pytest.raises(ASRClientError, match="model unavailable"):
            await worker.transcribe(b"audio")
    finally:
        await worker.aclose()
