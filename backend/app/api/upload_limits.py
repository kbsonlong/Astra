"""带大小、时长和并发上限的音频上传工具。"""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

from fastapi import UploadFile

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(ValueError):
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"upload exceeds {max_bytes} bytes")


class AudioTooLongError(ValueError):
    def __init__(self, max_seconds: float, duration_seconds: float) -> None:
        self.max_seconds = max_seconds
        self.duration_seconds = duration_seconds
        super().__init__(f"audio duration exceeds {max_seconds} seconds")


class AudioIPConcurrencyLimiter:
    """Limit active audio work shared by HTTP uploads, SSE and WebSocket sessions."""

    def __init__(self, max_concurrent_per_ip: int) -> None:
        if max_concurrent_per_ip <= 0:
            raise ValueError("max_concurrent_per_ip must be greater than 0")
        self.max_concurrent_per_ip = max_concurrent_per_ip
        self._active: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def try_acquire(self, ip: str) -> bool:
        async with self._lock:
            active = self._active.get(ip, 0)
            if active >= self.max_concurrent_per_ip:
                return False
            self._active[ip] = active + 1
            return True

    async def release(self, ip: str) -> None:
        async with self._lock:
            active = self._active.get(ip, 0)
            if active <= 1:
                self._active.pop(ip, None)
            else:
                self._active[ip] = active - 1

async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """读取供即时 ASR 使用的上传，并在内存分配前检查总大小。"""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


async def save_upload_limited(
    file: UploadFile, destination: Path, max_bytes: int
) -> int:
    """分块写入长会议音频；超限时删除不完整文件。"""
    total = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise UploadTooLargeError(max_bytes)
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return total


def _soundfile_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:
        return None


def probe_audio_duration(path: Path) -> float | None:
    """Return decoded duration when a local decoder can identify the file."""
    duration = _soundfile_duration(path)
    if duration is not None:
        return duration

    # libsndfile does not handle every format accepted by afconvert (notably
    # some m4a/mp3 files). Keep this fallback local and bounded to the upload.
    with tempfile.TemporaryDirectory(prefix="astra_duration_") as directory:
        decoded = Path(directory) / "decoded.wav"
        try:
            subprocess.run(
                [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@16000",
                    "-c",
                    "1",
                    str(path),
                    str(decoded),
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return _soundfile_duration(decoded)


async def validate_audio_duration(
    audio: bytes, filename: str, max_seconds: float
) -> None:
    """Reject a decodable upload whose duration exceeds the configured limit."""
    with tempfile.NamedTemporaryFile(
        prefix="astra_upload_", suffix=Path(filename).suffix, delete=True
    ) as temporary:
        temporary.write(audio)
        temporary.flush()
        duration = await asyncio.to_thread(probe_audio_duration, Path(temporary.name))
    if duration is not None and duration > max_seconds:
        raise AudioTooLongError(max_seconds, duration)


async def validate_audio_file_duration(path: Path, max_seconds: float) -> None:
    duration = await asyncio.to_thread(probe_audio_duration, path)
    if duration is not None and duration > max_seconds:
        raise AudioTooLongError(max_seconds, duration)
