"""带大小上限的上传读取工具。"""
from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(ValueError):
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"upload exceeds {max_bytes} bytes")


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
