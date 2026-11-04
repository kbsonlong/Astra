"""Bounded, disk-backed PCM recording for realtime WebSocket sessions."""
from __future__ import annotations

import os
import time
import wave
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

PCM_SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
FLUSH_BYTES = PCM_SAMPLE_RATE * PCM_SAMPLE_WIDTH


class RecordingLimitExceeded(ValueError):
    pass


def _recording_name(recording_id: str) -> str:
    try:
        return str(UUID(recording_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid recording id") from exc


class RealtimeRecording:
    """Append PCM16 frames directly to disk and finalize them as a WAV file."""

    def __init__(self, directory: str | Path, recording_id: str, *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than 0")
        self.directory = Path(directory).expanduser()
        self.recording_id = _recording_name(recording_id)
        self.max_bytes = max_bytes
        self._handle: BinaryIO | None = None
        self._size = self.pcm_path.stat().st_size if self.pcm_path.is_file() else 0
        self._unflushed_bytes = 0

    @property
    def pcm_path(self) -> Path:
        return self.directory / f"{self.recording_id}.pcm"

    @property
    def wav_path(self) -> Path:
        return self.directory / f"{self.recording_id}.wav"

    def append(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        if len(pcm16) % PCM_SAMPLE_WIDTH:
            raise ValueError("PCM16 payload must have an even byte length")
        if self.wav_path.exists():
            raise RuntimeError("recording is already finalized")
        if len(pcm16) > self.max_bytes - self._size:
            raise RecordingLimitExceeded(f"recording exceeds {self.max_bytes} bytes")
        if self._handle is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._handle = self.pcm_path.open("ab", buffering=64 * 1024)
        self._handle.write(pcm16)
        self._size += len(pcm16)
        self._unflushed_bytes += len(pcm16)
        if self._unflushed_bytes >= FLUSH_BYTES:
            self._handle.flush()
            self._unflushed_bytes = 0

    def close(self) -> None:
        """Flush partial PCM without finalizing it, allowing a reconnect to append."""
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        self._unflushed_bytes = 0

    def finish(self) -> Path | None:
        """Atomically make a browser-downloadable WAV; idempotent after success."""
        self.close()
        if self.wav_path.is_file():
            return self.wav_path
        if not self.pcm_path.is_file() or self.pcm_path.stat().st_size == 0:
            return None
        temporary = self.wav_path.with_suffix(".wav.tmp")
        try:
            with wave.open(str(temporary), "wb") as output:
                output.setnchannels(PCM_CHANNELS)
                output.setsampwidth(PCM_SAMPLE_WIDTH)
                output.setframerate(PCM_SAMPLE_RATE)
                with self.pcm_path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        output.writeframesraw(chunk)
            os.replace(temporary, self.wav_path)
            self.pcm_path.unlink(missing_ok=True)
            self._size = 0
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return self.wav_path


def recording_download_path(directory: str | Path, recording_id: str) -> Path:
    recording = RealtimeRecording(directory, recording_id, max_bytes=1)
    return recording.wav_path


def clean_expired_realtime_recordings(directory: str | Path, retention_days: int) -> int:
    """Remove only stale files in the dedicated realtime-recording directory."""
    if retention_days <= 0:
        raise ValueError("retention_days must be greater than 0")
    root = Path(directory).expanduser()
    if not root.is_dir():
        return 0
    cutoff = time.time() - retention_days * 24 * 60 * 60
    removed = 0
    for path in root.iterdir():
        if path.suffix not in {".pcm", ".wav", ".tmp"} or not path.is_file():
            continue
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
