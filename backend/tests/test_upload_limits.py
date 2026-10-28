import asyncio
import io
import wave

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.api.upload_limits import (
    AudioIPConcurrencyLimiter,
    AudioProbeStorageError,
    AudioProbeTimeoutError,
    AudioTooLongError,
    probe_audio_duration,
    validate_audio_duration,
)
from app.config import Settings
from app.main import create_app


def wav_bytes(seconds: float) -> bytes:
    frames = int(8000 * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\0\0" * frames)
    return buffer.getvalue()


def test_validate_audio_duration_rejects_decoded_audio_over_limit() -> None:
    with pytest.raises(AudioTooLongError, match="duration exceeds"):
        asyncio.run(validate_audio_duration(wav_bytes(2), "recording.wav", 1))


def test_audio_probe_terminates_a_hung_decoder(monkeypatch, tmp_path) -> None:
    class HungDecoder:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    decoder = HungDecoder()
    monkeypatch.setattr("app.api.upload_limits.subprocess.Popen", lambda *args, **kwargs: decoder)

    with pytest.raises(AudioProbeTimeoutError, match="exceeded"):
        probe_audio_duration(tmp_path / "input.m4a", timeout_seconds=0.01)


def test_audio_probe_rejects_decoder_temp_storage_over_budget(monkeypatch, tmp_path) -> None:
    class GrowingDecoder:
        returncode = None

        def __init__(self, command):
            Path(command[-1]).write_bytes(b"x" * 10)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    from pathlib import Path

    monkeypatch.setattr(
        "app.api.upload_limits.subprocess.Popen",
        lambda command, **kwargs: GrowingDecoder(command),
    )

    with pytest.raises(AudioProbeStorageError, match="temporary storage"):
        probe_audio_duration(tmp_path / "input.m4a", max_temp_bytes=1)


class NoopASR:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        raise AssertionError("ASR must not run for an oversized duration")


class NoopPipeline:
    asr = NoopASR()


def test_transcribe_route_rejects_duration_with_http_413() -> None:
    client = TestClient(
        create_app(
            settings=Settings(transcribe_max_duration_seconds=1),
            pipeline=NoopPipeline(),
        )
    )

    response = client.post(
        "/api/transcribe",
        files={"file": ("recording.wav", wav_bytes(2), "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "audio duration too long (max 1 seconds)"


def test_meeting_route_rejects_duration_with_http_413(tmp_path) -> None:
    client = TestClient(
        create_app(
            settings=Settings(meeting_output_dir=str(tmp_path), meeting_max_duration_seconds=1),
            enable_pipeline=False,
            enable_meeting=False,
        )
    )

    response = client.post(
        "/api/meeting/process",
        files={"file": ("recording.wav", wav_bytes(2), "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "audio duration too long (max 1 seconds)"
    assert list(tmp_path.iterdir()) == []


def test_ip_limiter_releases_slots_and_isolated_ips_can_continue() -> None:
    limiter = AudioIPConcurrencyLimiter(1)

    async def exercise() -> None:
        assert await limiter.try_acquire("10.0.0.1")
        assert not await limiter.try_acquire("10.0.0.1")
        assert await limiter.try_acquire("10.0.0.2")
        await limiter.release("10.0.0.1")
        assert await limiter.try_acquire("10.0.0.1")

    asyncio.run(exercise())


def test_websocket_reports_ip_concurrency_limit() -> None:
    app = create_app(settings=Settings(audio_max_concurrent_per_ip=1), enable_pipeline=False)
    assert asyncio.run(app.state.audio_ip_limiter.try_acquire("testclient"))
    try:
        with TestClient(app).websocket_connect("/ws") as websocket:
            assert websocket.receive_json() == {
                "type": "error",
                "code": "concurrency_limit",
            }
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_json()
    finally:
        asyncio.run(app.state.audio_ip_limiter.release("testclient"))
