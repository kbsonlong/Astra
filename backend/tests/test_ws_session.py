import asyncio
import io
import struct
from uuid import uuid4
import wave
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.core.pcm_protocol import PCM_FRAME_HEADER, PCM_FRAME_MAGIC, PCM_FRAME_VERSION
from app.core.audio_adapter import decode_audio_bytes
from app.core.tts_voice_store import TtsVoiceSnapshot


class FakePipeline:
    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
    ) -> tuple[str, str]:
        assert audio == b"pcm"
        await emit({"type": "tts_start", "generation_id": generation_id, "seq": 0})
        await emit({"type": "tts_end", "generation_id": generation_id})
        return "用户说的", "好的"


class HangingPipeline:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
    ) -> tuple[str, str]:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return "", ""


class ReferencePipeline:
    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
        reference: bytes | None = None,
    ) -> tuple[str, str]:
        assert audio == b"mic"
        assert reference == b"ref"
        return "用户说的", "好的"


class SnapshotPipeline:
    def __init__(self, expected: TtsVoiceSnapshot) -> None:
        self.expected = expected
        self.received: TtsVoiceSnapshot | None = None

    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
        reference: bytes | None = None,
        voice_snapshot: TtsVoiceSnapshot | None = None,
    ) -> tuple[str, str]:
        assert audio == b"pcm"
        assert reference is None
        assert voice_snapshot == self.expected
        self.received = voice_snapshot
        return "用户说的", "好的"


class SnapshotStore:
    def __init__(self, snapshot: TtsVoiceSnapshot) -> None:
        self.snapshot_value = snapshot
        self.calls: list[tuple[str, int | None]] = []

    def snapshot(self, voice_id: str, revision: int | None = None) -> TtsVoiceSnapshot:
        self.calls.append((voice_id, revision))
        return self.snapshot_value


def app_without_pipeline():
    return create_app(enable_pipeline=False)


def test_websocket_session_transitions_and_increments_generation() -> None:
    client = TestClient(app_without_pipeline())

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        assert websocket.receive_json() == {"type": "state_change", "state": "LISTENING"}

        websocket.send_bytes(b"pcm")
        websocket.send_json({"type": "speech_end"})
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "REASONING",
            "generation_id": 1,
        }

        websocket.send_json({"type": "interrupt", "generation_id": 1, "reason": "vad"})
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "LISTENING",
            "generation_id": 1,
        }


def test_old_generation_interrupt_does_not_change_current_state() -> None:
    client = TestClient(app_without_pipeline())

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_json({"type": "speech_end"})
        websocket.receive_json()
        websocket.send_json({"type": "interrupt", "generation_id": 99, "reason": "manual"})

        assert websocket.receive_json()["state"] == "REASONING"


def test_websocket_runs_injected_pipeline_and_returns_to_listening() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_bytes(b"pcm")
        websocket.send_json({"type": "speech_end"})
        assert websocket.receive_json()["state"] == "REASONING"
        assert websocket.receive_json()["type"] == "state_change"
        assert websocket.receive_json()["type"] == "tts_start"
        assert websocket.receive_json()["type"] == "tts_end"
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "LISTENING",
            "generation_id": 1,
        }


def test_websocket_disconnect_cancels_generation_and_releases_ip_slot() -> None:
    pipeline = HangingPipeline()
    app = create_app(pipeline=pipeline)

    with TestClient(app).websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_bytes(b"pcm")
        websocket.send_json({"type": "speech_end"})
        assert websocket.receive_json()["state"] == "REASONING"
        assert pipeline.started.wait(timeout=2)

    assert pipeline.cancelled.wait(timeout=2)
    deadline = time.monotonic() + 2
    while app.state.audio_ip_limiter._active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert app.state.audio_ip_limiter._active == {}


def test_websocket_forwards_far_end_reference_channel() -> None:
    client = TestClient(create_app(pipeline=ReferencePipeline()))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_bytes(b"mic")
        websocket.send_json({"type": "audio_channel", "channel": "reference"})
        websocket.receive_json()
        websocket.send_bytes(b"ref")
        websocket.send_json({"type": "audio_channel", "channel": "microphone"})
        websocket.receive_json()
        websocket.send_json({"type": "speech_end"})

        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "REASONING",
            "generation_id": 1,
        }
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "LISTENING",
            "generation_id": 1,
        }


def test_websocket_resolves_and_forwards_voice_snapshot_for_session(tmp_path: Path) -> None:
    snapshot = TtsVoiceSnapshot(
        voice_id="voice-1",
        voice_revision=3,
        backend_family="indextts25_mlx",
        language="zh",
        reference_path=tmp_path / "reference.wav",
        reference_sha256="a" * 64,
        default_params={"temperature": 0.7},
    )
    pipeline = SnapshotPipeline(snapshot)
    app = create_app(pipeline=pipeline)
    store = SnapshotStore(snapshot)
    app.state.tts_voice_store = store

    with TestClient(app).websocket_connect("/ws") as websocket:
        websocket.send_json(
            {"type": "start_session", "voice_id": "voice-1", "voice_revision": 3}
        )
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "LISTENING",
            "voice_id": "voice-1",
            "voice_revision": 3,
        }
        websocket.send_bytes(b"pcm")
        websocket.send_json({"type": "speech_end"})
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "REASONING",
            "generation_id": 1,
            "voice_id": "voice-1",
            "voice_revision": 3,
        }
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "LISTENING",
            "generation_id": 1,
            "voice_id": "voice-1",
            "voice_revision": 3,
        }

    assert store.calls == [("voice-1", 3)]
    assert pipeline.received == snapshot


class PcmPipeline:
    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
        reference: bytes | None = None,
    ) -> tuple[str, str]:
        assert len(decode_audio_bytes(audio).samples) == 160
        assert reference is not None
        assert len(decode_audio_bytes(reference, source="reference").samples) == 160
        return "用户说的", "好的"


class MicrophoneOnlyPcmPipeline:
    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
        reference: bytes | None = None,
    ) -> tuple[str, str]:
        assert len(decode_audio_bytes(audio).samples) == 160
        assert reference is None
        return "长语音首段", ""


class PendingPcmPipeline:
    def __init__(self) -> None:
        self.received_audio: bytes | None = None

    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
        reference: bytes | None = None,
    ) -> tuple[str, str]:
        self.received_audio = audio
        assert len(decode_audio_bytes(audio).samples) == 160
        assert reference is None
        await emit(
            {
                "type": "asr_final",
                "text": "停止时仍在缓冲的语音",
                "generation_id": generation_id,
            }
        )
        return "停止时仍在缓冲的语音", ""


def _pcm_frame(sequence: int, channel: int, value: int) -> bytes:
    samples = np.full(160, value, dtype="<i2").tobytes()
    return PCM_FRAME_HEADER.pack(
        PCM_FRAME_MAGIC, PCM_FRAME_VERSION, channel, sequence, 16_000, 160
    ) + samples


def test_websocket_accepts_10ms_pcm_frames() -> None:
    client = TestClient(create_app(pipeline=PcmPipeline()))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "audio_format",
                "format": "pcm16",
                "sample_rate": 16_000,
                "frame_samples": 160,
            }
        )
        assert websocket.receive_json() == {
            "type": "audio_format_ready",
            "format": "pcm16",
            "sample_rate": 16_000,
            "frame_samples": 160,
        }
        assert websocket.receive_json()["state"] == "LISTENING"
        websocket.send_bytes(_pcm_frame(0, 0, 1000))
        websocket.send_bytes(_pcm_frame(0, 1, 500))
        websocket.send_json({"type": "speech_end"})

        assert websocket.receive_json()["state"] == "REASONING"
        assert websocket.receive_json()["state"] == "LISTENING"


def test_end_session_transcribes_pending_pcm_before_recording_ready(tmp_path) -> None:
    recording_id = str(uuid4())
    pipeline = PendingPcmPipeline()
    client = TestClient(
        create_app(
            settings=Settings(
                realtime_recording_dir=str(tmp_path),
                task_store_path=str(tmp_path / "tasks.sqlite3"),
            ),
            pipeline=pipeline,
        )
    )

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session", "recording_id": recording_id})
        assert websocket.receive_json()["state"] == "LISTENING"
        assert websocket.receive_json()["type"] == "recording_started"
        websocket.send_json(
            {"type": "audio_format", "format": "pcm16", "sample_rate": 16_000, "frame_samples": 160}
        )
        assert websocket.receive_json()["type"] == "audio_format_ready"
        assert websocket.receive_json()["state"] == "LISTENING"
        websocket.send_bytes(_pcm_frame(0, 0, 1000))
        websocket.send_json({"type": "end_session"})

        events = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] == "recording_ready":
                break

    assert pipeline.received_audio is not None
    assert any(
        event == {
            "type": "asr_final",
            "text": "停止时仍在缓冲的语音",
            "generation_id": 1,
        }
        for event in events
    )


def test_websocket_streams_realtime_recording_to_authenticated_download(tmp_path) -> None:
    recording_id = str(uuid4())
    client = TestClient(
        create_app(
            settings=Settings(
                realtime_recording_dir=str(tmp_path),
                task_store_path=str(tmp_path / "tasks.sqlite3"),
                admin_token="test-admin-token",
            ),
            enable_pipeline=False,
        )
    )
    download_url = f"/api/realtime-recordings/{recording_id}"
    assert client.get(download_url).status_code == 401
    assert client.post("/api/auth/login", json={"token": "test-admin-token"}).status_code == 200

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session", "recording_id": recording_id})
        assert websocket.receive_json()["state"] == "LISTENING"
        assert websocket.receive_json() == {
            "type": "recording_started", "recording_id": recording_id
        }
        websocket.send_json(
            {"type": "audio_format", "format": "pcm16", "sample_rate": 16_000, "frame_samples": 160}
        )
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_bytes(_pcm_frame(0, 0, 1000))
        websocket.send_json({"type": "end_session"})
        assert websocket.receive_json()["state"] == "IDLE"
        ready = websocket.receive_json()
        assert ready == {
            "type": "recording_ready",
            "recording_id": recording_id,
            "download_url": download_url,
        }

    response = client.get(ready["download_url"])
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content), "rb") as output:
        assert output.getframerate() == 16_000
        assert output.getnframes() == 160


def test_websocket_reconnect_resumes_same_recording_and_heartbeat(tmp_path) -> None:
    recording_id = str(uuid4())
    client = TestClient(
        create_app(
            settings=Settings(
                realtime_recording_dir=str(tmp_path),
                task_store_path=str(tmp_path / "tasks.sqlite3"),
            ),
            enable_pipeline=False,
        )
    )

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session", "recording_id": recording_id})
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json(
            {"type": "audio_format", "format": "pcm16", "sample_rate": 16_000, "frame_samples": 160}
        )
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_bytes(_pcm_frame(0, 0, 1000))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session", "recording_id": recording_id})
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json(
            {"type": "audio_format", "format": "pcm16", "sample_rate": 16_000, "frame_samples": 160}
        )
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_bytes(_pcm_frame(1, 0, 2000))
        websocket.send_json({"type": "heartbeat"})
        assert websocket.receive_json() == {"type": "heartbeat_ack"}
        websocket.send_json({"type": "end_session"})
        websocket.receive_json()
        ready = websocket.receive_json()

    response = client.get(ready["download_url"])
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content), "rb") as output:
        assert output.getnframes() == 320


class BlockingPipeline:
    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
    ) -> tuple[str, str]:
        await asyncio.sleep(60)
        return "", ""


def test_interrupt_cancels_running_pipeline_without_stale_events() -> None:
    client = TestClient(create_app(pipeline=BlockingPipeline()))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_bytes(b"pcm")
        websocket.send_json({"type": "speech_end"})
        assert websocket.receive_json()["state"] == "REASONING"

        websocket.send_json({"type": "interrupt", "generation_id": 1, "reason": "manual"})
        assert websocket.receive_json() == {
            "type": "state_change",
            "state": "IDLE",
            "generation_id": 1,
        }


def test_websocket_keeps_recording_session_alive_when_segment_limit_is_reached() -> None:
    client = TestClient(
        create_app(
            settings=Settings(ws_max_audio_bytes=3), enable_pipeline=False
        )
    )

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_bytes(b"four")
        assert websocket.receive_json() == {
            "type": "error",
            "code": "audio_segment_limit_reached",
            "max_bytes": 3,
        }
        websocket.send_json({"type": "heartbeat"})
        assert websocket.receive_json() == {"type": "heartbeat_ack"}


def test_pcm_recording_keeps_limit_crossing_frame_and_can_finalize(tmp_path) -> None:
    recording_id = str(uuid4())
    client = TestClient(
        create_app(
            settings=Settings(
                ws_max_audio_bytes=320,
                realtime_recording_dir=str(tmp_path),
                task_store_path=str(tmp_path / "tasks.sqlite3"),
            ),
            pipeline=MicrophoneOnlyPcmPipeline(),
        )
    )

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session", "recording_id": recording_id})
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json(
            {"type": "audio_format", "format": "pcm16", "sample_rate": 16_000, "frame_samples": 160}
        )
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_bytes(_pcm_frame(0, 0, 1000))
        websocket.send_bytes(_pcm_frame(1, 0, 2000))
        assert websocket.receive_json()["code"] == "audio_segment_limit_reached"
        websocket.send_json({"type": "speech_end"})
        assert websocket.receive_json()["state"] == "REASONING"
        assert websocket.receive_json()["state"] == "LISTENING"
        websocket.send_json({"type": "end_session"})
        assert websocket.receive_json()["state"] == "IDLE"
        ready = websocket.receive_json()

    response = client.get(ready["download_url"])
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content), "rb") as output:
        assert output.getnframes() == 320
