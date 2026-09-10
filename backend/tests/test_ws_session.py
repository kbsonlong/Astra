import asyncio
from collections.abc import Mapping, Sequence

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.main import create_app


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


def test_websocket_rejects_audio_over_session_limit() -> None:
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
            "code": "audio_too_large",
            "max_bytes": 3,
        }
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()
