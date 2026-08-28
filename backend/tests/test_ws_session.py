import asyncio
from collections.abc import Mapping, Sequence

from fastapi.testclient import TestClient

from app.main import create_app


class FakePipeline:
    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
    ) -> None:
        assert audio == b"pcm"
        await emit({"type": "tts_start", "generation_id": generation_id, "seq": 0})
        await emit({"type": "tts_end", "generation_id": generation_id})


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


class BlockingPipeline:
    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit,
    ) -> None:
        await asyncio.sleep(60)


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
