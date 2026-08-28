from fastapi.testclient import TestClient

from app.main import create_app


def test_websocket_session_transitions_and_increments_generation() -> None:
    client = TestClient(create_app())

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
    client = TestClient(create_app())

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        websocket.receive_json()
        websocket.send_json({"type": "speech_end"})
        websocket.receive_json()
        websocket.send_json({"type": "interrupt", "generation_id": 99, "reason": "manual"})

        assert websocket.receive_json()["state"] == "REASONING"
