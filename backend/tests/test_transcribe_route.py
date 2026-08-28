from collections.abc import Mapping, Sequence

from fastapi.testclient import TestClient

from app.main import create_app


class FakeASR:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        assert audio == b"wav-data"
        assert filename == "test.wav"
        return "测试语音"

    def is_ready(self) -> bool:
        return True


class FakePipeline:
    asr = FakeASR()


def test_transcribe_route_returns_sdk_result() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", b"wav-data", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"filename": "test.wav", "bytes": 8, "text": "测试语音"}


def test_transcribe_route_rejects_empty_file() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/api/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "audio file is empty"
