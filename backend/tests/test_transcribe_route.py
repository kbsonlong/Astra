from collections.abc import Mapping, Sequence

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.asr_client import ASRClientError


class FakeASR:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        assert audio == b"wav-data"
        assert filename == "test.wav"
        return "测试语音"

    def is_ready(self) -> bool:
        return True


class FakeLLM:
    model = "test-model"

    async def stream_chat(self, messages, **kwargs):
        assert messages[1]["content"] == "测试语音"
        assert kwargs["max_tokens"] == 256
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
        yield "测试"
        yield "语音。"


class FakePipeline:
    asr = FakeASR()
    llm = FakeLLM()


class FailingASR:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        raise ASRClientError("mlx-audio model is not available locally")

    def is_ready(self) -> bool:
        return False


class FailingPipeline:
    asr = FailingASR()


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


def test_transcribe_stream_route_returns_asr_and_correction_events() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/api/transcribe/stream",
        files={"file": ("test.wav", b"wav-data", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "asr_final"' in response.text
    assert '"type": "correction_token"' in response.text
    assert '"text": "测试语音。"' in response.text
    assert "data: [DONE]" in response.text


def test_transcribe_route_returns_service_unavailable_for_sdk_error() -> None:
    client = TestClient(create_app(pipeline=FailingPipeline()))

    response = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", b"wav-data", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "mlx-audio model is not available locally"
