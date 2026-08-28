import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_base_url="http://llm.test/v1",
        llm_model="test-model",
        llm_api_key="sk-test-secret",
        asr_model="test-model",
        tts_model_path="test.onnx",
    )


def test_config_masks_api_key(settings: Settings) -> None:
    client = TestClient(create_app(settings))

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["llm_api_key"] == "sk-t...cret"
    assert "secret" not in response.text


@pytest.mark.anyio
async def test_collects_dependency_health(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, request=request)

    transport = httpx.MockTransport(handler)

    import app.health

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(app.health.httpx, "AsyncClient", MockClient)
    class Ready:
        def is_ready(self) -> bool:
            return True

    class FakePipeline:
        asr = Ready()
        tts = Ready()

    result = await app.health.collect_health(settings, FakePipeline())

    assert result["ok"] is True
    assert result["llm"]["models_ok"] is True
    assert result["asr"]["ok"] is True
    assert result["tts"]["ok"] is True
