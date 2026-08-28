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


def test_settings_reads_dotenv_values(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "ASR_MODEL=/models/whisper\nASR_LANGUAGE=en\nASR_MAX_TOKENS=256\n"
        "ASR_REPETITION_PENALTY=1.12\nASR_HOTWORDS=host,大佬\n"
        "ASR_SYSTEM_PROMPT=只输出实际说出的内容。\n"
        "TTS_MODEL_PATH=/models/piper.onnx\n",
        encoding="utf-8",
    )

    from app.config import Settings

    loaded = Settings.from_env()

    assert loaded.asr_model == "/models/whisper"
    assert loaded.asr_language == "en"
    assert loaded.asr_max_tokens == 256
    assert loaded.asr_repetition_penalty == 1.12
    assert loaded.asr_hotwords == ("host", "大佬")
    assert loaded.asr_system_prompt == "只输出实际说出的内容。"
    assert loaded.tts_model_path == "/models/piper.onnx"


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
