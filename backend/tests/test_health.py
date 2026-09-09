import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.core.training import TrainingManager


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
    # load_dotenv 默认不覆盖已存在的环境变量: 若同进程内先前测试在项目根
    # cwd 下跑过 Settings.from_env(), 真实 .env 的值会残留进 os.environ,
    # 导致本测试读到串扰值(顺序相关 flake)。先清掉相关键再加载。
    for key in (
        "ASR_MODEL", "ASR_LANGUAGE", "ASR_MAX_TOKENS", "ASR_REPETITION_PENALTY",
        "ASR_REPETITION_CONTEXT_SIZE", "ASR_HOTWORDS", "ASR_SYSTEM_PROMPT",
        "TTS_MODEL_PATH", "LLM_MODELS_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
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


def test_settings_resolves_relative_vad_model_from_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VAD_MODEL", "models/custom-vad.onnx")

    from app.config import PROJECT_ROOT, Settings

    assert Settings.from_env().vad_model == str(PROJECT_ROOT / "models/custom-vad.onnx")


def test_settings_normalizes_llm_base_url_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000//v1///")

    from app.config import Settings

    assert Settings.from_env().llm_base_url == "http://127.0.0.1:8000/v1"


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


def test_training_config_can_be_read_and_saved(tmp_path) -> None:
    settings = Settings(
        qwen3_training_config_path=str(tmp_path / "qwen3-training.json")
    )
    client = TestClient(
        create_app(settings, enable_pipeline=False, enable_meeting=False)
    )

    initial = client.get("/api/training/config")
    assert initial.status_code == 200
    payload = initial.json()["config"]
    assert payload["model_path"] == "Qwen/Qwen3-ASR-0.6B"
    payload["device"] = "mps"
    payload["batch_size"] = 2
    payload["learning_rate"] = 1e-5

    saved = client.put("/api/training/config", json=payload)
    assert saved.status_code == 200
    assert saved.json()["config"]["device"] == "mps"
    assert saved.json()["runtime_applied"] is False

    loaded = client.get("/api/training/config")
    assert loaded.json()["config"]["batch_size"] == 2
    assert loaded.json()["config"]["learning_rate"] == 1e-5


def test_training_config_rejects_unsupported_device(tmp_path) -> None:
    settings = Settings(
        qwen3_training_config_path=str(tmp_path / "qwen3-training.json")
    )
    client = TestClient(
        create_app(settings, enable_pipeline=False, enable_meeting=False)
    )
    response = client.put(
        "/api/training/config",
        json={"device": "metal", "batch_size": 1},
    )
    assert response.status_code == 422


def test_training_status_is_idle_without_a_job(tmp_path) -> None:
    settings = Settings(qwen3_training_config_path=str(tmp_path / "qwen3-training.json"))
    client = TestClient(create_app(settings, enable_pipeline=False, enable_meeting=False))
    response = client.get("/api/training/status")
    assert response.status_code == 200
    assert response.json() == {"status": "idle"}


def test_training_start_reports_spawn_error(tmp_path) -> None:
    settings = Settings(qwen3_training_config_path=str(tmp_path / "qwen3-training.json"))
    app = create_app(settings, enable_pipeline=False, enable_meeting=False)
    async def fail_start(config):
        raise OSError("mlx-tune runtime is unavailable")
    app.state.training_manager.start = fail_start
    client = TestClient(app)
    response = client.post("/api/training/start")
    assert response.status_code == 422
