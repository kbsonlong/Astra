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
    settings = Settings(
        **{
            **settings.__dict__,
            "vad_model": "/private/models/vad.onnx",
            "speaker_store_path": "/private/data/speakers.sqlite3",
            "speaker_sample_dir": "/private/data/samples",
            "tts_model_path": "/private/models/voice.onnx",
            "task_store_path": "/private/data/tasks.sqlite3",
        }
    )
    client = TestClient(create_app(settings))

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["llm_api_key"] == "sk-t...cret"
    assert "secret" not in response.text
    for path in (
        "/private/models/vad.onnx",
        "/private/data/speakers.sqlite3",
        "/private/data/samples",
        "/private/models/voice.onnx",
        "/private/data/tasks.sqlite3",
    ):
        assert path not in response.text
    for key in (
        "vad_model",
        "speaker_store_path",
        "speaker_sample_dir",
        "tts_model_path",
        "task_store_path",
    ):
        assert key not in response.json()


def test_llm_config_can_be_saved_and_preserves_masked_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        llm_base_url="http://old.test/v1",
        llm_chat_path="/chat/completions",
        llm_models_path="/models",
        llm_model="old-model",
        llm_api_key="sk-existing-secret",
        config_path=str(tmp_path / "runtime.env"),
    )
    client = TestClient(
        create_app(settings, enable_pipeline=False, enable_meeting=False)
    )
    for key in (
        "LLM_BASE_URL",
        "LLM_CHAT_PATH",
        "LLM_MODELS_PATH",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_REQUEST_TIMEOUT_SECONDS",
        "LLM_CONNECT_TIMEOUT_SECONDS",
        "LLM_STREAM_IDLE_TIMEOUT_SECONDS",
        "LLM_CORRECTION_ENABLED",
        "LLM_CORRECTION_MAX_TOKENS",
        "LLM_CORRECTION_SYSTEM_PROMPT",
        "MEETING_LLM_CORRECTION_ENABLED",
        "MEETING_LLM_CORRECTION_CANDIDATES",
    ):
        monkeypatch.delenv(key, raising=False)
    payload = {
        "llm_base_url": "http://new.test/v1/",
        "llm_chat_path": "/v1/chat/completions",
        "llm_models_path": "/v1/models",
        "llm_model": "new-model",
        "llm_api_key": None,
        "llm_request_timeout_seconds": 90,
        "llm_connect_timeout_seconds": 4,
        "llm_stream_idle_timeout_seconds": 20,
        "llm_correction_enabled": True,
        "llm_correction_max_tokens": 128,
        "llm_correction_system_prompt": "只修正明显错误。",
        "meeting_llm_correction_enabled": False,
        "meeting_llm_correction_candidates": ["术语A->术语B"],
    }

    response = client.put("/api/config", json=payload)

    assert response.status_code == 200
    assert response.json()["llm_base_url"] == "http://new.test/v1"
    assert response.json()["llm_api_key"] == "sk-e...cret"
    assert response.json()["runtime_applied"] is True
    env_text = (tmp_path / "runtime.env").read_text(encoding="utf-8")
    assert "LLM_BASE_URL='http://new.test/v1'" in env_text
    assert "LLM_MODEL='new-model'" in env_text
    assert "LLM_API_KEY='sk-existing-secret'" in env_text
    assert not (tmp_path / ".env").exists()

    saved = client.get("/api/config").json()
    assert saved["llm_model"] == "new-model"
    assert saved["meeting_llm_correction_candidates"] == ["术语A->术语B"]


def test_settings_reads_dotenv_values(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    # load_dotenv 默认不覆盖已存在的环境变量: 若同进程内先前测试在项目根
    # cwd 下跑过 Settings.from_env(), 真实 .env 的值会残留进 os.environ,
    # 导致本测试读到串扰值(顺序相关 flake)。先清掉相关键再加载。
    for key in (
        "ASTRA_CONFIG_PATH",
        "ASR_MODEL", "ASR_LANGUAGE", "ASR_MAX_TOKENS", "ASR_REPETITION_PENALTY",
        "ASR_REPETITION_CONTEXT_SIZE", "ASR_HOTWORDS", "ASR_SYSTEM_PROMPT",
        "TTS_MODEL_PATH", "LLM_MODELS_PATH", "TRANSCRIBE_MAX_UPLOAD_BYTES",
        "TRANSCRIBE_MAX_DURATION_SECONDS", "MEETING_MAX_UPLOAD_BYTES",
        "MEETING_MAX_DURATION_SECONDS", "WS_MAX_AUDIO_BYTES",
        "AUDIO_MAX_CONCURRENT_PER_IP", "AUDIO_ENHANCEMENT_ENABLED",
        "AUDIO_ANS_MODEL", "AUDIO_AEC_MODEL", "AUDIO_ENHANCEMENT_MODEL_DIR",
        "MEETING_MAX_CONCURRENT_JOBS", "TRAINING_MAX_CONCURRENT_JOBS",
        "MEETING_TASK_TIMEOUT_SECONDS", "TRAINING_TASK_TIMEOUT_SECONDS",
        "MEETING_ARTIFACT_RETENTION_DAYS",
        "MEETING_ARTIFACT_MAX_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "ASR_MODEL=/models/whisper\nASR_LANGUAGE=en\nASR_MAX_TOKENS=256\n"
        "ASR_REPETITION_PENALTY=1.12\nASR_HOTWORDS=host,大佬\n"
        "ASR_SYSTEM_PROMPT=只输出实际说出的内容。\n"
        "TTS_MODEL_PATH=/models/piper.onnx\n"
        "TRANSCRIBE_MAX_UPLOAD_BYTES=123\n"
        "TRANSCRIBE_MAX_DURATION_SECONDS=12.5\n"
        "MEETING_MAX_UPLOAD_BYTES=456\n"
        "MEETING_MAX_DURATION_SECONDS=34.5\n"
        "WS_MAX_AUDIO_BYTES=789\n"
        "AUDIO_MAX_CONCURRENT_PER_IP=5\n"
        "AUDIO_ENHANCEMENT_ENABLED=true\n"
        "AUDIO_ANS_MODEL=zipenhancer_16k\n"
        "AUDIO_AEC_MODEL=jaec_16k\n"
        "AUDIO_ENHANCEMENT_MODEL_DIR=/models/audio\n"
        "MEETING_MAX_CONCURRENT_JOBS=2\n"
        "TRAINING_MAX_CONCURRENT_JOBS=3\n"
        "MEETING_TASK_TIMEOUT_SECONDS=12.5\n"
        "TRAINING_TASK_TIMEOUT_SECONDS=25.5\n"
        "MEETING_ARTIFACT_RETENTION_DAYS=14\n"
        "MEETING_ARTIFACT_MAX_BYTES=12345\n",
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
    assert loaded.transcribe_max_upload_bytes == 123
    assert loaded.transcribe_max_duration_seconds == 12.5
    assert loaded.meeting_max_upload_bytes == 456
    assert loaded.meeting_max_duration_seconds == 34.5
    assert loaded.ws_max_audio_bytes == 789
    assert loaded.audio_max_concurrent_per_ip == 5
    assert loaded.audio_enhancement_enabled is True
    assert loaded.audio_ans_model == "zipenhancer_16k"
    assert loaded.audio_aec_model == "jaec_16k"
    assert loaded.audio_enhancement_model_dir == "/models/audio"
    assert loaded.meeting_max_concurrent_jobs == 2
    assert loaded.training_max_concurrent_jobs == 3
    assert loaded.meeting_task_timeout_seconds == 12.5
    assert loaded.training_task_timeout_seconds == 25.5
    assert loaded.meeting_artifact_retention_days == 14
    assert loaded.meeting_artifact_max_bytes == 12345


def test_settings_uses_explicit_astra_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config_path = tmp_path / "astra-runtime.env"
    config_path.write_text("LLM_MODEL=explicit-model\n", encoding="utf-8")
    monkeypatch.setenv("ASTRA_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("LLM_MODEL", raising=False)

    loaded = Settings.from_env()

    assert loaded.config_path == str(config_path)
    assert loaded.llm_model == "explicit-model"


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
