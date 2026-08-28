from fastapi import FastAPI

from .api.ws_session import router as ws_router
from .api.http_routes import router as http_router
from .config import Settings
from .health import collect_health
from .core.pipeline import VoicePipeline
from .models.asr_client import MlxAudioAsrClient
from .models.llm_client import OpenAICompatLLMClient
from .models.tts_client import PiperSdkTtsClient


def create_app(
    settings: Settings | None = None,
    pipeline: VoicePipeline | None = None,
    *,
    enable_pipeline: bool = True,
) -> FastAPI:
    app = FastAPI(title="Astra API", version="0.1.0")
    current = settings or Settings.from_env()
    app.state.settings = current
    app.state.pipeline = pipeline
    if enable_pipeline and pipeline is None:
        app.state.pipeline = VoicePipeline(
            MlxAudioAsrClient(current.asr_model, current.asr_language),
            OpenAICompatLLMClient(
                current.llm_base_url,
                current.llm_model,
                current.llm_api_key,
                current.llm_request_timeout_seconds,
                current.llm_connect_timeout_seconds,
                current.llm_stream_idle_timeout_seconds,
            ),
            PiperSdkTtsClient(current.tts_model_path),
        )
    app.include_router(ws_router)
    app.include_router(http_router)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return await collect_health(app.state.settings, app.state.pipeline)

    @app.get("/api/config")
    async def config() -> dict[str, object]:
        current = app.state.settings
        return {
            "llm_base_url": current.llm_base_url,
            "llm_model": current.llm_model,
            "llm_api_key": current.llm_api_key_masked,
            "asr_model": current.asr_model,
            "asr_language": current.asr_language,
            "tts_model_path": current.tts_model_path,
            "version": current.version,
        }

    return app


app = create_app()
