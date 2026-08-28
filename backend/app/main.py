from fastapi import FastAPI

from .config import Settings
from .health import collect_health


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Astra API", version="0.1.0")
    app.state.settings = settings or Settings.from_env()

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return await collect_health(app.state.settings)

    @app.get("/api/config")
    async def config() -> dict[str, object]:
        current = app.state.settings
        return {
            "llm_base_url": current.llm_base_url,
            "llm_model": current.llm_model,
            "llm_api_key": current.llm_api_key_masked,
            "asr_endpoint": current.asr_endpoint,
            "tts_endpoint": current.tts_endpoint,
            "version": current.version,
        }

    return app


app = create_app()
