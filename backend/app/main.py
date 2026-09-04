from fastapi import FastAPI

from .api.ws_session import router as ws_router
from .api.http_routes import router as http_router
from .config import Settings
from .health import collect_health
from .core.pipeline import VoicePipeline
from .models.asr_client import (
    MlxAudioAsrClient,
    SherpaSenseVoiceAsrClient,
    SherpaZipformerBilingualAsrClient,
)
from .models.llm_client import OpenAICompatLLMClient
from .models.tts_client import PiperSdkTtsClient


def _build_asr_client(current: Settings) -> object:
    engine = (current.asr_engine or "mlx").lower()
    if engine in {"sherpa", "sherpa-sensevoice", "sensevoice", "sense-voice"}:
        return SherpaSenseVoiceAsrClient(
            model_dir=current.sherpa_model_dir,
            language=current.asr_language,
            num_threads=current.sherpa_num_threads,
            provider=current.sherpa_provider,
            auto_language=current.sherpa_auto_language,
            use_itn=current.sherpa_use_itn,
            chunk_duration=current.asr_chunk_duration_seconds,
            long_audio_threshold=current.asr_long_audio_threshold_seconds,
            hotwords=current.asr_hotwords,
        )
    if engine in {"zipformer", "sherpa-zipformer", "sherpa-zipformer-bilingual", "zipformer-bilingual", "zipformer-bilingual-zh-en"}:
        return SherpaZipformerBilingualAsrClient(
            model_dir=current.zipformer_model_dir,
            language=current.asr_language,
            num_threads=current.zipformer_num_threads,
            provider=current.zipformer_provider,
            decoding_method=current.zipformer_decoding_method,
            chunk_duration=current.asr_chunk_duration_seconds,
            long_audio_threshold=current.asr_long_audio_threshold_seconds,
            hotwords=current.asr_hotwords,
        )
    return MlxAudioAsrClient(
        current.asr_model,
        current.asr_language,
        max_tokens=current.asr_max_tokens,
        repetition_penalty=current.asr_repetition_penalty,
        repetition_context_size=current.asr_repetition_context_size,
        chunk_duration=current.asr_chunk_duration_seconds,
        long_audio_threshold=current.asr_long_audio_threshold_seconds,
        hotwords=current.asr_hotwords,
        system_prompt=current.asr_system_prompt,
    )


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
            _build_asr_client(current),
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
            "llm_correction_enabled": current.llm_correction_enabled,
            "llm_correction_max_tokens": current.llm_correction_max_tokens,
            "asr_engine": current.asr_engine,
            "asr_model": current.asr_model,
            "asr_language": current.asr_language,
            "asr_max_tokens": current.asr_max_tokens,
            "asr_repetition_penalty": current.asr_repetition_penalty,
            "asr_repetition_context_size": current.asr_repetition_context_size,
            "asr_chunk_duration_seconds": current.asr_chunk_duration_seconds,
            "asr_long_audio_threshold_seconds": current.asr_long_audio_threshold_seconds,
            "asr_hotwords": list(current.asr_hotwords),
            "asr_system_prompt_configured": bool(current.asr_system_prompt),
            "sherpa_model_dir": current.sherpa_model_dir,
            "sherpa_num_threads": current.sherpa_num_threads,
            "sherpa_provider": current.sherpa_provider,
            "sherpa_auto_language": current.sherpa_auto_language,
            "sherpa_use_itn": current.sherpa_use_itn,
            "zipformer_model_dir": current.zipformer_model_dir,
            "zipformer_num_threads": current.zipformer_num_threads,
            "zipformer_provider": current.zipformer_provider,
            "zipformer_decoding_method": current.zipformer_decoding_method,
            "tts_model_path": current.tts_model_path,
            "version": current.version,
        }

    return app


app = create_app()
