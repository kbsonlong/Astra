from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from .api.auth import (
    auth_required,
    is_authorized_request,
    router as auth_router,
)
from .api.meeting_routes import router as meeting_router
from .api.speaker_routes import router as speaker_router
from .api.notification_routes import router as notification_router
from .api.ws_session import router as ws_router
from .api.http_routes import router as http_router
from .config import (
    Qwen3TrainingConfig,
    Settings,
    _normalize_base_url,
    load_qwen3_training_config,
    llm_environment_values,
    persist_llm_environment,
    save_qwen3_training_config,
)
from .core.meeting import MeetingPipeline
from .health import collect_health
from .core.pipeline import VoicePipeline
from .core.correction import parse_correction_rules
from .core.workflow import CorrectionStage, ResemblyzerDiarizationStage
from .models.asr_client import MlxAudioAsrClient
from .models.asr_worker import AsrWorkerClient
from .models.llm_client import OpenAICompatLLMClient
from .models.tts_client import PiperSdkTtsClient
from .models.punctuation_client import build_punctuation_client
from .core.speaker_registry import SpeakerProfileStore
from .main_types import LLMSettingsPayload, TrainingConfigPayload
from .api.training_routes import router as training_router
from .api.upload_limits import AudioIPConcurrencyLimiter
from .core.training import TrainingManager
from .core.task_store import TaskStore
from .core.review_store import ReviewStore
from .core.artifact_retention import clean_meeting_artifacts


def _reconfigure_llm_client(client: object | None, settings: Settings, *, meeting: bool = False) -> None:
    if not isinstance(client, OpenAICompatLLMClient):
        return
    client.reconfigure(
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key,
        600.0 if meeting else settings.llm_request_timeout_seconds,
        5.0 if meeting else settings.llm_connect_timeout_seconds,
        120.0 if meeting else settings.llm_stream_idle_timeout_seconds,
        chat_path=settings.llm_chat_path,
        models_path=settings.llm_models_path,
    )


def _llm_config_response(current: Settings) -> dict[str, object]:
    return {
        "llm_base_url": current.llm_base_url,
        "llm_chat_path": current.llm_chat_path,
        "llm_models_path": current.llm_models_path,
        "llm_model": current.llm_model,
        "llm_api_key": current.llm_api_key_masked,
        "llm_api_key_configured": bool(current.llm_api_key),
        "llm_correction_enabled": current.llm_correction_enabled,
        "llm_correction_max_tokens": current.llm_correction_max_tokens,
        "llm_correction_system_prompt": current.llm_correction_system_prompt,
        "meeting_llm_correction_enabled": current.meeting_llm_correction_enabled,
        "meeting_llm_correction_candidates": list(
            current.meeting_llm_correction_candidates
        ),
    }


def _runtime_config_response(current: Settings) -> dict[str, object]:
    return {
        "meeting_rule_correction_enabled": current.meeting_rule_correction_enabled,
        "asr_model": current.asr_model,
        "asr_language": current.asr_language,
        "asr_max_tokens": current.asr_max_tokens,
        "asr_repetition_penalty": current.asr_repetition_penalty,
        "asr_repetition_context_size": current.asr_repetition_context_size,
        "asr_chunk_duration_seconds": current.asr_chunk_duration_seconds,
        "asr_long_audio_threshold_seconds": current.asr_long_audio_threshold_seconds,
        "asr_worker_queue_size": current.asr_worker_queue_size,
        "asr_worker_request_timeout_seconds": current.asr_worker_request_timeout_seconds,
        "asr_worker_shutdown_timeout_seconds": current.asr_worker_shutdown_timeout_seconds,
        "asr_hotwords": list(current.asr_hotwords),
        "asr_system_prompt_configured": bool(current.asr_system_prompt),
        "punctuation_enabled": current.punctuation_enabled,
        "punctuation_engine": current.punctuation_engine,
        "punctuation_device": current.punctuation_device,
        "sd_engine": current.sd_engine,
        "speaker_match_threshold": current.speaker_match_threshold,
        "speaker_match_margin": current.speaker_match_margin,
        "speaker_max_speakers": current.speaker_max_speakers,
        "speaker_duplicate_threshold": current.speaker_duplicate_threshold,
        "meeting_max_concurrent_jobs": current.meeting_max_concurrent_jobs,
        "training_max_concurrent_jobs": current.training_max_concurrent_jobs,
        "meeting_task_timeout_seconds": current.meeting_task_timeout_seconds,
        "training_task_timeout_seconds": current.training_task_timeout_seconds,
        "meeting_artifact_retention_days": current.meeting_artifact_retention_days,
        "meeting_artifact_max_bytes": current.meeting_artifact_max_bytes,
        "transcribe_max_upload_bytes": current.transcribe_max_upload_bytes,
        "transcribe_max_duration_seconds": current.transcribe_max_duration_seconds,
        "meeting_max_upload_bytes": current.meeting_max_upload_bytes,
        "meeting_max_duration_seconds": current.meeting_max_duration_seconds,
        "ws_max_audio_bytes": current.ws_max_audio_bytes,
        "audio_max_concurrent_per_ip": current.audio_max_concurrent_per_ip,
        "audio_enhancement_enabled": current.audio_enhancement_enabled,
        "audio_ans_model": current.audio_ans_model,
        "audio_enhancement_model_dir": current.audio_enhancement_model_dir,
        "version": current.version,
    }


def _build_meeting_stages(
    current: Settings, speaker_store: SpeakerProfileStore
) -> tuple[object, object | None]:
    punctuation = build_punctuation_client(
        enabled=current.punctuation_enabled,
        engine=current.punctuation_engine,
        model=current.punctuation_model,
        device=current.punctuation_device,
    )
    sd_engine = current.sd_engine.lower()
    if sd_engine in {"", "none", "noop"}:
        diarization = None
    elif sd_engine in {"resemblyzer", "resemblyzer-ward"}:
        diarization = ResemblyzerDiarizationStage(
            profile_store=speaker_store, max_speakers=current.speaker_max_speakers
        )
    else:
        raise ValueError(f"unsupported SD engine: {current.sd_engine}")
    return punctuation, diarization


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    records = app.state.task_store.reconcile({
        "meeting": app.state.settings.meeting_task_timeout_seconds,
        "training": app.state.settings.training_task_timeout_seconds,
    })
    for record in records:
        app.state.training_manager.restore(record)
    clean_meeting_artifacts(
        base_dir=app.state.settings.meeting_output_dir,
        task_store=app.state.task_store,
        retention_days=app.state.settings.meeting_artifact_retention_days,
        max_bytes=app.state.settings.meeting_artifact_max_bytes,
    )
    yield
    worker = getattr(app.state, "asr_worker", None)
    if worker is not None:
        await worker.aclose()


def create_app(
    settings: Settings | None = None,
    pipeline: VoicePipeline | None = None,
    meeting_pipeline: MeetingPipeline | None = None,
    *,
    enable_pipeline: bool = True,
    enable_meeting: bool = True,
) -> FastAPI:
    app = FastAPI(title="Astra API", version="0.1.0", lifespan=app_lifespan)
    current = settings or Settings.from_env()
    app.state.settings = current
    app.state.audio_ip_limiter = AudioIPConcurrencyLimiter(
        current.audio_max_concurrent_per_ip
    )

    @app.middleware("http")
    async def require_admin_session(request: Request, call_next):
        if (
            request.url.path.startswith("/api/")
            and not request.url.path.startswith("/api/auth/")
            and not is_authorized_request(request)
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "administrator authentication required"},
            )
        return await call_next(request)

    app.state.task_store = TaskStore(current.task_store_path)
    app.state.review_store = ReviewStore(current.task_store_path)
    app.state.training_manager = TrainingManager(
        app.state.task_store,
        max_concurrent_jobs=current.training_max_concurrent_jobs,
        task_timeout_seconds=current.training_task_timeout_seconds,
    )
    app.state.training_config_loader = lambda: load_qwen3_training_config(
        current.qwen3_training_config_path
    )
    app.state.speaker_store = SpeakerProfileStore(
        current.speaker_store_path,
        match_threshold=current.speaker_match_threshold,
        match_margin=current.speaker_match_margin,
        duplicate_threshold=current.speaker_duplicate_threshold,
        sample_dir=current.speaker_sample_dir,
    )
    app.state.asr_worker = None
    app.state.pipeline = pipeline
    if enable_pipeline and pipeline is None:
        realtime_asr = MlxAudioAsrClient(
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
        app.state.asr_worker = AsrWorkerClient(
            realtime_asr,
            max_queue=current.asr_worker_queue_size,
            request_timeout_seconds=current.asr_worker_request_timeout_seconds,
            shutdown_timeout_seconds=current.asr_worker_shutdown_timeout_seconds,
        )
        app.state.pipeline = VoicePipeline(
            app.state.asr_worker,
            OpenAICompatLLMClient(
                current.llm_base_url,
                current.llm_model,
                current.llm_api_key,
                current.llm_request_timeout_seconds,
                current.llm_connect_timeout_seconds,
                current.llm_stream_idle_timeout_seconds,
                chat_path=current.llm_chat_path,
                models_path=current.llm_models_path,
            ),
            PiperSdkTtsClient(current.tts_model_path),
        )
    app.state.meeting_pipeline = meeting_pipeline
    if enable_meeting and meeting_pipeline is None:
        from .core.meeting import MeetingPipeline

        # 会议转写要忠实输出: 不复用带 hotwords/system_prompt 的语音
        # 对话 client(热词会诱导模型复读注入)。用干净配置的 MlxAudio
        # client——热词加权只属于实时对话场景。
        meeting_asr = MlxAudioAsrClient(
            current.asr_model,
            current.asr_language,
            max_tokens=current.asr_max_tokens,
            repetition_penalty=current.asr_repetition_penalty,
            repetition_context_size=current.asr_repetition_context_size,
            hotwords=(),
            system_prompt="",
        )
        punctuation, diarization = _build_meeting_stages(
            current, app.state.speaker_store
        )
        meeting_llm = OpenAICompatLLMClient(
            current.llm_base_url,
            current.llm_model,
            current.llm_api_key,
            request_timeout_seconds=600.0,
            connect_timeout_seconds=5.0,
            stream_idle_timeout_seconds=120.0,
            chat_path=current.llm_chat_path,
            models_path=current.llm_models_path,
        )
        correction_stage = CorrectionStage(
            meeting_llm,
            rules_enabled=current.meeting_rule_correction_enabled,
            llm_enabled=current.meeting_llm_correction_enabled,
            candidate_rules=parse_correction_rules(
                current.meeting_llm_correction_candidates
            ),
            system_prompt=current.llm_correction_system_prompt,
            max_tokens=current.llm_correction_max_tokens,
        )
        app.state.meeting_pipeline = MeetingPipeline(
            llm=meeting_llm,
            asr=meeting_asr,
            vad_model=current.vad_model,
            punctuation=punctuation,
            diarization=diarization,
            correction_stage=correction_stage,
            prompt_templates_path=current.meeting_prompt_templates_path,
        )
    app.include_router(auth_router)
    app.include_router(ws_router)
    app.include_router(http_router)
    app.include_router(meeting_router)
    app.include_router(speaker_router)
    app.include_router(notification_router)
    app.include_router(training_router)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        payload = await collect_health(
            app.state.settings, app.state.pipeline, app.state.meeting_pipeline
        )
        payload["auth"] = {
            "required": auth_required(app.state.settings),
            "session_secret_configured": bool(
                app.state.settings.admin_session_secret
            ),
        }
        return payload

    @app.get("/api/config")
    async def config() -> dict[str, object]:
        current = app.state.settings
        return {**_llm_config_response(current), **_runtime_config_response(current)}

    @app.put("/api/config")
    async def update_config(payload: LLMSettingsPayload) -> dict[str, object]:
        current = app.state.settings
        api_key = current.llm_api_key if payload.llm_api_key is None else payload.llm_api_key
        updated = replace(
            current,
            llm_base_url=(
                _normalize_base_url(payload.llm_base_url)
                if payload.llm_base_url is not None
                else current.llm_base_url
            ),
            llm_chat_path=(
                payload.llm_chat_path.strip()
                if payload.llm_chat_path is not None
                else current.llm_chat_path
            ),
            llm_models_path=(
                payload.llm_models_path.strip()
                if payload.llm_models_path is not None
                else current.llm_models_path
            ),
            llm_model=(
                payload.llm_model.strip()
                if payload.llm_model is not None
                else current.llm_model
            ),
            llm_api_key=api_key,
            llm_request_timeout_seconds=(
                payload.llm_request_timeout_seconds
                if payload.llm_request_timeout_seconds is not None
                else current.llm_request_timeout_seconds
            ),
            llm_connect_timeout_seconds=(
                payload.llm_connect_timeout_seconds
                if payload.llm_connect_timeout_seconds is not None
                else current.llm_connect_timeout_seconds
            ),
            llm_stream_idle_timeout_seconds=(
                payload.llm_stream_idle_timeout_seconds
                if payload.llm_stream_idle_timeout_seconds is not None
                else current.llm_stream_idle_timeout_seconds
            ),
            llm_correction_enabled=(
                payload.llm_correction_enabled
                if payload.llm_correction_enabled is not None
                else current.llm_correction_enabled
            ),
            llm_correction_max_tokens=(
                payload.llm_correction_max_tokens
                if payload.llm_correction_max_tokens is not None
                else current.llm_correction_max_tokens
            ),
            llm_correction_system_prompt=(
                payload.llm_correction_system_prompt
                if payload.llm_correction_system_prompt is not None
                else current.llm_correction_system_prompt
            ),
            meeting_llm_correction_enabled=(
                payload.meeting_llm_correction_enabled
                if payload.meeting_llm_correction_enabled is not None
                else current.meeting_llm_correction_enabled
            ),
            meeting_llm_correction_candidates=(
                tuple(item.strip() for item in payload.meeting_llm_correction_candidates if item.strip())
                if payload.meeting_llm_correction_candidates is not None
                else current.meeting_llm_correction_candidates
            ),
        )
        try:
            persist_llm_environment(
                llm_environment_values(updated), path=current.config_path
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"无法保存 LLM 配置: {exc}") from exc

        app.state.settings = updated
        _reconfigure_llm_client(
            getattr(app.state.pipeline, "llm", None), updated
        )
        _reconfigure_llm_client(
            getattr(app.state.meeting_pipeline, "llm", None), updated, meeting=True
        )
        _reconfigure_llm_client(
            getattr(app.state.meeting_pipeline, "mt_llm", None), updated, meeting=True
        )
        correction = getattr(app.state.meeting_pipeline, "correction_stage", None)
        if correction is not None:
            correction.llm_enabled = updated.meeting_llm_correction_enabled
            correction.candidate_rules = parse_correction_rules(
                updated.meeting_llm_correction_candidates
            )
            correction.system_prompt = updated.llm_correction_system_prompt
            correction.max_tokens = updated.llm_correction_max_tokens
            correction.engine_name = (
                "rules+llm" if correction.llm_enabled else "rules"
            )
        return {**_llm_config_response(updated), "saved": True, "runtime_applied": True}

    @app.get("/api/training/config")
    async def training_config() -> dict[str, object]:
        try:
            config = load_qwen3_training_config(
                app.state.settings.qwen3_training_config_path
            )
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "config": config.to_dict(),
            "runtime_applied": False,
            "note": "参数供离线 Qwen3-ASR SFT 使用；保存后不会在当前 API 进程中启动训练或热切换模型。",
        }

    @app.put("/api/training/config")
    async def update_training_config(payload: TrainingConfigPayload) -> dict[str, object]:
        try:
            config = save_qwen3_training_config(
                app.state.settings.qwen3_training_config_path,
                payload.to_config(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "config": config.to_dict(),
            "runtime_applied": False,
            "note": "训练配置已保存；请由独立训练脚本读取，当前 API 不会启动训练。",
        }

    return app


app = create_app()
