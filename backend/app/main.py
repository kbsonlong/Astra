import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

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
from .core.tts_router import TtsRouter
from .core.correction import parse_correction_rules
from .core.workflow import CorrectionStage, ResemblyzerDiarizationStage
from .core.zipenhancer import build_audio_enhancement_pipeline
from .models.asr_client import MlxAudioAsrClient
from .models.asr_worker import AsrWorkerClient
from .models.llm_client import OpenAICompatLLMClient
from .models.tts_client import (
    CosyVoiceTtsClient,
    IndexTTS25MlxClient,
    MlxAudioTtsClient,
    PiperSdkTtsClient,
)
from .models.punctuation_client import build_punctuation_client
from .core.speaker_registry import SpeakerProfileStore
from .main_types import LLMSettingsPayload, TrainingConfigPayload
from .api.training_routes import router as training_router
from .api.realtime_recording_routes import router as realtime_recording_router
from .api.tts_voice_routes import router as tts_voice_router
from .api.upload_limits import AudioIPConcurrencyLimiter
from .core.training import TrainingManager
from .core.task_store import TaskStore
from .core.review_store import ReviewStore
from .core.tts_voice_store import TtsVoiceStore
from .core.artifact_retention import clean_meeting_artifacts
from .core.realtime_recording import clean_expired_realtime_recordings


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
    # The exposed field set now lives in config.RUNTIME_CONFIG_FIELDS as a
    # domain-grouped allowlist; this stays as a thin shim for existing callers.
    return current.runtime_config_public()


def _build_realtime_asr(current: Settings) -> object:
    """按 asr_backend 选择实时 ASR client。"""
    if current.asr_backend == "funasr":
        from .models.funasr_client import FunAsrClient

        return FunAsrClient(
            current.asr_funasr_model,
            language=current.asr_funasr_language,
            diarize=False,
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


def _build_tts_engine(current: Settings, backend: str) -> object:
    """按 tts_backend 选择实例化 TTS client (VoiceStudio 路线图方向一)。

    piper (默认): 保留现有 Piper 兜底；mlx_audio: 中文原生 MLX；
    cosyvoice: 零样本克隆（参考音频只在服务器本地读取）。
    """
    if backend == "mlx_audio":
        return MlxAudioTtsClient(
            current.tts_mlx_model,
            voice=current.tts_mlx_voice,
            lang_code=current.tts_mlx_lang_code,
            speed=current.tts_mlx_speed,
            sample_rate=current.tts_mlx_sample_rate,
        )
    if backend == "cosyvoice":
        return CosyVoiceTtsClient(
            current.tts_cosyvoice_model,
            prompt_wav=current.tts_cosyvoice_prompt_wav,
            prompt_text=current.tts_cosyvoice_prompt_text,
            language=current.tts_cosyvoice_language,
            speed=current.tts_cosyvoice_speed,
        )
    if backend == "indextts_mlx":
        return IndexTTS25MlxClient(
            current.tts_indextts_model_dir,
            repo_id=current.tts_indextts_repo_id,
            model_revision=current.tts_indextts_model_revision,
            use_normalization=current.tts_indextts_use_normalization,
            allow_download=current.tts_indextts_allow_download,
        )
    return PiperSdkTtsClient(current.tts_model_path)


def _build_tts_client(current: Settings) -> object:
    """Build the legacy single engine or the configured multi-engine router."""
    if not current.tts_router_enabled:
        return _build_tts_engine(current, current.tts_backend)
    engines = {
        backend: _build_tts_engine(current, backend)
        for backend in current.tts_enabled_backends
    }
    return TtsRouter(
        engines,
        default_backend=current.tts_default_backend,
        fallback_backend=current.tts_default_fallback_backend,
        max_queue=current.tts_engine_queue_size,
        user_enabled_backends=current.tts_user_enabled_backends,
        indextts_validated=current.tts_indextts_validated,
    )


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
    want_workbench = getattr(app.state, "want_workbench", True)
    want_realtime = getattr(app.state, "want_realtime", True)
    supervisor: asyncio.Task[None] | None = None
    recording_cleaner: asyncio.Task[None] | None = None
    supervisor_id = uuid4().hex
    app.state.supervisor_id = supervisor_id

    # Workbench owns the task registry: reconcile/restore in-flight jobs and run
    # the meeting-artifact cleanup under a single supervisor lease.
    if want_workbench and app.state.task_store is not None:
        owns_supervisor_lease = await asyncio.to_thread(
            app.state.task_store.try_acquire_supervisor, supervisor_id
        )
        records = (
            await asyncio.to_thread(
                app.state.task_store.reconcile,
                {
                    "meeting": app.state.settings.meeting_task_timeout_seconds,
                    "training": app.state.settings.training_task_timeout_seconds,
                },
            )
            if owns_supervisor_lease
            else app.state.task_store.processing_tasks()
        )
        for record in records:
            app.state.training_manager.restore(record)
        if owns_supervisor_lease:
            clean_meeting_artifacts(
                base_dir=app.state.settings.meeting_output_dir,
                task_store=app.state.task_store,
                retention_days=app.state.settings.meeting_artifact_retention_days,
                max_bytes=app.state.settings.meeting_artifact_max_bytes,
            )
        supervisor = asyncio.create_task(_task_supervisor(app, supervisor_id))

    # Realtime owns its recording directory cleanup.
    if want_realtime:
        clean_expired_realtime_recordings(
            app.state.settings.realtime_recording_dir,
            app.state.settings.realtime_recording_retention_days,
        )
        # A realtime-only process runs its own periodic cleaner; when it shares
        # a process with workbench ("all"), the supervisor handles the recurring
        # cleanup instead.
        if not want_workbench:
            recording_cleaner = asyncio.create_task(
                _realtime_recording_cleaner(app)
            )
    try:
        yield
    finally:
        if recording_cleaner is not None:
            recording_cleaner.cancel()
            await asyncio.gather(recording_cleaner, return_exceptions=True)
        if supervisor is not None:
            supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)
            await asyncio.to_thread(
                app.state.task_store.release_supervisor, supervisor_id
            )
    worker = getattr(app.state, "asr_worker", None)
    if worker is not None:
        await worker.aclose()
    tts = getattr(getattr(app.state, "pipeline", None), "tts", None)
    close_tts = getattr(tts, "aclose", None)
    if callable(close_tts):
        await close_tts()


async def _task_supervisor(app: FastAPI, supervisor_id: str) -> None:
    """收敛无浏览器连接时的退出、超时和异常任务。"""
    # When this process also serves realtime (profile "all"), the supervisor
    # keeps cleaning recordings as before. A workbench-only process leaves
    # recording cleanup to the realtime process's own cleaner.
    also_clean_recordings = getattr(app.state, "want_realtime", True)
    last_cleanup = 0.0
    while True:
        try:
            owns_lease = await asyncio.to_thread(
                app.state.task_store.try_acquire_supervisor, supervisor_id
            )
            if owns_lease:
                await asyncio.to_thread(
                    app.state.task_store.reconcile,
                    {
                        "meeting": app.state.settings.meeting_task_timeout_seconds,
                        "training": app.state.settings.training_task_timeout_seconds,
                    },
                )
                now = time.monotonic()
                if now - last_cleanup >= 60:
                    await asyncio.to_thread(
                        clean_meeting_artifacts,
                        base_dir=app.state.settings.meeting_output_dir,
                        task_store=app.state.task_store,
                        retention_days=app.state.settings.meeting_artifact_retention_days,
                        max_bytes=app.state.settings.meeting_artifact_max_bytes,
                    )
                    if also_clean_recordings:
                        await asyncio.to_thread(
                            clean_expired_realtime_recordings,
                            app.state.settings.realtime_recording_dir,
                            app.state.settings.realtime_recording_retention_days,
                        )
                    last_cleanup = now
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A transient SQLite/filesystem error must not kill supervision.
            await asyncio.sleep(1)


async def _realtime_recording_cleaner(app: FastAPI) -> None:
    """Periodically expire realtime recordings in a realtime-only process.

    The workbench supervisor owns this cleanup when both roles share a process
    (profile "all"); a dedicated realtime process needs its own loop.
    """
    while True:
        try:
            await asyncio.to_thread(
                clean_expired_realtime_recordings,
                app.state.settings.realtime_recording_dir,
                app.state.settings.realtime_recording_retention_days,
            )
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(5)

def create_app(
    settings: Settings | None = None,
    pipeline: VoicePipeline | None = None,
    meeting_pipeline: MeetingPipeline | None = None,
    *,
    enable_pipeline: bool = True,
    enable_meeting: bool = True,
    profile: str = "all",
) -> FastAPI:
    # Profiles let the realtime voice link and the heavier workbench
    # (meeting / training / voice library / speaker) run as separate processes
    # that no longer contend for GPU/CPU in one app. "all" keeps the original
    # single-process behavior and is the default for every existing caller.
    if profile not in {"all", "realtime", "workbench"}:
        raise ValueError(
            f"profile must be one of all/realtime/workbench, got {profile!r}"
        )
    want_realtime = profile in {"all", "realtime"}
    want_workbench = profile in {"all", "workbench"}
    app = FastAPI(title="Astra API", version="0.1.0", lifespan=app_lifespan)
    current = settings or Settings.from_env()
    app.state.profile = profile
    app.state.want_realtime = want_realtime
    app.state.want_workbench = want_workbench
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

    # Workbench owns the task registry, training and speaker stores. The
    # realtime profile skips them so it does not run meeting/training
    # supervision.
    app.state.task_store = None
    app.state.review_store = None
    app.state.training_manager = None
    app.state.training_config_loader = None
    app.state.speaker_store = None
    if want_workbench:
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
    # The voice library is written by workbench but also read (snapshot) by the
    # realtime ws session, so both profiles open the shared store.
    app.state.tts_voice_store = TtsVoiceStore(
        current.tts_voice_store_path,
        current.tts_voice_audio_dir,
    )
    app.state.asr_worker = None
    app.state.pipeline = pipeline
    if want_realtime and enable_pipeline and pipeline is None:
        from .core.jaec import build_audio_aec_pipeline

        realtime_asr = _build_realtime_asr(current)
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
            _build_tts_client(current),
            enhancement=build_audio_aec_pipeline(
                enabled=current.audio_enhancement_enabled,
                aec_model=current.audio_aec_model,
                model_dir=current.audio_enhancement_model_dir,
            ),
            enhancement_max_queue=current.audio_enhancement_max_queue,
            enhancement_frame_timeout_ms=current.audio_enhancement_frame_timeout_ms,
        )
    app.state.meeting_pipeline = meeting_pipeline
    if want_workbench and enable_meeting and meeting_pipeline is None:
        from .core.meeting import MeetingPipeline
        from .core.audio_separation import (
            build_audio_separation_stage,
            build_overlap_detector,
        )

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
        inline_asr = None
        if current.asr_backend == "funasr":
            from .models.funasr_client import FunAsrClient

            inline_asr = FunAsrClient(
                current.asr_funasr_model,
                language=current.asr_funasr_language,
                diarize=current.asr_funasr_diarize,
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
            enhancement=build_audio_enhancement_pipeline(
                enabled=current.audio_enhancement_enabled,
                ans_model=current.audio_ans_model,
                model_dir=current.audio_enhancement_model_dir,
                device=current.audio_enhancement_device,
            ),
            separation=build_audio_separation_stage(
                enabled=current.audio_enhancement_enabled,
                separation_model=current.audio_separation_model,
                model_dir=current.audio_enhancement_model_dir,
                window_seconds=current.audio_separation_window_seconds,
                device=current.audio_enhancement_device,
            ),
            separation_trigger=current.audio_separation_trigger,
            overlap_detector=build_overlap_detector(
                model=current.audio_overlap_detector_model,
                model_dir=current.audio_overlap_model_dir,
            ),
            inline_asr=inline_asr,
            speaker_store=app.state.speaker_store,
        )
    app.include_router(auth_router)
    # Voice-library routes are needed in both profiles: realtime reads snapshots,
    # workbench manages the library.
    app.include_router(tts_voice_router)
    if want_realtime:
        app.include_router(ws_router)
        app.include_router(http_router)
        app.include_router(realtime_recording_router)
    if want_workbench:
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

    # Offline Qwen3-ASR training config is a workbench concern; a realtime-only
    # process does not expose it.
    if want_workbench:

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


def _default_app() -> FastAPI:
    # The ASGI app honors ASTRA_PROFILE (all/realtime/workbench) so a split
    # deployment runs the same `uvicorn app.main:app` command per process,
    # varying only the profile and port. Defaults to "all" for single-process
    # setups. Build Settings first so a value set only in the dotenv file (not a
    # real env var) is loaded before the profile is resolved.
    settings = Settings.from_env()
    return create_app(settings, profile=os.getenv("ASTRA_PROFILE", "all"))


app = _default_app()
