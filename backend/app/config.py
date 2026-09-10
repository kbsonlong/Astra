from dataclasses import asdict, dataclass, fields
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv, set_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _positive_int_env(name: str, default: int) -> int:
    value = _int_env(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = _float_env(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}




@dataclass(frozen=True)
class Qwen3TrainingConfig:
    """Offline Qwen3-ASR SFT parameters; separate from runtime inference settings."""

    model_path: str = "Qwen/Qwen3-ASR-0.6B"
    train_file: str = "~/Astra/data/asr/train.jsonl"
    eval_file: str = "~/Astra/data/asr/dev.jsonl"
    output_dir: str = "~/Astra/models/qwen3-asr-meeting"
    device: str = "cuda"
    precision: str = "bf16"
    batch_size: int = 1
    grad_acc: int = 8
    learning_rate: float = 2e-5
    epochs: int = 1
    save_steps: int = 200
    save_total_limit: int = 3
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    resume_from: str = ""
    resume_latest: bool = False

    def validate(self) -> "Qwen3TrainingConfig":
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
        for name in ("train_file", "eval_file", "output_dir"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be one of auto, cuda, mps, cpu")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be one of bf16, fp16, fp32")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        if not 1 <= self.grad_acc <= 1024:
            raise ValueError("grad_acc must be between 1 and 1024")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be greater than 0 and at most 1")
        if not 1 <= self.epochs <= 100:
            raise ValueError("epochs must be between 1 and 100")
        if not 1 <= self.save_steps <= 1_000_000:
            raise ValueError("save_steps must be between 1 and 1000000")
        if not 1 <= self.save_total_limit <= 100:
            raise ValueError("save_total_limit must be between 1 and 100")
        if not 0 <= self.num_workers <= 64:
            raise ValueError("num_workers must be between 0 and 64")
        if not 1 <= self.prefetch_factor <= 32:
            raise ValueError("prefetch_factor must be between 1 and 32")
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Qwen3TrainingConfig":
        names = {field.name for field in fields(cls)}
        data = {name: value[name] for name in names if name in value}
        return cls(**data).validate()  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_qwen3_training_config(path: str | Path) -> Qwen3TrainingConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        return Qwen3TrainingConfig().validate()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read training config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("training config must be a JSON object")
    return Qwen3TrainingConfig.from_mapping(payload)


def save_qwen3_training_config(
    path: str | Path, config: Qwen3TrainingConfig
) -> Qwen3TrainingConfig:
    config.validate()
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, config_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return config


def _project_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = "/" + "/".join(part for part in parsed.path.split("/") if part)
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


def persist_llm_environment(values: Mapping[str, str], path: str | Path | None = None) -> None:
    """Persist managed LLM variables and make them visible to child workers."""
    env_path = Path(path or (Path.cwd() / ".env")).expanduser()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in values.items():
        set_key(str(env_path), key, value, quote_mode="auto")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://192.168.3.18:8000/v1"
    llm_chat_path: str = "/chat/completions"
    llm_models_path: str = "/models"
    llm_model: str = ""
    llm_api_key: str = ""
    # 留空时保持兼容模式；对局域网开放前必须设置 ADMIN_TOKEN。
    admin_token: str = ""
    admin_session_secret: str = ""
    admin_session_ttl_seconds: int = 12 * 60 * 60
    auth_cookie_secure: bool = False
    llm_request_timeout_seconds: float = 120.0
    llm_connect_timeout_seconds: float = 3.0
    llm_stream_idle_timeout_seconds: float = 15.0
    llm_correction_enabled: bool = True
    llm_correction_max_tokens: int = 256
    llm_correction_system_prompt: str = (
        "你是中文语音识别纠错器。只修正明显的同音词、技术术语、人名和专有名词错误。"
        "不新增、不删减、不总结，只输出修正后的文本。"
        "术语映射：后视网络模式->host 网络模式；后视->host；单科->单机；多科->多机；宗师->忠思；下限->下线；冷资源中心->云资源中心；光单->关单；空单->工单；谷歌表哥->Google 表格；嫁给豆包->交给豆包。"
        "保留技术术语中英文之间的空格。"
    )
    meeting_rule_correction_enabled: bool = True
    meeting_llm_correction_enabled: bool = False
    meeting_llm_correction_candidates: tuple[str, ...] = ()
    asr_model: str = "mlx-community/Qwen3-ASR-0.6B-4bit"
    asr_language: str = "Chinese"
    asr_max_tokens: int = 512
    asr_repetition_penalty: float = 1.08
    asr_repetition_context_size: int = 100
    asr_chunk_duration_seconds: float = 30.0
    asr_long_audio_threshold_seconds: float = 60.0
    asr_worker_queue_size: int = 8
    asr_worker_request_timeout_seconds: float = 120.0
    asr_worker_shutdown_timeout_seconds: float = 5.0
    asr_hotwords: tuple[str, ...] = ()
    asr_system_prompt: str = ""
    # 离线会议 Workflow: VAD -> ASR -> punctuation -> speaker diarization
    vad_model: str = "models/silero_vad.onnx"
    punctuation_enabled: bool = True
    punctuation_engine: str = "passthrough"
    punctuation_model: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    punctuation_device: str = "mps"
    sd_engine: str = "resemblyzer"
    speaker_store_path: str = "~/.astra/speakers.sqlite3"
    speaker_match_threshold: float = 0.75
    speaker_match_margin: float = 0.05
    speaker_max_speakers: int = 32
    speaker_duplicate_threshold: float = 0.82
    speaker_sample_dir: str = "~/.astra/speaker_samples"
    meeting_prompt_templates_path: str = "~/.astra/meeting-prompt-templates.json"
    tts_model_path: str = "models/zh_CN-huayan-medium.onnx"
    meeting_output_dir: str = "~/Astra/meetings"
    task_store_path: str = "~/.astra/tasks.sqlite3"
    meeting_max_concurrent_jobs: int = 1
    training_max_concurrent_jobs: int = 1
    meeting_task_timeout_seconds: float = 2 * 60 * 60
    training_task_timeout_seconds: float = 12 * 60 * 60
    transcribe_max_upload_bytes: int = 25 * 1024 * 1024
    meeting_max_upload_bytes: int = 500 * 1024 * 1024
    ws_max_audio_bytes: int = 25 * 1024 * 1024
    qwen3_training_config_path: str = "~/.astra/qwen3-asr-training.json"
    version: str = "mvp"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        return cls(
            llm_base_url=_normalize_base_url(
                os.getenv("LLM_BASE_URL", cls.llm_base_url)
            ),
            llm_chat_path=os.getenv("LLM_CHAT_PATH", cls.llm_chat_path),
            llm_models_path=os.getenv("LLM_MODELS_PATH", cls.llm_models_path),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_api_key=os.getenv("LLM_API_KEY", cls.llm_api_key),
            admin_token=os.getenv("ADMIN_TOKEN", cls.admin_token),
            admin_session_secret=os.getenv(
                "ADMIN_SESSION_SECRET", cls.admin_session_secret
            ),
            admin_session_ttl_seconds=_positive_int_env(
                "ADMIN_SESSION_TTL_SECONDS", cls.admin_session_ttl_seconds
            ),
            auth_cookie_secure=_bool_env(
                "AUTH_COOKIE_SECURE", cls.auth_cookie_secure
            ),
            llm_request_timeout_seconds=_float_env(
                "LLM_REQUEST_TIMEOUT_SECONDS", cls.llm_request_timeout_seconds
            ),
            llm_connect_timeout_seconds=_float_env(
                "LLM_CONNECT_TIMEOUT_SECONDS", cls.llm_connect_timeout_seconds
            ),
            llm_stream_idle_timeout_seconds=_float_env(
                "LLM_STREAM_IDLE_TIMEOUT_SECONDS", cls.llm_stream_idle_timeout_seconds
            ),
            llm_correction_enabled=_bool_env(
                "LLM_CORRECTION_ENABLED", cls.llm_correction_enabled
            ),
            llm_correction_max_tokens=_int_env(
                "LLM_CORRECTION_MAX_TOKENS", cls.llm_correction_max_tokens
            ),
            llm_correction_system_prompt=os.getenv(
                "LLM_CORRECTION_SYSTEM_PROMPT", cls.llm_correction_system_prompt
            ),
            meeting_rule_correction_enabled=_bool_env(
                "MEETING_RULE_CORRECTION_ENABLED", cls.meeting_rule_correction_enabled
            ),
            meeting_llm_correction_enabled=_bool_env(
                "MEETING_LLM_CORRECTION_ENABLED", cls.meeting_llm_correction_enabled
            ),
            meeting_llm_correction_candidates=_csv_env(
                "MEETING_LLM_CORRECTION_CANDIDATES",
                cls.meeting_llm_correction_candidates,
            ),
            asr_model=os.getenv("ASR_MODEL", cls.asr_model),
            asr_language=os.getenv("ASR_LANGUAGE", cls.asr_language),
            asr_max_tokens=_int_env("ASR_MAX_TOKENS", cls.asr_max_tokens),
            asr_repetition_penalty=_float_env(
                "ASR_REPETITION_PENALTY", cls.asr_repetition_penalty
            ),
            asr_repetition_context_size=_int_env(
                "ASR_REPETITION_CONTEXT_SIZE", cls.asr_repetition_context_size
            ),
            asr_chunk_duration_seconds=_float_env(
                "ASR_CHUNK_DURATION_SECONDS", cls.asr_chunk_duration_seconds
            ),
            asr_long_audio_threshold_seconds=_float_env(
                "ASR_LONG_AUDIO_THRESHOLD_SECONDS",
                cls.asr_long_audio_threshold_seconds,
            ),
            asr_worker_queue_size=_positive_int_env(
                "ASR_WORKER_QUEUE_SIZE", cls.asr_worker_queue_size
            ),
            asr_worker_request_timeout_seconds=_positive_float_env(
                "ASR_WORKER_REQUEST_TIMEOUT_SECONDS",
                cls.asr_worker_request_timeout_seconds,
            ),
            asr_worker_shutdown_timeout_seconds=_positive_float_env(
                "ASR_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
                cls.asr_worker_shutdown_timeout_seconds,
            ),
            asr_hotwords=_csv_env("ASR_HOTWORDS", cls.asr_hotwords),
            asr_system_prompt=os.getenv("ASR_SYSTEM_PROMPT", cls.asr_system_prompt),
            vad_model=_project_path(os.getenv("VAD_MODEL", cls.vad_model)),
            punctuation_enabled=_bool_env(
                "PUNCTUATION_ENABLED", cls.punctuation_enabled
            ),
            punctuation_engine=os.getenv(
                "PUNCTUATION_ENGINE", cls.punctuation_engine
            ),
            punctuation_model=os.getenv(
                "PUNCTUATION_MODEL", cls.punctuation_model
            ),
            punctuation_device=os.getenv(
                "PUNCTUATION_DEVICE", cls.punctuation_device
            ),
            sd_engine=os.getenv("SD_ENGINE", cls.sd_engine),
            speaker_store_path=os.getenv("SPEAKER_STORE_PATH", cls.speaker_store_path),
            speaker_match_threshold=_float_env(
                "SPEAKER_MATCH_THRESHOLD", cls.speaker_match_threshold
            ),
            speaker_match_margin=_float_env(
                "SPEAKER_MATCH_MARGIN", cls.speaker_match_margin
            ),
            speaker_max_speakers=_int_env(
                "SPEAKER_MAX_SPEAKERS", cls.speaker_max_speakers
            ),
            speaker_duplicate_threshold=_float_env(
                "SPEAKER_DUPLICATE_THRESHOLD", cls.speaker_duplicate_threshold
            ),
            speaker_sample_dir=os.getenv("SPEAKER_SAMPLE_DIR", cls.speaker_sample_dir),
            meeting_prompt_templates_path=os.getenv(
                "MEETING_PROMPT_TEMPLATES_PATH", cls.meeting_prompt_templates_path
            ),
            tts_model_path=os.getenv("TTS_MODEL_PATH", cls.tts_model_path),
            meeting_output_dir=os.getenv("MEETING_OUTPUT_DIR", cls.meeting_output_dir),
            task_store_path=os.getenv("TASK_STORE_PATH", cls.task_store_path),
            meeting_max_concurrent_jobs=_positive_int_env(
                "MEETING_MAX_CONCURRENT_JOBS", cls.meeting_max_concurrent_jobs
            ),
            training_max_concurrent_jobs=_positive_int_env(
                "TRAINING_MAX_CONCURRENT_JOBS", cls.training_max_concurrent_jobs
            ),
            meeting_task_timeout_seconds=_positive_float_env(
                "MEETING_TASK_TIMEOUT_SECONDS", cls.meeting_task_timeout_seconds
            ),
            training_task_timeout_seconds=_positive_float_env(
                "TRAINING_TASK_TIMEOUT_SECONDS", cls.training_task_timeout_seconds
            ),
            transcribe_max_upload_bytes=_positive_int_env(
                "TRANSCRIBE_MAX_UPLOAD_BYTES", cls.transcribe_max_upload_bytes
            ),
            meeting_max_upload_bytes=_positive_int_env(
                "MEETING_MAX_UPLOAD_BYTES", cls.meeting_max_upload_bytes
            ),
            ws_max_audio_bytes=_positive_int_env(
                "WS_MAX_AUDIO_BYTES", cls.ws_max_audio_bytes
            ),
            qwen3_training_config_path=os.getenv(
                "QWEN3_TRAINING_CONFIG_PATH", cls.qwen3_training_config_path
            ),
            version=os.getenv("ASTRA_VERSION", cls.version),
        )

    @property
    def llm_api_key_masked(self) -> str:
        if not self.llm_api_key:
            return ""
        if len(self.llm_api_key) <= 8:
            return "********"
        return f"{self.llm_api_key[:4]}...{self.llm_api_key[-4:]}"


def llm_environment_values(settings: Settings) -> dict[str, str]:
    """Build LLM environment values for a newly spawned worker."""
    return {
        "LLM_BASE_URL": settings.llm_base_url,
        "LLM_CHAT_PATH": settings.llm_chat_path,
        "LLM_MODELS_PATH": settings.llm_models_path,
        "LLM_MODEL": settings.llm_model,
        "LLM_API_KEY": settings.llm_api_key,
        "LLM_REQUEST_TIMEOUT_SECONDS": str(settings.llm_request_timeout_seconds),
        "LLM_CONNECT_TIMEOUT_SECONDS": str(settings.llm_connect_timeout_seconds),
        "LLM_STREAM_IDLE_TIMEOUT_SECONDS": str(settings.llm_stream_idle_timeout_seconds),
        "LLM_CORRECTION_ENABLED": str(settings.llm_correction_enabled).lower(),
        "LLM_CORRECTION_MAX_TOKENS": str(settings.llm_correction_max_tokens),
        "LLM_CORRECTION_SYSTEM_PROMPT": settings.llm_correction_system_prompt,
        "MEETING_LLM_CORRECTION_ENABLED": str(
            settings.meeting_llm_correction_enabled
        ).lower(),
        "MEETING_LLM_CORRECTION_CANDIDATES": ",".join(
            settings.meeting_llm_correction_candidates
        ),
    }
