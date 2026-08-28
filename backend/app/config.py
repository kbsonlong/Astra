from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://192.168.3.18:8000/v1"
    llm_chat_path: str = "/chat/completions"
    llm_models_path: str = "/models"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_request_timeout_seconds: float = 120.0
    llm_connect_timeout_seconds: float = 3.0
    llm_stream_idle_timeout_seconds: float = 15.0
    llm_correction_enabled: bool = True
    llm_correction_max_tokens: int = 256
    llm_correction_system_prompt: str = (
        "你是中文语音识别纠错器。只修正明显的同音词、技术术语、人名和专有名词错误。"
        "不新增、不删减、不总结，只输出修正后的文本。"
        "术语映射：后视网络模式->host 网络模式；后视->host；单科->单机；多科->多机；集群机->集群。"
        "保留技术术语中英文之间的空格。"
    )
    asr_model: str = "mlx-community/Qwen3-ASR-0.6B-4bit"
    asr_language: str = "Chinese"
    asr_max_tokens: int = 512
    asr_repetition_penalty: float = 1.08
    asr_repetition_context_size: int = 100
    asr_chunk_duration_seconds: float = 30.0
    asr_long_audio_threshold_seconds: float = 60.0
    asr_hotwords: tuple[str, ...] = ()
    asr_system_prompt: str = ""
    tts_model_path: str = "models/zh_CN-huayan-medium.onnx"
    version: str = "mvp"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url).rstrip("/"),
            llm_chat_path=os.getenv("LLM_CHAT_PATH", cls.llm_chat_path),
            llm_models_path=os.getenv("LLM_MODELS_PATH", cls.llm_models_path),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_api_key=os.getenv("LLM_API_KEY", cls.llm_api_key),
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
            asr_hotwords=_csv_env("ASR_HOTWORDS", cls.asr_hotwords),
            asr_system_prompt=os.getenv("ASR_SYSTEM_PROMPT", cls.asr_system_prompt),
            tts_model_path=os.getenv("TTS_MODEL_PATH", cls.tts_model_path),
            version=os.getenv("ASTRA_VERSION", cls.version),
        )

    @property
    def llm_api_key_masked(self) -> str:
        if not self.llm_api_key:
            return ""
        if len(self.llm_api_key) <= 8:
            return "********"
        return f"{self.llm_api_key[:4]}...{self.llm_api_key[-4:]}"
