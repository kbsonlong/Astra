from dataclasses import dataclass
import os


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


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
    asr_endpoint: str = "http://whisper-asr:8080"
    tts_endpoint: str = "http://piper-tts:8080"
    version: str = "mvp"

    @classmethod
    def from_env(cls) -> "Settings":
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
            asr_endpoint=os.getenv("ASR_ENDPOINT", cls.asr_endpoint).rstrip("/"),
            tts_endpoint=os.getenv("TTS_ENDPOINT", cls.tts_endpoint).rstrip("/"),
            version=os.getenv("ASTRA_VERSION", cls.version),
        )

    @property
    def llm_api_key_masked(self) -> str:
        if not self.llm_api_key:
            return ""
        if len(self.llm_api_key) <= 8:
            return "********"
        return f"{self.llm_api_key[:4]}...{self.llm_api_key[-4:]}"
