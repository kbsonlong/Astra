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
        "术语映射：后视网络模式->host 网络模式；后视->host；单科->单机；多科->多机；集群机->集群；宗师->忠思；下限->下线；冷资源中心->云资源中心；光单->关单；空单->工单；谷歌表哥->Google 表格；嫁给豆包->交给豆包。"
        "保留技术术语中英文之间的空格。"
    )
    asr_engine: str = "mlx"
    asr_model: str = "mlx-community/Qwen3-ASR-0.6B-4bit"
    asr_language: str = "Chinese"
    # 会议管线: whisper-large-v3-turbo 本地目录(默认 ModelScope 缓存)
    whisper_model_path: str = (
        "/Users/kbsonlong/.cache/modelscope/hub/models/mlx-community/whisper-large-v3-turbo"
    )
    asr_max_tokens: int = 512
    asr_repetition_penalty: float = 1.08
    asr_repetition_context_size: int = 100
    asr_chunk_duration_seconds: float = 30.0
    asr_long_audio_threshold_seconds: float = 60.0
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
    sherpa_model_dir: str = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    sherpa_num_threads: int = 2
    sherpa_provider: str = "cpu"
    sherpa_auto_language: bool = True
    sherpa_use_itn: bool = True
    zipformer_model_dir: str = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    zipformer_num_threads: int = 2
    zipformer_provider: str = "cpu"
    zipformer_decoding_method: str = "greedy_search"
    tts_model_path: str = "models/zh_CN-huayan-medium.onnx"
    meeting_output_dir: str = "~/Astra/meetings"
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
            asr_engine=os.getenv("ASR_ENGINE", cls.asr_engine),
            asr_model=os.getenv("ASR_MODEL", cls.asr_model),
            asr_language=os.getenv("ASR_LANGUAGE", cls.asr_language),
            whisper_model_path=os.getenv(
                "WHISPER_MODEL_PATH", cls.whisper_model_path
            ),
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
            vad_model=os.getenv("VAD_MODEL", cls.vad_model),
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
            sherpa_model_dir=os.getenv("SHERPA_MODEL_DIR", cls.sherpa_model_dir),
            sherpa_num_threads=_int_env("SHERPA_NUM_THREADS", cls.sherpa_num_threads),
            sherpa_provider=os.getenv("SHERPA_PROVIDER", cls.sherpa_provider),
            sherpa_auto_language=_bool_env("SHERPA_AUTO_LANGUAGE", cls.sherpa_auto_language),
            sherpa_use_itn=_bool_env("SHERPA_USE_ITN", cls.sherpa_use_itn),
            zipformer_model_dir=os.getenv("ZIPFORMER_MODEL_DIR", cls.zipformer_model_dir),
            zipformer_num_threads=_int_env("ZIPFORMER_NUM_THREADS", cls.zipformer_num_threads),
            zipformer_provider=os.getenv("ZIPFORMER_PROVIDER", cls.zipformer_provider),
            zipformer_decoding_method=os.getenv(
                "ZIPFORMER_DECODING_METHOD", cls.zipformer_decoding_method
            ),
            tts_model_path=os.getenv("TTS_MODEL_PATH", cls.tts_model_path),
            meeting_output_dir=os.getenv("MEETING_OUTPUT_DIR", cls.meeting_output_dir),
            version=os.getenv("ASTRA_VERSION", cls.version),
        )

    @property
    def llm_api_key_masked(self) -> str:
        if not self.llm_api_key:
            return ""
        if len(self.llm_api_key) <= 8:
            return "********"
        return f"{self.llm_api_key[:4]}...{self.llm_api_key[-4:]}"
