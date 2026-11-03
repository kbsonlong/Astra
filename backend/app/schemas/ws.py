from typing import Literal

from pydantic import BaseModel


class ClientMessage(BaseModel):
    type: Literal[
        "start_session",
        "speech_end",
        "interrupt",
        "end_session",
        "audio_channel",
        "audio_format",
    ]
    generation_id: int | None = None
    reason: Literal["vad", "manual"] | None = None
    channel: Literal["microphone", "reference"] | None = None
    format: Literal["pcm16"] | None = None
    sample_rate: int | None = None
    frame_samples: int | None = None
    recording_id: str | None = None
    voice_id: str | None = None
    voice_revision: int | None = None


class StateChange(BaseModel):
    type: Literal["state_change"] = "state_change"
    state: Literal["IDLE", "LISTENING", "REASONING", "SPEAKING"]
    generation_id: int | None = None
    voice_id: str | None = None
    voice_revision: int | None = None


class ServerEvent(BaseModel):
    """Base class for every outbound WebSocket event.

    Mirrors the strict validation already applied to inbound ``ClientMessage``
    so that outbound events carry a schema instead of being hand-built dicts.
    Call :meth:`to_wire` to obtain the JSON payload actually sent to clients.
    """

    def to_wire(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


class EnhancementStageStatus(BaseModel):
    stage: str
    status: str
    detail: str | None = None
    latency_ms: float | None = None


class EnhancementStatusEvent(ServerEvent):
    type: Literal["enhancement_status"] = "enhancement_status"
    generation_id: int
    stages: list[dict[str, object]]


class AsrFinalEvent(ServerEvent):
    type: Literal["asr_final"] = "asr_final"
    generation_id: int
    text: str


class LlmTokenEvent(ServerEvent):
    type: Literal["llm_token"] = "llm_token"
    generation_id: int
    token: str


class TtsStartEvent(ServerEvent):
    type: Literal["tts_start"] = "tts_start"
    generation_id: int
    seq: int


class TtsEndEvent(ServerEvent):
    type: Literal["tts_end"] = "tts_end"
    generation_id: int


class ErrorEvent(ServerEvent):
    type: Literal["error"] = "error"
    code: str
    message: str | None = None
    generation_id: int | None = None
    max_bytes: int | None = None


class RecordingStartedEvent(ServerEvent):
    type: Literal["recording_started"] = "recording_started"
    recording_id: str


class RecordingReadyEvent(ServerEvent):
    type: Literal["recording_ready"] = "recording_ready"
    recording_id: str
    download_url: str


class AudioFormatReadyEvent(ServerEvent):
    type: Literal["audio_format_ready"] = "audio_format_ready"
    format: Literal["pcm16"] = "pcm16"
    sample_rate: int
    frame_samples: int
