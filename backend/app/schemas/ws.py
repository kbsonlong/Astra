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


class StateChange(BaseModel):
    type: Literal["state_change"] = "state_change"
    state: Literal["IDLE", "LISTENING", "REASONING", "SPEAKING"]
    generation_id: int | None = None
