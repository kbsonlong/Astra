from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


CONSENT_CONFIRMATION = "本人已授权 Astra 在本机进行语音合成"


class VoiceMetadataPatch(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)

    @field_validator("display_name")
    @classmethod
    def trim_display_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("display_name is required")
        return value


class ConsentRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=256)
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("confirmation")
    @classmethod
    def trim_confirmation(cls, value: str) -> str:
        return value.strip()

    @field_validator("note")
    @classmethod
    def trim_note(cls, value: str | None) -> str | None:
        value = value.strip() if value else None
        return value or None


class ConsentRevokeRequest(BaseModel):
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("note")
    @classmethod
    def trim_note(cls, value: str | None) -> str | None:
        value = value.strip() if value else None
        return value or None
