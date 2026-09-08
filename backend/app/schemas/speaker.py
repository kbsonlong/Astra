from pydantic import BaseModel, Field, field_validator


class SpeakerCreateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)

    @field_validator("display_name")
    @classmethod
    def trim_display_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("display_name is required")
        return value


class SpeakerUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)

    @field_validator("display_name")
    @classmethod
    def trim_display_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("display_name is required")
        return value
