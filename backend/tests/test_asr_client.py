import pytest

from app.models.asr_client import MlxAudioAsrClient


@pytest.mark.anyio
async def test_transcribe_uses_mlx_audio_sdk_with_temp_wav() -> None:
    observed: dict[str, object] = {}

    class Result:
        text = "  hello world  "

    def fake_load_model(model: str) -> object:
        observed["model_id"] = model
        return object()

    def fake_generate_transcription(**kwargs: object) -> Result:
        with open(kwargs["audio"], "rb") as audio:  # type: ignore[arg-type]
            observed["audio_bytes"] = audio.read()
        observed.update(kwargs)
        return Result()

    client = MlxAudioAsrClient(
        "test-model",
        load_model=fake_load_model,
        generate_transcription=fake_generate_transcription,
    )

    assert await client.transcribe(b"wav") == "hello world"
    assert observed["audio_bytes"] == b"wav"
    assert observed["model_id"] == "test-model"
    assert observed["language"] == "zh"
    assert observed["format"] == "txt"
