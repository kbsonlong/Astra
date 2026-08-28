import pytest

from app.models.asr_client import ASRClientError, MlxAudioAsrClient


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
        hotwords=("host", "host 网络模式"),
        load_model=fake_load_model,
        generate_transcription=fake_generate_transcription,
    )

    assert await client.transcribe(b"wav", filename="test.wav") == "hello world"
    assert observed["audio_bytes"] == b"wav"
    assert observed["model_id"] == "test-model"
    assert str(observed["audio"]).endswith("/test.wav")
    assert observed["language"] == "Chinese"
    assert observed["max_tokens"] == 512
    assert observed["temperature"] == 0.0
    assert observed["repetition_penalty"] == 1.08
    assert observed["repetition_context_size"] == 100
    assert observed["chunk_duration"] == 30.0
    assert observed["hotwords"] == ["host", "host 网络模式"]
    assert observed["format"] == "txt"


@pytest.mark.anyio
async def test_transcribe_includes_sdk_error_detail() -> None:
    def fake_load_model(model: str) -> object:
        raise FileNotFoundError(model)

    client = MlxAudioAsrClient(
        "missing-model",
        load_model=fake_load_model,
        generate_transcription=lambda **kwargs: None,
    )

    with pytest.raises(ASRClientError, match="FileNotFoundError: missing-model"):
        await client.transcribe(b"wav")
