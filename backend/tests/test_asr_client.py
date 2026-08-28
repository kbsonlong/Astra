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

    def fake_load_audio(path: str) -> list[float]:
        return [0.0]

    def fake_generate_transcription(**kwargs: object) -> Result:
        with open(kwargs["audio"], "rb") as audio:  # type: ignore[arg-type]
            observed["audio_bytes"] = audio.read()
        observed.update(kwargs)
        return Result()

    client = MlxAudioAsrClient(
        "test-model",
        hotwords=("host", "host 网络模式"),
        system_prompt="只输出实际说出的内容。",
        load_audio=fake_load_audio,
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
    assert observed["system_prompt"] == "只输出实际说出的内容。"
    assert observed["format"] == "txt"


@pytest.mark.anyio
async def test_transcribe_includes_sdk_error_detail() -> None:
    def fake_load_model(model: str) -> object:
        raise FileNotFoundError(model)

    client = MlxAudioAsrClient(
        "missing-model",
        load_audio=lambda path: [0.0],
        load_model=fake_load_model,
        generate_transcription=lambda **kwargs: None,
    )

    with pytest.raises(ASRClientError, match="FileNotFoundError: missing-model"):
        await client.transcribe(b"wav")


@pytest.mark.anyio
async def test_long_audio_is_transcribed_in_independent_chunks() -> None:
    observed: list[object] = []

    class Result:
        def __init__(self, text: str) -> None:
            self.text = text

    def fake_generate_transcription(**kwargs: object) -> Result:
        observed.append(kwargs["audio"])
        return Result(f"chunk-{len(observed)}")

    client = MlxAudioAsrClient(
        "test-model",
        chunk_duration=1.0,
        long_audio_threshold=1.0,
        load_audio=lambda path: [0.0] * 32000,
        load_model=lambda model: object(),
        generate_transcription=fake_generate_transcription,
    )

    assert await client.transcribe(b"audio", filename="meeting.m4a") == "chunk-1 chunk-2"
    assert len(observed) == 2
