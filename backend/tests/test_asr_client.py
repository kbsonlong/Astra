import pytest

from app.models.asr_client import MlxWhisperAsrClient


@pytest.mark.anyio
async def test_transcribe_uses_mlx_whisper_sdk_with_temp_wav() -> None:
    observed: dict[str, object] = {}

    def fake_transcribe(path: str, **kwargs: object) -> dict[str, str]:
        with open(path, "rb") as audio:
            observed["audio"] = audio.read()
        observed.update(kwargs)
        return {"text": "  hello world  "}

    client = MlxWhisperAsrClient("test-model", transcribe=fake_transcribe)

    assert await client.transcribe(b"wav") == "hello world"
    assert observed["audio"] == b"wav"
    assert observed["path_or_hf_repo"] == "test-model"
    assert observed["language"] == "zh"
