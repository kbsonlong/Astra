import pytest

from app.models.tts_client import PiperSdkTtsClient, TTSClientError


@pytest.mark.anyio
async def test_synthesize_uses_piper_sdk() -> None:
    class FakeVoice:
        def synthesize_wav(self, text: str, output) -> None:
            assert text == "hello"
            output.write(b"RIFFfake-wav")

    client = PiperSdkTtsClient("test.onnx", voice=FakeVoice())

    assert await client.synthesize("hello") == b"RIFFfake-wav"


@pytest.mark.anyio
async def test_synthesize_rejects_empty_text() -> None:
    client = PiperSdkTtsClient("test.onnx", voice=object())
    try:
        with pytest.raises(TTSClientError, match="empty"):
            await client.synthesize("  ")
    finally:
        pass
