import httpx
import pytest

from app.models.asr_client import WhisperCppAsrClient


@pytest.mark.anyio
async def test_transcribe_posts_wav_to_whisper_inference() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        assert request.url.path == "/inference"
        assert b'filename="speech.wav"' in body
        assert b"response_format" in body
        return httpx.Response(200, json={"text": "  hello world  "})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = WhisperCppAsrClient("http://asr.test", http_client)
    try:
        assert await client.transcribe(b"wav") == "hello world"
    finally:
        await http_client.aclose()
