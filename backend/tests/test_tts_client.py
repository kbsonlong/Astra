import httpx
import pytest

from app.models.tts_client import PiperHttpTtsClient, TTSClientError


@pytest.mark.anyio
async def test_synthesize_returns_piper_wav_bytes() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/synthesize"
        assert (await request.aread()) == b'{"text":"hello"}'
        return httpx.Response(200, content=b"RIFFfake-wav")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = PiperHttpTtsClient("http://tts.test", http_client)
    try:
        assert await client.synthesize("hello") == b"RIFFfake-wav"
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_synthesize_rejects_empty_text() -> None:
    client = PiperHttpTtsClient("http://tts.test", httpx.AsyncClient())
    try:
        with pytest.raises(TTSClientError, match="empty"):
            await client.synthesize("  ")
    finally:
        await client._client.aclose()
