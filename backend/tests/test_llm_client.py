import httpx
import pytest

from app.models.llm_client import OpenAICompatLLMClient


@pytest.mark.anyio
async def test_lists_models_with_bearer_auth() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json={"data": [{"id": "test-model"}]})

    client = OpenAICompatLLMClient(
        "http://llm.test/v1",
        "test-model",
        api_key="secret",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        assert (await client.list_models())["data"][0]["id"] == "test-model"
    finally:
        await client._client.aclose()


@pytest.mark.anyio
async def test_stream_chat_extracts_content_and_done_marker() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        assert b'"stream":true' in body
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"
                b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"
                b"data: [DONE]\n\n"
            ),
        )

    client = OpenAICompatLLMClient(
        "http://llm.test/v1",
        "test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        tokens = [token async for token in client.stream_chat([{"role": "user", "content": "Hi"}])]
    finally:
        await client._client.aclose()

    assert tokens == ["Hel", "lo"]
