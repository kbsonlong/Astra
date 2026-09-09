import httpx
import pytest

from app.models.llm_client import LLMClientError, OpenAICompatLLMClient


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


@pytest.mark.anyio
async def test_stream_chat_tolerates_non_choices_chunks() -> None:
    """omlx chunked SSE 会在流中插入 usage/keepalive 等无 choices 的
    chunk, 以及 delta 无 content 的 reasoning/空片——必须跳过不报错。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        events = [
            'data: {"choices":[{"delta":{"content":"你"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":""}}]}\n\n',  # 空 content
            'data: {"usage":{"total_tokens":5}}\n\n',           # 无 choices
            'data: {"choices":[]}\n\n',                         # 空 choices
            'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',  # 仅 reasoning
            'data: {"choices":[{"delta":{"content":"好"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(events).encode("utf-8"),
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

    assert tokens == ["你", "好"]


@pytest.mark.anyio
async def test_stream_chat_wraps_connect_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("remote LLM did not respond", request=request)

    client = OpenAICompatLLMClient(
        "http://llm.test/v1",
        "test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(LLMClientError, match="ConnectTimeout"):
            _ = [
                token
                async for token in client.stream_chat(
                    [{"role": "user", "content": "Hi"}]
                )
            ]
    finally:
        await client._client.aclose()
