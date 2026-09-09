import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx


class LLMClientError(RuntimeError):
    """Raised when the OpenAI-compatible service returns an invalid response."""


class OpenAICompatLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        request_timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 3.0,
        stream_idle_timeout_seconds: float = 15.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout = httpx.Timeout(
            request_timeout_seconds,
            connect=connect_timeout_seconds,
            read=stream_idle_timeout_seconds,
        )
        self._client = http_client or httpx.AsyncClient(timeout=self.request_timeout)
        self._owns_client = http_client is None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def list_models(self) -> Mapping[str, Any]:
        try:
            response = await self._client.get(f"{self.base_url}/models", headers=self._headers)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMClientError(
                f"models request failed: {exc.__class__.__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise LLMClientError("/models response must be an object")
        return payload

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except ValueError as exc:
                        raise LLMClientError("invalid chat stream chunk") from exc
                    # 容错: 部分服务在流中插入 usage/keepalive 等无 choices 的
                    # chunk(如 omlx chunked SSE), 以及 delta 无 content 的
                    # reasoning/空片——均跳过, 不当作错误。
                    try:
                        choices = chunk.get("choices") or []
                        delta = (choices[0] or {}).get("delta") or {}
                    except (KeyError, IndexError, TypeError, AttributeError):
                        # 无 choices / 非标准结构(usage/keepalive 等)——跳过
                        continue
                    token = delta.get("content", "")
                    if not isinstance(token, str):
                        continue
                    if token:
                        yield str(token)
        except LLMClientError:
            raise
        except httpx.HTTPError as exc:
            raise LLMClientError(
                f"chat request failed: {exc.__class__.__name__}"
            ) from exc
