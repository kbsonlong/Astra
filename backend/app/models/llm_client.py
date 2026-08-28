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
        response = await self._client.get(f"{self.base_url}/models", headers=self._headers)
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMClientError("invalid /models response") from exc
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
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise LLMClientError("chat request failed") from exc

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise LLMClientError("invalid chat stream chunk") from exc
                if token:
                    yield str(token)
