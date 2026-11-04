"""OpenAI-compatible image generation client used for meeting brief artwork."""
from __future__ import annotations

import base64
from typing import Any

import httpx


class ImageClientError(RuntimeError):
    """Raised when the configured image generation service returns an invalid response."""


class OpenAICompatImageClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        image_path: str = "/images/generations",
        image_size: str = "1024x1024",
        timeout_seconds: float = 180.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.image_path = "/" + image_path.strip("/")
        self.image_size = image_size
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = http_client is None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def generate(self, prompt: str) -> bytes:
        try:
            response = await self._client.post(
                f"{self.base_url}{self.image_path}",
                headers=self._headers,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "size": self.image_size,
                    "response_format": "b64_json",
                },
            )
            response.raise_for_status()
            payload: Any = response.json()
            encoded = payload.get("data", [{}])[0].get("b64_json")
            if not isinstance(encoded, str) or not encoded:
                raise ImageClientError("image response did not contain b64_json")
            return base64.b64decode(encoded, validate=True)
        except ImageClientError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ImageClientError(f"image generation failed: {exc.__class__.__name__}") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
