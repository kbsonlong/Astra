import httpx


class TTSClientError(RuntimeError):
    """Raised when Piper cannot synthesize audio."""


class PiperHttpTtsClient:
    def __init__(self, endpoint: str, http_client: httpx.AsyncClient | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise TTSClientError("text must not be empty")
        try:
            response = await self._client.post(
                f"{self.endpoint}/synthesize",
                json={"text": text},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TTSClientError("piper synthesis failed") from exc
        if not response.content:
            raise TTSClientError("piper returned empty audio")
        return response.content
