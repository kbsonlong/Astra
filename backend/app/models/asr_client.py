from typing import Any

import httpx


class ASRClientError(RuntimeError):
    """Raised when whisper.cpp cannot return a transcription."""


class WhisperCppAsrClient:
    def __init__(self, endpoint: str, http_client: httpx.AsyncClient | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        try:
            response = await self._client.post(
                f"{self.endpoint}/inference",
                files={"file": (filename, audio, "audio/wav")},
                data={"response_format": "json"},
            )
            response.raise_for_status()
            payload: Any = response.json()
            text = payload.get("text") if isinstance(payload, dict) else None
        except (httpx.HTTPError, ValueError) as exc:
            raise ASRClientError("whisper transcription failed") from exc
        if not isinstance(text, str):
            raise ASRClientError("whisper response does not contain text")
        return text.strip()
