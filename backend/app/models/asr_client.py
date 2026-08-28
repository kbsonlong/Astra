import asyncio
import os
import tempfile
from collections.abc import Callable
from typing import Any


class ASRClientError(RuntimeError):
    """Raised when the Whisper SDK cannot return a transcription."""


class MlxWhisperAsrClient:
    def __init__(
        self,
        model: str,
        language: str = "zh",
        transcribe: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self._transcribe = transcribe

    def is_ready(self) -> bool:
        return bool(self.model)

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        if not audio:
            raise ASRClientError("audio must not be empty")
        try:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1] or ".wav") as handle:
                handle.write(audio)
                handle.flush()
                transcribe = self._transcribe or self._load_sdk()
                result = await asyncio.to_thread(
                    transcribe,
                    handle.name,
                    path_or_hf_repo=self.model,
                    language=self.language,
                )
        except Exception as exc:
            if isinstance(exc, ASRClientError):
                raise
            raise ASRClientError("mlx-whisper transcription failed") from exc
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str):
            raise ASRClientError("mlx-whisper response does not contain text")
        return text.strip()

    @staticmethod
    def _load_sdk() -> Callable[..., dict[str, Any]]:
        try:
            import mlx_whisper
        except ImportError as exc:
            raise ASRClientError("mlx-whisper is not installed") from exc
        return mlx_whisper.transcribe
