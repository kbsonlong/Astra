import asyncio
import io
from collections.abc import Callable
from typing import Any


class TTSClientError(RuntimeError):
    """Raised when the Piper SDK cannot synthesize audio."""


class PiperSdkTtsClient:
    def __init__(
        self,
        model_path: str,
        voice: Any | None = None,
        voice_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self._voice = voice
        self._voice_loader = voice_loader

    def is_ready(self) -> bool:
        return self._voice is not None or bool(self.model_path)

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise TTSClientError("text must not be empty")
        try:
            voice = self._voice or self._load_voice()
            audio = await asyncio.to_thread(self._synthesize_sync, voice, text)
        except Exception as exc:
            if isinstance(exc, TTSClientError):
                raise
            raise TTSClientError("Piper synthesis failed") from exc
        if not audio:
            raise TTSClientError("Piper returned empty audio")
        return audio

    def _load_voice(self) -> Any:
        if not self.model_path:
            raise TTSClientError("Piper model path is not configured")
        if self._voice_loader is not None:
            self._voice = self._voice_loader(self.model_path)
            return self._voice
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise TTSClientError("piper-tts is not installed") from exc
        self._voice = PiperVoice.load(self.model_path)
        return self._voice

    @staticmethod
    def _synthesize_sync(voice: Any, text: str) -> bytes:
        output = io.BytesIO()
        voice.synthesize_wav(text, output)
        return output.getvalue()
