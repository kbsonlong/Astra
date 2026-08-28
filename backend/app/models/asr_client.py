import asyncio
import logging
import os
import tempfile
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)


class ASRClientError(RuntimeError):
    """Raised when the ASR SDK cannot return a transcription."""


class MlxAudioAsrClient:
    def __init__(
        self,
        model: str,
        language: str = "Chinese",
        max_tokens: int = 512,
        repetition_penalty: float = 1.08,
        repetition_context_size: int = 100,
        chunk_duration: float = 30.0,
        load_model: Callable[[str], Any] | None = None,
        generate_transcription: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size
        self.chunk_duration = chunk_duration
        self._load_model = load_model
        self._generate_transcription = generate_transcription
        self._model_instance: Any | None = None

    def is_ready(self) -> bool:
        return bool(self.model)

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        if not audio:
            raise ASRClientError("audio must not be empty")
        try:
            with tempfile.TemporaryDirectory() as directory:
                audio_name = os.path.basename(filename) or "speech.wav"
                if not os.path.splitext(audio_name)[1]:
                    audio_name += ".wav"
                audio_path = os.path.join(directory, audio_name)
                with open(audio_path, "wb") as handle:
                    handle.write(audio)
                output_path = os.path.join(directory, "transcript")
                if self._load_model is None or self._generate_transcription is None:
                    sdk_load_model, sdk_generate = self._load_sdk()
                else:
                    sdk_load_model, sdk_generate = None, None
                load_model = self._load_model or sdk_load_model
                generate_transcription = self._generate_transcription or sdk_generate
                assert load_model is not None
                assert generate_transcription is not None
                model = await self._get_model(load_model)
                result = await asyncio.to_thread(
                    generate_transcription,
                    model=model,
                    audio=audio_path,
                    output_path=output_path,
                    format="txt",
                    language=self.language,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    repetition_penalty=self.repetition_penalty,
                    repetition_context_size=self.repetition_context_size,
                    chunk_duration=self.chunk_duration,
                )
        except Exception as exc:
            if isinstance(exc, ASRClientError):
                raise
            logger.exception("mlx-audio transcription failed for model %s", self.model)
            raise ASRClientError(
                f"mlx-audio transcription failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            raise ASRClientError("mlx-audio response does not contain text")
        return text.strip()

    async def _get_model(self, load_model: Callable[[str], Any]) -> Any:
        if self._model_instance is None:
            self._model_instance = await asyncio.to_thread(load_model, self.model)
        return self._model_instance

    @staticmethod
    def _load_sdk() -> tuple[Callable[[str], Any], Callable[..., Any]]:
        try:
            from mlx_audio.stt.generate import generate_transcription
            from mlx_audio.stt.utils import load_model
        except ImportError as exc:
            raise ASRClientError("mlx-audio is not installed") from exc
        return load_model, generate_transcription
