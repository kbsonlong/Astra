import asyncio
import inspect
import logging
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
        long_audio_threshold: float = 60.0,
        hotwords: tuple[str, ...] = (),
        system_prompt: str = "",
        load_audio: Callable[[str], Any] | None = None,
        load_model: Callable[[str], Any] | None = None,
        generate_transcription: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size
        self.chunk_duration = chunk_duration
        self.long_audio_threshold = long_audio_threshold
        self.hotwords = hotwords
        self.system_prompt = system_prompt
        self._load_audio = load_audio
        self._load_model = load_model
        self._generate_transcription = generate_transcription
        self._model_instance: Any | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="astra-mlx-asr"
        )
        self._stream = None

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
                if (
                    self._load_model is None
                    or self._generate_transcription is None
                    or self._load_audio is None
                ):
                    sdk_load_model, sdk_generate, sdk_load_audio = self._load_sdk()
                else:
                    sdk_load_model, sdk_generate, sdk_load_audio = None, None, None
                load_model = self._load_model or sdk_load_model
                generate_transcription = self._generate_transcription or sdk_generate
                load_audio = self._load_audio or sdk_load_audio
                assert load_model is not None
                model = await self._run_mlx(load_model, self.model)
                if load_audio is None:
                    raise ASRClientError("mlx-audio audio loader is unavailable")
                audio_signal = await self._run_mlx(load_audio, audio_path)
                duration = len(audio_signal) / 16000
                if duration <= self.long_audio_threshold:
                    result = await self._generate(
                        model, generate_transcription, audio_path, output_path
                    )
                    text = getattr(result, "text", None)
                else:
                    texts = []
                    samples_per_chunk = max(1, int(self.chunk_duration * 16000))
                    for offset in range(0, len(audio_signal), samples_per_chunk):
                        chunk = audio_signal[offset : offset + samples_per_chunk]
                        result = await self._generate(
                            model,
                            generate_transcription,
                            chunk,
                            os.path.join(directory, f"transcript-{offset}"),
                        )
                        chunk_text = getattr(result, "text", None)
                        if isinstance(chunk_text, str) and chunk_text.strip():
                            texts.append(chunk_text.strip())
                    text = " ".join(texts)
        except Exception as exc:
            if isinstance(exc, ASRClientError):
                raise
            logger.exception("mlx-audio transcription failed for model %s", self.model)
            raise ASRClientError(
                f"mlx-audio transcription failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        if not isinstance(text, str):
            raise ASRClientError("mlx-audio response does not contain text")
        return text.strip()

    async def _generate(
        self,
        model: Any,
        generate_transcription: Callable[..., Any] | None,
        audio: Any,
        output_path: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "audio": audio,
            "output_path": output_path,
            "format": "txt",
            "language": self.language,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "repetition_penalty": self.repetition_penalty,
            "repetition_context_size": self.repetition_context_size,
            "chunk_duration": self.chunk_duration,
            "hotwords": list(self.hotwords),
            "system_prompt": self.system_prompt or None,
        }
        if generate_transcription is not None:
            return await self._run_mlx(generate_transcription, **kwargs)

        model_generate = model.generate
        parameters = inspect.signature(model_generate).parameters
        model_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"model", "audio"} and key in parameters
        }
        return await self._run_mlx(model_generate, audio, **model_kwargs)

    async def _run_mlx(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        if (
            self._load_model is not None
            and self._generate_transcription is not None
            and self._load_audio is not None
        ):
            return await asyncio.to_thread(func, *args, **kwargs)
        return await loop.run_in_executor(
            self._executor, self._run_mlx_on_worker, func, args, kwargs
        )

    def _run_mlx_on_worker(
        self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        import mlx.core as mx

        if self._stream is None:
            self._stream = mx.new_stream(mx.gpu)
            mx.set_default_stream(self._stream)
        with mx.stream(self._stream):
            return func(*args, **kwargs)

    @staticmethod
    def _load_sdk() -> tuple[Callable[[str], Any], None, Callable[[str], Any]]:
        # Keep imports inside the MLX worker. MLX streams are thread-local.
        return MlxAudioAsrClient._load_model_from_sdk, None, MlxAudioAsrClient._load_audio_from_sdk

    @staticmethod
    def _load_model_from_sdk(model: str) -> Any:
        try:
            from mlx_audio.stt.utils import load_model
        except ImportError as exc:
            raise ASRClientError("mlx-audio is not installed") from exc
        return load_model(model)

    @staticmethod
    def _load_audio_from_sdk(path: str) -> Any:
        try:
            from mlx_audio.stt.utils import load_audio
        except ImportError as exc:
            raise ASRClientError("mlx-audio is not installed") from exc
        return load_audio(path)
