import asyncio
import inspect
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)

# model_path -> (创建线程 id, 模型实例)
# MLX GPU stream 是 thread-local 的, 模型实例只能在创建线程内复用;
# 跨线程调用(测试 to_thread / 未来的独立 worker)会命中不同线程键, 自动重载。
# 键不含 client 配置——hotwords/system_prompt 每次 generate 时单独传入,
# 因此 voice 与 meeting 两个 client 可安全共享同一份权重, 不重复占内存。
_MLX_MODEL_CACHE: dict[str, tuple[int, Any]] = {}


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
                    # SDK 生产路径: 模型跨调用缓存(同线程复用), 避免每次
                    # transcribe 从磁盘重载权重(会议 131 段 x ~0.6-2s)。
                    model = await self._load_model_cached(self.model)
                    load_audio = MlxAudioAsrClient._load_audio_from_sdk
                    generate_transcription = None
                else:
                    # 注入路径(测试): 保持原行为, 不走缓存。
                    model = await self._run_mlx(self._load_model, self.model)
                    load_audio = self._load_audio
                    generate_transcription = self._generate_transcription
                assert model is not None
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
        return self._strip_prompt_leak(text.strip())

    def _strip_prompt_leak(self, text: str) -> str:
        """切除 Qwen3-ASR 在音频结束后复读的 prompt 泄漏。"""
        if not text or len(text) < 8:
            return text
        fragments = set()
        if self.system_prompt:
            fragments.add(self.system_prompt.strip())
        if self.hotwords:
            joined = ", ".join(h for h in self.hotwords if h and h.strip())
            if len(joined) >= 4:
                fragments.add(joined.strip())
        if not fragments:
            return text
        cleaned = text
        removed = False
        for frag in sorted(fragments, key=len, reverse=True):
            if frag in cleaned:
                cleaned = cleaned.replace(frag, "").strip()
                removed = True
        if removed:
            cleaned = re.sub(r"[，,。.、；;\s]+$", "", cleaned)
        return cleaned

    @classmethod
    def clear_model_cache(cls) -> None:
        """测试隔离用: 清空模块级 MLX 模型缓存。"""
        _MLX_MODEL_CACHE.clear()

    async def _load_model_cached(self, model: str) -> Any:
        """SDK 路径的模型加载(带缓存)。"""
        tid = threading.get_ident()
        hit = _MLX_MODEL_CACHE.get(model)
        if hit is not None and hit[0] == tid:
            return hit[1]
        instance = await self._run_mlx(type(self)._load_model_from_sdk, model)
        _MLX_MODEL_CACHE[model] = (tid, instance)
        return instance

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
        if (
            self._load_model is not None
            and self._generate_transcription is not None
            and self._load_audio is not None
        ):
            return await asyncio.to_thread(func, *args, **kwargs)
        # MLX GPU streams are thread-local. Keep all production MLX work on
        # Uvicorn's main thread instead of moving it through an executor.
        return func(*args, **kwargs)

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
