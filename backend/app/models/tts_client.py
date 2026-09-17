import asyncio
import io
import logging
import struct
import threading
import wave
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


logger = logging.getLogger(__name__)


# model id -> (创建线程 id, 模型实例)
# 与 asr_client 的约定一致: MLX GPU stream 是 thread-local 的, 模型实例只能在
# 创建线程内复用; 跨线程调用(测试 to_thread / 未来独立 worker)命中不同线程键,
# 自动重载。voice/lang_code/speed 每次 synthesize 时单独传入, 不进缓存键,
# 因此不同声音配置可安全共享同一份权重。
_MLX_TTS_MODEL_CACHE: dict[str, tuple[int, Any]] = {}


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

    def capabilities(self) -> dict[str, Any]:
        """引擎能力声明 (VoiceStudio 路线图方向二)。"""
        return {
            "engine": "piper",
            "backend": "piper",
            "languages": ["zh"],
            "clone": False,
            "streaming": False,
            "device": ["cpu"],
        }

    def is_available(self) -> tuple[bool, str]:
        """可用性探测: (可用?, 原因)。理由文本不含本地路径, 避免泄漏。"""
        if self._voice is not None:
            return True, "ready"
        if not self.model_path:
            return False, "Piper 模型路径未配置 (设置 TTS_MODEL_PATH)"
        try:
            import piper  # noqa: F401
        except ImportError:
            return False, "piper-tts 未安装 (pip install piper-tts)"
        return True, "ready"

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


class MlxAudioTtsClient:
    """基于 mlx-audio 的中文原生 TTS 后端 (VoiceStudio 路线图方向一)。

    契约与 PiperSdkTtsClient 一致: is_ready() -> bool; async synthesize(text) -> WAV bytes。

    生产路径通过 ``mlx_audio.tts.utils.load_model`` 加载模型, 调用 ``model.generate(
    text, voice=, lang_code=, speed=)`` 得到波形迭代器 (``result.audio`` 为 mx.array),
    再拼接成 16-bit PCM WAV。模型按线程缓存, 复用 asr_client 的 MLX thread-local 约定。

    测试通过注入 ``model``/``model_loader`` 绕过真实 MLX 依赖。
    """

    def __init__(
        self,
        model: str,
        *,
        voice: str = "zf_xiaobei",
        lang_code: str = "z",
        speed: float = 1.0,
        sample_rate: int = 24000,
        model_instance: Any | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self._sample_rate = sample_rate
        self._model_instance = model_instance
        self._model_loader = model_loader
        # MLX 模型的加载和推理必须固定在同一条线程，且不能阻塞 ASGI 事件循环。
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="astra-mlx-tts"
        )
        self._closed = False

    def is_ready(self) -> bool:
        return self._model_instance is not None or bool(self.model)

    def capabilities(self) -> dict[str, Any]:
        """引擎能力声明 (VoiceStudio 路线图方向二)。"""
        return {
            "engine": "mlx_audio",
            "backend": "mlx_audio",
            "model": self.model,
            "voice": self.voice,
            "lang_code": self.lang_code,
            # Kokoro 等 mlx-audio 模型支持多语种, 中文为一等公民
            "languages": ["zh", "en", "ja"],
            "clone": False,
            "streaming": False,
            "device": ["mps"],
        }

    def is_available(self) -> tuple[bool, str]:
        """可用性探测: (可用?, 原因)。"""
        if self._model_instance is not None or self._model_loader is not None:
            return True, "ready"
        if not self.model:
            return False, "mlx-audio 模型未配置 (设置 TTS_MLX_MODEL)"
        try:
            from mlx_audio.tts.utils import load_model  # noqa: F401
        except ImportError:
            return False, "mlx-audio 未安装 (pip install mlx-audio; 需 Apple Silicon)"
        return True, "ready"

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise TTSClientError("text must not be empty")
        if self._closed:
            raise TTSClientError("mlx-audio TTS client is closed")
        try:
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(
                self._executor, self._synthesize_blocking, text
            )
        except Exception as exc:
            if isinstance(exc, TTSClientError):
                raise
            logger.exception("mlx-audio synthesis failed for model %s", self.model)
            raise TTSClientError(
                f"mlx-audio synthesis failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        if not audio:
            raise TTSClientError("mlx-audio returned empty audio")
        return audio

    def _synthesize_blocking(self, text: str) -> bytes:
        model = self._model_instance or self._load_model_cached_blocking(self.model)
        samples, sample_rate = self._generate_waveform(model, text)
        return self._encode_wav(samples, sample_rate)

    def _generate_waveform(self, model: Any, text: str) -> tuple[list[float], int]:
        """调用 model.generate 并把波形迭代器拍平成 float 列表 + 采样率。"""
        results = model.generate(
            text,
            voice=self.voice,
            lang_code=self.lang_code,
            speed=self.speed,
        )
        samples: list[float] = []
        sample_rate = self._resolve_sample_rate(model)
        for result in self._iter_results(results):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            samples.extend(self._to_float_list(audio))
            rate = getattr(result, "sample_rate", None)
            if isinstance(rate, (int, float)) and rate > 0:
                sample_rate = int(rate)
        if not samples:
            raise TTSClientError("mlx-audio produced no audio samples")
        return samples, sample_rate

    @staticmethod
    def _iter_results(results: Any) -> Iterable[Any]:
        if results is None:
            return []
        if isinstance(results, (list, tuple)):
            return results
        # generate() 通常返回生成器/迭代器
        return results

    def _resolve_sample_rate(self, model: Any) -> int:
        rate = getattr(model, "sample_rate", None)
        if isinstance(rate, (int, float)) and rate > 0:
            return int(rate)
        return self._sample_rate

    @staticmethod
    def _to_float_list(audio: Any) -> list[float]:
        """把 mx.array / numpy / 序列波形转换成 python float 列表。"""
        tolist = getattr(audio, "tolist", None)
        if callable(tolist):
            flat = tolist()
        else:
            flat = list(audio)
        result: list[float] = []
        _flatten(flat, result)
        return result

    @staticmethod
    def _encode_wav(samples: list[float], sample_rate: int) -> bytes:
        """float [-1, 1] 波形 -> 16-bit 单声道 PCM WAV bytes。"""
        pcm = bytearray()
        for value in samples:
            clamped = -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
            pcm += struct.pack("<h", int(round(clamped * 32767.0)))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate) or 24000)
            handle.writeframes(bytes(pcm))
        return buffer.getvalue()

    @classmethod
    def clear_model_cache(cls) -> None:
        """测试隔离用: 清空模块级 MLX TTS 模型缓存。"""
        _MLX_TTS_MODEL_CACHE.clear()

    def _load_model_cached_blocking(self, model: str) -> Any:
        # 该方法仅在 self._executor 的单线程中调用；线程 id 与模型实例严格配对。
        tid = threading.get_ident()
        hit = _MLX_TTS_MODEL_CACHE.get(model)
        if hit is not None and hit[0] == tid:
            return hit[1]
        loader = self._model_loader or type(self)._load_model_from_sdk
        instance = loader(model)
        _MLX_TTS_MODEL_CACHE[model] = (tid, instance)
        return instance

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    @staticmethod
    def _load_model_from_sdk(model: str) -> Any:
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise TTSClientError("mlx-audio is not installed") from exc
        return load_model(model)


def _flatten(value: Any, out: list[float]) -> None:
    """递归展平嵌套序列为 float 列表 (处理多维波形)。"""
    if isinstance(value, (int, float)):
        out.append(float(value))
        return
    try:
        iterator = iter(value)
    except TypeError:
        out.append(float(value))
        return
    for item in iterator:
        _flatten(item, out)
