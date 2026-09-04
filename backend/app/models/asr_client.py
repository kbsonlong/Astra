import asyncio
import glob
import inspect
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
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
        """切除 Qwen3-ASR 在音频结束后复读的 prompt 泄漏。

        mlx-audio 0.5 的 qwen3_asr 把 system_prompt/hotwords 拼进输入
        prompt, 长段/无语音尾时模型会把这些文本当"待续写内容"原样
        抄进输出(实测 30s 段句尾复读整段 prompt)。泄漏特征: 输出中
        出现与注入 prompt 完全相同的长串, 通常位于尾部。

        策略: 把 system_prompt + 热词拼接串作为已知泄漏源, 从输出中
        移除完全匹配(长度>=4)的片段。长串整段匹配才切, 不误伤正文
        里真实说出的短词(如单独出现的 host)。
        """
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
        # 泄漏被切后原句尾常残留标点("扔。host, ..." -> "扔。, ..."),
        # 只在这种情况清理孤立标点; 未发生切除时不动正文。
        if removed:
            cleaned = re.sub(r"[，,。.、；;\s]+$", "", cleaned)
        return cleaned

    @classmethod
    def clear_model_cache(cls) -> None:
        """测试隔离用: 清空模块级 MLX 模型缓存。"""
        _MLX_MODEL_CACHE.clear()

    async def _load_model_cached(self, model: str) -> Any:
        """SDK 路径的模型加载(带缓存)。

        MLX GPU stream 是 thread-local 的: 缓存键包含创建线程 id,
        同线程内跨调用复用实例; 若在另一个线程调用(测试 to_thread、
        未来会议 worker 进程化), 键不命中会自动重载一份独立实例,
        不会把属于别的线程 stream 的模型拿来用。
        """
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


class SherpaSenseVoiceAsrClient:
    def __init__(
        self,
        model_dir: str,
        language: str = "",
        num_threads: int = 2,
        provider: str = "cpu",
        auto_language: bool = True,
        use_itn: bool = True,
        chunk_duration: float = 30.0,
        long_audio_threshold: float = 60.0,
        hotwords: tuple[str, ...] = (),
        *,
        _create_recognizer: Callable[..., Any] | None = None,
        _read_wav: Callable[[str], tuple[int, Any]] | None = None,
        _resample: Callable[[Any, int, int], Any] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.language = "" if auto_language else language
        self.num_threads = num_threads
        self.provider = provider
        self.auto_language = auto_language
        self.use_itn = use_itn
        self.chunk_duration = chunk_duration
        self.long_audio_threshold = long_audio_threshold
        self.hotwords = hotwords
        self._create_recognizer = _create_recognizer
        self._read_wav = _read_wav
        self._resample = _resample
        self._recognizer: Any | None = None
        self._model_path: str | None = None
        self._tokens_path: str | None = None

    def is_ready(self) -> bool:
        if self._recognizer is not None:
            return True
        try:
            self._ensure_model_files()
        except ASRClientError:
            return False
        return bool(self._model_path and self._tokens_path and os.path.isdir(self.model_dir))

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        if not audio:
            raise ASRClientError("audio must not be empty")
        try:
            recognizer = await self._get_or_create_recognizer()
            with tempfile.TemporaryDirectory() as directory:
                audio_name = os.path.basename(filename) or "speech.wav"
                if not os.path.splitext(audio_name)[1]:
                    audio_name += ".wav"
                audio_path = os.path.join(directory, audio_name)
                with open(audio_path, "wb") as handle:
                    handle.write(audio)
                read_wav = self._read_wav or self._default_read_wav
                sample_rate, samples = await asyncio.to_thread(read_wav, audio_path)
                target_rate = 16000
                if sample_rate != target_rate:
                    resample = self._resample or self._default_resample
                    samples = await asyncio.to_thread(resample, samples, sample_rate, target_rate)
                    sample_rate = target_rate
                duration = len(samples) / sample_rate
                if duration <= self.long_audio_threshold:
                    text = await self._decode_chunk(recognizer, samples, sample_rate)
                else:
                    texts = []
                    samples_per_chunk = max(1, int(self.chunk_duration * sample_rate))
                    for offset in range(0, len(samples), samples_per_chunk):
                        chunk = samples[offset : offset + samples_per_chunk]
                        chunk_text = await self._decode_chunk(recognizer, chunk, sample_rate)
                        if chunk_text:
                            texts.append(chunk_text)
                    text = " ".join(texts)
        except Exception as exc:
            if isinstance(exc, ASRClientError):
                raise
            logger.exception("sherpa-onnx SenseVoice transcription failed")
            raise ASRClientError(
                f"sherpa-onnx SenseVoice transcription failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        if not isinstance(text, str):
            raise ASRClientError("sherpa-onnx SenseVoice response does not contain text")
        return text.strip()

    async def _decode_chunk(self, recognizer: Any, samples: Any, sample_rate: int) -> str:
        return await asyncio.to_thread(self._decode_blocking, recognizer, samples, sample_rate)

    @staticmethod
    def _decode_blocking(recognizer: Any, samples: Any, sample_rate: int) -> str:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ASRClientError("sherpa-onnx is not installed") from exc
        stream = recognizer.create_stream()
        try:
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            result = stream.result
            text = ""
            if hasattr(result, "text"):
                text = getattr(result, "text")
                if isinstance(text, str):
                    return text.strip()
            if isinstance(result, dict):
                text = result.get("text", "") or ""
                if isinstance(text, str):
                    return text.strip()
            try:
                import json as _json

                parsed = _json.loads(str(result))
                if isinstance(parsed, dict):
                    text = parsed.get("text", "") or ""
                    if isinstance(text, str):
                        return text.strip()
            except Exception:
                pass
            return str(result).strip()
        finally:
            del stream

    async def _get_or_create_recognizer(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        self._ensure_model_files()
        factory = self._create_recognizer or self._default_create_recognizer
        recognizer = await asyncio.to_thread(
            factory,
            self._model_path,
            self._tokens_path,
            self.language,
            self.use_itn,
            self.num_threads,
            self.provider,
        )
        self._recognizer = recognizer
        return recognizer

    def _ensure_model_files(self) -> None:
        if self._model_path and self._tokens_path:
            return
        if not os.path.isdir(self.model_dir):
            raise ASRClientError(
                f"sherpa model dir not found: {self.model_dir}. "
                f"Download from HuggingFace csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 "
                f"and place model.int8.onnx + tokens.txt into {self.model_dir}"
            )
        model_candidates = glob.glob(os.path.join(self.model_dir, "*model*.onnx"))
        if not model_candidates:
            raise ASRClientError(
                f"No *model*.onnx found in {self.model_dir}, expected model.int8.onnx (SenseVoice int8 quant)"
            )
        tokens_candidates = glob.glob(os.path.join(self.model_dir, "*tokens*.txt"))
        if not tokens_candidates:
            raise ASRClientError(f"No *tokens*.txt found in {self.model_dir}")
        self._model_path = str(Path(model_candidates[0]).resolve())
        self._tokens_path = str(Path(tokens_candidates[0]).resolve())

    @staticmethod
    def _default_create_recognizer(
        model_path: str,
        tokens_path: str,
        language: str,
        use_itn: bool,
        num_threads: int,
        provider: str,
    ) -> Any:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ASRClientError("sherpa-onnx is not installed") from exc
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
            use_itn=use_itn,
            debug=False,
        )
        if language:
            setattr(recognizer, "_sense_voice_language", language)
        return recognizer

    @staticmethod
    def _default_read_wav(path: str) -> tuple[int, Any]:
        try:
            import wave
        except ImportError as exc:
            raise ASRClientError("wave module is unavailable") from exc
        import numpy as np

        with wave.open(path, "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            raw = wav.readframes(n_frames)
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        elif sample_width == 1:
            dtype = np.uint8
        else:
            raise ASRClientError(f"unsupported wav sample width: {sample_width}")
        pcm = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)
        if np.issubdtype(pcm.dtype, np.integer):
            max_val = float(np.iinfo(pcm.dtype).max) + 1.0
            pcm = pcm.astype(np.float32) / max_val
            pcm = np.clip(pcm, -1.0, 1.0, out=pcm)
        else:
            pcm = pcm.astype(np.float32)
        return sample_rate, pcm

    @staticmethod
    def _default_resample(samples: Any, src_rate: int, dst_rate: int) -> Any:
        if src_rate == dst_rate:
            return samples
        import numpy as np

        ratio = dst_rate / src_rate
        src_len = len(samples)
        dst_len = max(1, int(round(src_len * ratio)))
        src_idx = np.arange(dst_len, dtype=np.float32) / ratio
        idx0 = np.floor(src_idx).astype(np.int64)
        idx1 = np.minimum(idx0 + 1, src_len - 1)
        frac = (src_idx - idx0).astype(np.float32)
        return (samples[idx0] * (1.0 - frac) + samples[idx1] * frac).astype(np.float32)


class SherpaZipformerBilingualAsrClient:
    def __init__(
        self,
        model_dir: str,
        language: str = "",
        num_threads: int = 2,
        provider: str = "cpu",
        decoding_method: str = "greedy_search",
        chunk_duration: float = 30.0,
        long_audio_threshold: float = 60.0,
        hotwords: tuple[str, ...] = (),
        model_type: str = "zipformer",
        modeling_unit: str = "bpe",
        sample_rate: int = 16000,
        feature_dim: int = 80,
        *,
        _create_recognizer: Callable[..., Any] | None = None,
        _read_wav: Callable[[str], tuple[int, Any]] | None = None,
        _resample: Callable[[Any, int, int], Any] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.language = language
        self.num_threads = num_threads
        self.provider = provider
        self.decoding_method = decoding_method
        self.chunk_duration = chunk_duration
        self.long_audio_threshold = long_audio_threshold
        self.hotwords = hotwords
        self.model_type = model_type
        self.modeling_unit = modeling_unit
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self._create_recognizer = _create_recognizer
        self._read_wav = _read_wav
        self._resample = _resample
        self._recognizer: Any = None
        self._tokens_path: str | None = None
        self._bpe_vocab_path: str | None = None
        self._encoder_path: str | None = None
        self._decoder_path: str | None = None
        self._joiner_path: str | None = None

    def is_ready(self) -> bool:
        if self._recognizer is not None:
            return True
        try:
            self._ensure_model_files()
        except ASRClientError:
            return False
        has_tok_or_bpe = bool(self._tokens_path or self._bpe_vocab_path)
        return has_tok_or_bpe and bool(self._encoder_path and self._decoder_path and self._joiner_path)

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        if not audio:
            raise ASRClientError("audio must not be empty")
        try:
            recognizer = await self._get_or_create_recognizer()
            with tempfile.TemporaryDirectory() as directory:
                audio_name = os.path.basename(filename) or "speech.wav"
                if not os.path.splitext(audio_name)[1]:
                    audio_name += ".wav"
                audio_path = os.path.join(directory, audio_name)
                with open(audio_path, "wb") as handle:
                    handle.write(audio)
                read_wav = self._read_wav or SherpaSenseVoiceAsrClient._default_read_wav
                sample_rate, samples = await asyncio.to_thread(read_wav, audio_path)
                target_rate = 16000
                if sample_rate != target_rate:
                    resample = self._resample or SherpaSenseVoiceAsrClient._default_resample
                    samples = await asyncio.to_thread(resample, samples, sample_rate, target_rate)
                    sample_rate = target_rate
                duration = len(samples) / sample_rate
                if duration <= self.long_audio_threshold:
                    text = await self._decode_chunk(recognizer, samples, sample_rate)
                else:
                    texts = []
                    samples_per_chunk = max(1, int(self.chunk_duration * sample_rate))
                    for offset in range(0, len(samples), samples_per_chunk):
                        chunk = samples[offset : offset + samples_per_chunk]
                        chunk_text = await self._decode_chunk(recognizer, chunk, sample_rate)
                        if chunk_text:
                            texts.append(chunk_text)
                    text = "".join(texts)
        except Exception as exc:
            if isinstance(exc, ASRClientError):
                raise
            logger.exception("sherpa-onnx Zipformer Bilingual transcription failed")
            raise ASRClientError(
                f"sherpa-onnx Zipformer Bilingual transcription failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        if not isinstance(text, str):
            raise ASRClientError("sherpa-onnx Zipformer Bilingual response does not contain text")
        return text.strip()

    async def _decode_chunk(self, recognizer: Any, samples: Any, sample_rate: int) -> str:
        return await asyncio.to_thread(self._decode_blocking, recognizer, samples, sample_rate)

    @staticmethod
    def _decode_blocking(recognizer: Any, samples: Any, sample_rate: int) -> str:
        try:
            import sherpa_onnx
            import numpy as np
        except ImportError as exc:
            raise ASRClientError("sherpa-onnx/numpy is not installed") from exc
        stream = recognizer.create_stream()
        try:
            stream.accept_waveform(sample_rate, samples)
            tail = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
            stream.accept_waveform(sample_rate, tail)
            stream.input_finished()
            # Streaming non-autoregressive decode loop (same as sherpa streaming demo)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result_text = recognizer.get_result(stream)
            if isinstance(result_text, str):
                return result_text.strip()
            # some versions wrap in OnlineRecognizerResult obj with .text attr
            if hasattr(result_text, "text"):
                t = getattr(result_text, "text")
                if isinstance(t, str):
                    return t.strip()
            return str(result_text).strip()
        finally:
            del stream

    async def _get_or_create_recognizer(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        self._ensure_model_files()
        factory = self._create_recognizer or self._default_create_recognizer
        recognizer = await asyncio.to_thread(
            factory,
            self._tokens_path,
            self._bpe_vocab_path,
            self._encoder_path,
            self._decoder_path,
            self._joiner_path,
            self.num_threads,
            self.provider,
            self.decoding_method,
            self.model_type,
            self.modeling_unit,
            self.sample_rate,
            self.feature_dim,
        )
        self._recognizer = recognizer
        return recognizer

    def _ensure_model_files(self) -> None:
        has_any_vocab = self._tokens_path or self._bpe_vocab_path
        if (
            has_any_vocab
            and self._encoder_path
            and self._decoder_path
            and self._joiner_path
        ):
            return
        if not os.path.isdir(self.model_dir):
            raise ASRClientError(
                f"Zipformer Bilingual model dir not found: {self.model_dir}. "
                f"Run `_step1d_download_zipformer.py` to download "
                f"csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
            )

        def find(patterns: list[str], desc: str) -> str:
            for pat in patterns:
                candidates = glob.glob(os.path.join(self.model_dir, pat))
                if candidates:
                    return str(Path(candidates[0]).resolve())
            raise ASRClientError(
                f"No {desc} file (tried patterns={patterns}) found in {self.model_dir}"
            )

        self._tokens_path = None
        self._bpe_vocab_path = None
        # Always try tokens.txt first (char-level / sentencepiece token table).
        tok_candidates = glob.glob(os.path.join(self.model_dir, "tokens.txt"))
        if tok_candidates:
            self._tokens_path = str(Path(tok_candidates[0]).resolve())
        # Independently detect bpe.model / bpe.vocab (for BPE tokenization config)
        bpe_candidates = glob.glob(os.path.join(self.model_dir, "bpe.model"))
        if bpe_candidates:
            self._bpe_vocab_path = str(Path(bpe_candidates[0]).resolve())
        else:
            bpe_v_candidates = glob.glob(os.path.join(self.model_dir, "bpe.vocab"))
            if bpe_v_candidates:
                self._bpe_vocab_path = str(Path(bpe_v_candidates[0]).resolve())
        # Fallback: if no tokens.txt but bpe.model exists, use bpe.model as tokens_path
        # (so Python wrapper asserts tokens file existence).
        if self._tokens_path is None and self._bpe_vocab_path is not None:
            self._tokens_path = self._bpe_vocab_path
        if self._tokens_path is None:
            raise ASRClientError(
                f"No tokens.txt nor bpe.model/bpe.vocab found in {self.model_dir}"
            )
        self._encoder_path = find(
            ["*encoder*int8*.onnx", "*encoder*.onnx"], "encoder onnx"
        )
        self._decoder_path = find(
            ["*decoder*int8*.onnx", "*decoder*.onnx"], "decoder onnx"
        )
        self._joiner_path = find(
            ["*joiner*int8*.onnx", "*joiner*.onnx"], "joiner onnx"
        )

    @staticmethod
    def _default_create_recognizer(
        tokens_path: str | None,
        bpe_vocab_path: str | None,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        num_threads: int,
        provider: str,
        decoding_method: str,
        model_type: str,
        modeling_unit: str,
        sample_rate: int,
        feature_dim: int,
    ) -> Any:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ASRClientError("sherpa-onnx is not installed") from exc
        # Zipformer Bilingual zh+en (2023-02-20) uses BPE vocab + zipformer2 config.
        kwargs = dict(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=num_threads,
            decoding_method=decoding_method,
            provider=provider,
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            model_type=model_type,
            modeling_unit=modeling_unit,
            debug=False,
            enable_endpoint_detection=False,
        )
        if bpe_vocab_path:
            kwargs["bpe_vocab"] = bpe_vocab_path
            # Python sherpa-onnx 1.13.7 wrapper asserts tokens file exists (not empty).
            # Pass bpe.model itself as tokens so the file-existence check passes;
            # the C++ core prefers bpe_vocab when both are provided.
            if not tokens_path:
                tokens_path = bpe_vocab_path
        if tokens_path:
            kwargs["tokens"] = tokens_path
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        return recognizer
