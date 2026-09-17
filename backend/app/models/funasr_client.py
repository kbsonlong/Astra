"""FunASR / SenseVoice ASR client with inline speaker diarization.

VoiceStudio 路线图方向三 (P2): SenseVoice 负责转写 + 标点, 可选 cam++ 说话人模型做
内联分离, 一次 generate 直接产出带时间戳与说话人的分段。相比 Qwen3-ASR + 独立
ResemblyzerDiarizationStage, 长会议场景更省算力。

契约:
- ``transcribe_segments(wav) -> (language, list[Segment])``: 内联分离主入口, 供
  workflow 的 InlineAsrDiarizeStage 调用。
- ``diarizes_inline() -> bool``: 声明该后端自带说话人标签。
- ``transcribe(audio, filename) -> str``: 兼容 ASRStage 协议 (逐段文本, 拼接分段)。
- ``capabilities()`` / ``is_available()``: 引擎抽象 (方向二)。

解析借鉴 VoiceStudio ``_normalize_funasr``: 优先 VAD ``sentence_info`` (毫秒时间戳
+ 可选 ``spk`` 说话人 id), 无则回退单段 ``text`` + ``timestamp``; 清洗 SenseVoice
富标签 ``<|...|>``。解析为纯函数, 无需安装 funasr 即可测试。
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Callable
from typing import Any

from app.core.workflow import Segment

logger = logging.getLogger(__name__)


class FunAsrClientError(RuntimeError):
    """Raised when the FunASR SDK cannot return a transcription."""


# SenseVoice emits rich tokens like ``<|en|><|NEUTRAL|><|Speech|>`` around text.
_FUNASR_TAG_RE = re.compile(r"<\|[^|>]*\|>")

# model id -> (创建线程 id, 模型实例); 与 asr_client 的 MLX thread-local 约定一致。
_FUNASR_MODEL_CACHE: dict[str, tuple[int, Any]] = {}


def _ms_to_s(value: Any) -> float | None:
    """Milliseconds → seconds (FunASR reports ms). None on bad input."""
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def _clean_funasr_text(text: Any) -> str:
    return _FUNASR_TAG_RE.sub("", str(text or "")).strip()


def _speaker_label(spk: Any) -> str:
    if isinstance(spk, bool):  # bool 是 int 的子类, 显式排除
        return str(spk)
    if isinstance(spk, (int, float)):
        return f"Speaker {int(spk) + 1}"
    return str(spk)


def normalize_funasr_segments(res: Any) -> tuple[str | None, list[Segment]]:
    """Normalise FunASR ``generate()`` output → ``(language, [Segment])``.

    Defensive about FunASR's output variations. Pure — testable without funasr.
    """
    item = (res[0] if isinstance(res, (list, tuple)) and res else res) or {}
    if not isinstance(item, dict):
        item = {"text": str(item)}
    language = item.get("language") or item.get("lang") or None

    segments: list[Segment] = []
    for entry in item.get("sentence_info") or []:
        if not isinstance(entry, dict):
            continue
        text = _clean_funasr_text(entry.get("text") or entry.get("sentence", ""))
        if not text:
            continue
        start = _ms_to_s(entry.get("start", 0)) or 0.0
        end = _ms_to_s(entry.get("end"))
        segment = Segment(
            start=start,
            end=end if end is not None else start,
            text=text,
            raw_text=text,
        )
        spk = entry.get("spk")
        if spk is not None:
            segment.speaker = _speaker_label(spk)
            segment.speaker_confidence = "inline"
        segments.append(segment)

    if not segments:
        text = _clean_funasr_text(item.get("text", ""))
        if text:
            timestamps = item.get("timestamp") or []  # [[start_ms, end_ms], ...]
            start = _ms_to_s(timestamps[0][0]) if timestamps else 0.0
            end = _ms_to_s(timestamps[-1][1]) if timestamps else None
            start = start or 0.0
            segments.append(
                Segment(
                    start=start,
                    end=end if end is not None else start,
                    text=text,
                    raw_text=text,
                )
            )

    return language, segments


class FunAsrClient:
    """FunASR (SenseVoiceSmall + FSMN-VAD, 可选 cam++ 内联说话人分离)。"""

    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        *,
        language: str = "auto",
        diarize: bool = True,
        vad_model: str = "fsmn-vad",
        spk_model: str = "cam++",
        model_instance: Any | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.diarize = diarize
        self.vad_model = vad_model
        self.spk_model = spk_model
        self._model_instance = model_instance
        self._model_loader = model_loader

    # ── 引擎抽象 (方向二) ────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self._model_instance is not None or bool(self.model)

    def diarizes_inline(self) -> bool:
        """声明该后端自带说话人标签 (cam++ 内联分离启用时)。"""
        return bool(self.diarize)

    def capabilities(self) -> dict[str, Any]:
        return {
            "engine": "funasr",
            "backend": "funasr",
            "model": self.model,
            "language": self.language,
            # SenseVoice 支持 50+ 语种, 中文为一等公民
            "languages": ["zh", "yue", "en", "ja", "ko"],
            "diarize": self.diarizes_inline(),
            "inline_diarization": self.diarizes_inline(),
            "streaming": False,
            "device": ["cuda", "cpu"],
        }

    def is_available(self) -> tuple[bool, str]:
        if self._model_instance is not None:
            return True, "ready"
        if not self.model:
            return False, "FunASR 模型未配置 (设置 ASR_FUNASR_MODEL)"
        if self._model_loader is not None:
            return True, "ready"
        try:
            from funasr import AutoModel  # noqa: F401
        except ImportError:
            return False, "funasr 未安装 (pip install funasr)"
        return True, "ready"

    # ── 内联分离主入口 ──────────────────────────────────────────────────
    async def transcribe_segments(
        self, wav: str, *, filename: str = "speech.wav"
    ) -> tuple[str, list[Segment]]:
        """整段 wav -> (language, [Segment])。segment 已带时间戳与说话人。"""
        try:
            model = self._model_instance or await self._load_model_cached(self.model)
            raw = await self._run(self._generate, model, wav)
            language, segments = normalize_funasr_segments(raw)
        except Exception as exc:
            if isinstance(exc, FunAsrClientError):
                raise
            logger.exception("FunASR transcription failed for model %s", self.model)
            raise FunAsrClientError(
                f"funasr transcription failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        return (language or "zh"), segments

    # ── ASRStage 兼容 (逐段文本拼接) ────────────────────────────────────
    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        """兼容 ASRStage 协议: 把内联分段文本拼接为整段文本。

        注意: 该路径丢弃说话人与时间戳, 仅用于非内联场景的降级兼容。内联分离
        请用 transcribe_segments 经 InlineAsrDiarizeStage。
        """
        if not audio:
            raise FunAsrClientError("audio must not be empty")
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            name = os.path.basename(filename) or "speech.wav"
            if not os.path.splitext(name)[1]:
                name += ".wav"
            path = os.path.join(directory, name)
            with open(path, "wb") as handle:
                handle.write(audio)
            _, segments = await self.transcribe_segments(path, filename=name)
        return " ".join(seg.text for seg in segments if seg.text).strip()

    def _generate(self, model: Any, wav: str) -> Any:
        """调用 FunASR model.generate。cam++ 内联分离由 model 装配时启用。"""
        return model.generate(
            input=wav,
            language=self.language,
            use_itn=True,
            batch_size_s=300,
        )

    @classmethod
    def clear_model_cache(cls) -> None:
        _FUNASR_MODEL_CACHE.clear()

    async def _load_model_cached(self, model: str) -> Any:
        tid = threading.get_ident()
        hit = _FUNASR_MODEL_CACHE.get(model)
        if hit is not None and hit[0] == tid:
            return hit[1]
        loader = self._model_loader or self._load_model_from_sdk
        instance = await self._run(loader, model)
        _FUNASR_MODEL_CACHE[model] = (tid, instance)
        return instance

    async def _run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._model_instance is not None or self._model_loader is not None:
            # 注入路径(测试)。
            return await asyncio.to_thread(func, *args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    def _load_model_from_sdk(self, model: str) -> Any:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise FunAsrClientError("funasr is not installed") from exc
        kwargs: dict[str, Any] = {"model": model, "vad_model": self.vad_model}
        if self.diarize and self.spk_model:
            kwargs["spk_model"] = self.spk_model
        return AutoModel(**kwargs)
