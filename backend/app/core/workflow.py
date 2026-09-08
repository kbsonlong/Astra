"""可组合的音频处理 Workflow。

默认数据流为 VAD -> ASR -> 标点恢复 -> 说话人分离；纠错阶段按配置追加。
模型实现通过小型协议注入，编排层不依赖具体的 FunASR、sherpa 或声纹库，因而既能在
Mac mini 上运行真实模型，也能用轻量 fake 做契约测试。
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .correction import (
    DEFAULT_CORRECTION_RULES,
    apply_text_rules,
    candidate_outputs,
)
from .speaker_registry import SpeakerMatch, SpeakerProfileStore

logger = logging.getLogger(__name__)

_REPEATED_PUNCTUATION = re.compile(r"([，。！？；：、,.!?;:])\1+")
_WEAK_BEFORE_TERMINAL = re.compile(r"([，、；：,;:])([。！？!?])")
_TERMINAL_BEFORE_WEAK = re.compile(r"([。！？!?])([，、；：,;:])")


def clean_repeated_punctuation(text: str) -> str:
    """清理标点模型常见的重复和冲突标点，不改动正文或换行。"""
    cleaned = _REPEATED_PUNCTUATION.sub(r"\1", text)
    # 句末标点优先，避免出现“，。”、“？！，”这类模型输出。
    cleaned = _WEAK_BEFORE_TERMINAL.sub(r"\2", cleaned)
    cleaned = _TERMINAL_BEFORE_WEAK.sub(r"\1", cleaned)
    return cleaned


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""
    speaker_id: str | None = None
    speaker_name: str = ""
    speaker_similarity: float | None = None
    speaker_confidence: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SpeechChunk:
    start: float
    end: float
    audio: bytes


@dataclass(frozen=True)
class WorkflowResult:
    language: str
    segments: list[Segment]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """阶段之间传递的可扩展上下文。"""

    wav: str | Path
    filename: str
    language: str
    chunks: list[SpeechChunk] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class WorkflowStage(Protocol):
    name: str

    async def run(self, context: WorkflowContext) -> None: ...


class VADStage(Protocol):
    async def detect(self, wav: str | Path) -> Sequence[SpeechChunk]: ...


class ASRStage(Protocol):
    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str: ...


class PunctuationStage(Protocol):
    async def restore(self, text: str, *, language: str = "zh") -> str: ...


class SpeakerDiarizationStage(Protocol):
    async def assign(self, wav: str | Path, segments: list[Segment]) -> None: ...


class PassthroughPunctuation:
    """标点模型未配置或不可用时的稳定降级实现。"""

    async def restore(self, text: str, *, language: str = "zh") -> str:
        return text

    def is_ready(self) -> bool:
        return True


class VadWorkflowStage:
    name = "vad"

    def __init__(self, vad: VADStage) -> None:
        self.vad = vad
        self.engine_name = vad.__class__.__name__

    async def run(self, context: WorkflowContext) -> None:
        context.chunks = list(await self.vad.detect(context.wav))

    def is_ready(self) -> bool:
        ready = getattr(self.vad, "is_ready", None)
        return bool(ready()) if callable(ready) else True


class AsrWorkflowStage:
    name = "asr"

    def __init__(self, asr: ASRStage) -> None:
        self.asr = asr
        self.engine_name = asr.__class__.__name__

    async def run(self, context: WorkflowContext) -> None:
        context.segments = []
        for index, chunk in enumerate(context.chunks):
            raw_text = await self.asr.transcribe(
                chunk.audio,
                filename=f"{Path(context.filename).stem}-{index}.wav",
            )
            text = (raw_text or "").strip()
            if text:
                context.segments.append(Segment(chunk.start, chunk.end, text))

    def is_ready(self) -> bool:
        ready = getattr(self.asr, "is_ready", None)
        return bool(ready()) if callable(ready) else True


class PunctuationWorkflowStage:
    name = "punctuation"

    def __init__(self, punctuation: PunctuationStage) -> None:
        self.punctuation = punctuation
        self.engine_name = punctuation.__class__.__name__

    async def run(self, context: WorkflowContext) -> None:
        for segment in context.segments:
            try:
                text = await self.punctuation.restore(
                    segment.text,
                    language=context.language,
                )
            except Exception as exc:
                logger.warning(
                    "punctuation restoration failed, keep ASR text: %s", exc
                )
                text = segment.text
            segment.text = clean_repeated_punctuation((text or "").strip())
        context.segments[:] = [
            segment for segment in context.segments if segment.text
        ]

    def is_ready(self) -> bool:
        ready = getattr(self.punctuation, "is_ready", None)
        return bool(ready()) if callable(ready) else True


class CorrectionStage:
    """唯一文本纠错阶段：规则优先，LLM 仅确认显式候选词。"""

    name = "correction"

    def __init__(
        self,
        llm: Any = None,
        *,
        rules_enabled: bool = True,
        rules: dict[str, str] | None = None,
        llm_enabled: bool = False,
        candidate_rules: dict[str, str] | None = None,
        system_prompt: str = "",
        max_tokens: int = 256,
    ) -> None:
        self.llm = llm
        self.rules_enabled = rules_enabled
        self.rules = dict(rules or DEFAULT_CORRECTION_RULES)
        self.llm_enabled = llm_enabled
        self.candidate_rules = dict(candidate_rules or {})
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.engine_name = "rules+llm" if llm_enabled else "rules"

    async def run(self, context: WorkflowContext) -> None:
        for segment in context.segments:
            text = (
                apply_text_rules(segment.text, self.rules)
                if self.rules_enabled
                else segment.text
            )
            if self.llm_enabled and self.llm is not None:
                text = await self._llm_candidate_clean(text, context.language)
            segment.text = text

    async def _llm_candidate_clean(self, text: str, language: str) -> str:
        candidates = {
            source: target
            for source, target in self.candidate_rules.items()
            if source in text
        }
        if not candidates or not getattr(self.llm, "model", ""):
            return text
        prompt = (
            f"{self.system_prompt}\n"
            "本次只允许确认以下候选替换："
            + "；".join(f"{source}->{target}" for source, target in candidates.items())
            + "。除候选词外必须逐字保留原文，不得增删、改写、调序，"
            "不得修改数字、英文、时间、人名和标点。无法确认时原样输出。"
        )
        tokens: list[str] = []
        try:
            async for token in self.llm.stream_chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
                chat_template_kwargs={"enable_thinking": False},
            ):
                tokens.append(token)
        except Exception as exc:
            logger.warning("LLM candidate correction failed, keep text: %s", exc)
            return text
        corrected = "".join(tokens).strip()
        if corrected in candidate_outputs(text, candidates):
            return corrected
        logger.warning("LLM candidate correction changed non-candidate text, reject")
        return text

    def is_ready(self) -> bool:
        if not self.llm_enabled:
            return True
        return bool(self.llm is not None and getattr(self.llm, "model", ""))


class DiarizationWorkflowStage:
    name = "sd"

    def __init__(self, diarization: SpeakerDiarizationStage) -> None:
        self.diarization = diarization
        self.engine_name = diarization.__class__.__name__

    async def run(self, context: WorkflowContext) -> None:
        await self.diarization.assign(context.wav, context.segments)

    def is_ready(self) -> bool:
        ready = getattr(self.diarization, "is_ready", None)
        return bool(ready()) if callable(ready) else True


class WorkflowEngine:
    """按注册顺序执行阶段；阶段本身不依赖具体模型。"""

    def __init__(self, stages: Sequence[WorkflowStage], *, language: str = "zh") -> None:
        self.stages = tuple(stages)
        self.language = language

    async def run(
        self, wav: str | Path, *, filename: str = "speech.wav"
    ) -> WorkflowResult:
        context = WorkflowContext(
            wav=wav,
            filename=filename,
            language=self.language,
        )
        for stage in self.stages:
            await stage.run(context)
        return WorkflowResult(
            language=context.language,
            segments=context.segments,
            metadata=context.metadata,
        )

    def stage_status(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for stage in self.stages:
            ready = getattr(stage, "is_ready", None)
            result[stage.name] = {
                "enabled": True,
                "ok": bool(ready()) if callable(ready) else True,
                "engine": getattr(stage, "engine_name", stage.__class__.__name__),
            }
        return result


class WorkflowBuilder:
    """构建可复用的有序阶段列表。"""

    def __init__(self, *, language: str = "zh") -> None:
        self.language = language
        self._stages: list[WorkflowStage] = []

    def use(self, stage: WorkflowStage) -> "WorkflowBuilder":
        self._stages.append(stage)
        return self

    def build(self) -> WorkflowEngine:
        if not self._stages:
            raise ValueError("workflow must contain at least one stage")
        return WorkflowEngine(self._stages, language=self.language)


class SileroVADStage:
    def __init__(self, model_path: str, *, sample_rate: int = 16000) -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate

    async def detect(self, wav: str | Path) -> Sequence[SpeechChunk]:
        return await asyncio.to_thread(self._detect_blocking, str(wav))

    def _detect_blocking(self, wav: str) -> list[SpeechChunk]:
        try:
            import numpy as np
            import sherpa_onnx
            from scipy.io import wavfile
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD requires sherpa-onnx, scipy, numpy and soundfile"
            ) from exc

        sr, data = wavfile.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1)
        else:
            data = data.astype(np.float32)
        if sr != self.sample_rate:
            raise RuntimeError(f"VAD expects {self.sample_rate}Hz audio, got {sr}Hz")

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = self.model_path
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.min_silence_duration = 0.5
        cfg.silero_vad.max_speech_duration = 30.0
        # sherpa keeps a circular audio buffer until a segment can be emitted.
        # Allocate for this file so a long mostly-silent recording does not
        # overwrite samples before flush; this is ~150MB for a 38-minute file.
        vad = sherpa_onnx.VoiceActivityDetector(
            cfg, buffer_size_in_seconds=max(60.0, len(data) / sr + 1.0)
        )
        chunks: list[SpeechChunk] = []

        def drain_segments() -> None:
            while not vad.empty():
                segment = vad.front
                start = segment.start / sr
                samples = np.asarray(segment.samples, dtype=np.float32)
                buffer = io.BytesIO()
                sf.write(buffer, samples, sr, subtype="PCM_16", format="WAV")
                chunks.append(
                    SpeechChunk(start, start + len(samples) / sr, buffer.getvalue())
                )
                vad.pop()

        for i in range(0, len(data), 512):
            vad.accept_waveform(data[i : i + 512])
            # Consume completed segments continuously. Keeping all completed
            # speech in sherpa's internal queue overflows on long meetings.
            drain_segments()
        vad.flush()
        drain_segments()
        return chunks

    def is_ready(self) -> bool:
        return bool(self.model_path and Path(self.model_path).is_file())


class ResemblyzerDiarizationStage:
    """VAD 段级声纹 embedding + 自动簇数聚类的本地 SD 实现。"""

    def __init__(
        self,
        profile_store: SpeakerProfileStore | None = None,
        *,
        cluster_distance_threshold: float = 0.30,
        max_speakers: int = 8,
    ) -> None:
        self.profile_store = profile_store
        self.cluster_distance_threshold = cluster_distance_threshold
        self.max_speakers = max(2, max_speakers)

    async def assign(self, wav: str | Path, segments: list[Segment]) -> None:
        if not segments:
            return
        try:
            assignments = await asyncio.to_thread(
                self._assign_blocking, str(wav), segments
            )
        except Exception as exc:
            logger.warning("diarization failed, skip: %s", exc)
            return
        for segment, (label, match) in zip(segments, assignments):
            segment.speaker = label
            if match is not None:
                segment.speaker_id = match.speaker_id
                segment.speaker_name = match.display_name
                segment.speaker_similarity = match.similarity
                segment.speaker_confidence = match.confidence

    def _assign_blocking(
        self, wav: str, segments: list[Segment]
    ) -> list[tuple[str, SpeakerMatch | None]]:
        import numpy as np
        from resemblyzer import VoiceEncoder
        from scipy.io import wavfile

        sr, data = wavfile.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1)

        encoder = VoiceEncoder(device="cpu")
        embeddings: list[np.ndarray] = []
        for segment in segments:
            start = max(0, int(segment.start * sr))
            end = min(data.shape[0], int(segment.end * sr))
            chunk = data[start:end]
            if chunk.shape[0] < int(0.4 * sr):
                embeddings.append(np.zeros(256, dtype=np.float32))
                continue
            try:
                embeddings.append(encoder.embed_utterance(chunk))
            except Exception:
                embeddings.append(np.zeros(256, dtype=np.float32))

        valid = [i for i, embedding in enumerate(embeddings) if embedding.any()]
        if len(valid) < 2:
            match = (
                self.profile_store.match(embeddings[valid[0]])
                if valid and self.profile_store is not None
                else None
            )
            return [("S1", match) for _ in segments]
        matrix = np.stack([embeddings[i] for i in valid])
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        raw = self._cluster_embeddings(matrix)

        mapped: dict[int, str] = {}
        for position, segment_index in enumerate(valid):
            cluster_id = int(raw[position])
            if cluster_id not in mapped:
                mapped[cluster_id] = f"S{len(mapped) + 1}"
        cluster_matches: dict[int, SpeakerMatch | None] = {}
        if self.profile_store is not None:
            for cluster_id in mapped:
                cluster_vectors = [
                    embeddings[index]
                    for index, raw_cluster in zip(valid, raw)
                    if int(raw_cluster) == cluster_id
                ]
                centroid = np.mean(np.stack(cluster_vectors), axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-9
                cluster_matches[cluster_id] = self.profile_store.match(centroid)

        assignments: list[tuple[str, SpeakerMatch | None]] = []
        last = "S1"
        last_match: SpeakerMatch | None = None
        for index in range(len(segments)):
            cluster_id = int(raw[valid.index(index)]) if index in valid else None
            if cluster_id is not None:
                last = mapped[cluster_id]
                last_match = cluster_matches.get(cluster_id)
            assignments.append((last, last_match))
        return assignments

    def _cluster_embeddings(self, matrix):
        """按余弦距离切树，避免把多人强行压成两个簇。"""
        from scipy.cluster.hierarchy import fcluster, linkage

        tree = linkage(matrix, method="average", metric="cosine")
        raw = fcluster(
            tree,
            self.cluster_distance_threshold,
            criterion="distance",
        )
        if len(set(raw)) > self.max_speakers:
            raw = fcluster(tree, self.max_speakers, criterion="maxclust")
        logger.info(
            "diarization auto clusters=%d valid_segments=%d",
            len(set(raw)),
            len(matrix),
        )
        return raw

    def is_ready(self) -> bool:
        try:
            import resemblyzer  # noqa: F401
            import scipy  # noqa: F401
        except ImportError:
            return False
        return True


class AudioWorkflow:
    """默认音频流程门面；具体编排由 WorkflowBuilder/WorkflowEngine 执行。"""

    def __init__(
        self,
        vad: VADStage,
        asr: ASRStage,
        punctuation: PunctuationStage | None = None,
        diarization: SpeakerDiarizationStage | None = None,
        *,
        correction: WorkflowStage | None = None,
        text_cleanup: WorkflowStage | None = None,
        language: str = "zh",
    ) -> None:
        self.vad = vad
        self.asr = asr
        self.punctuation = punctuation or PassthroughPunctuation()
        self.diarization = diarization
        self.language = language
        builder = WorkflowBuilder(language=language)
        builder.use(VadWorkflowStage(vad))
        builder.use(AsrWorkflowStage(asr))
        builder.use(PunctuationWorkflowStage(self.punctuation))
        if diarization is not None:
            builder.use(DiarizationWorkflowStage(diarization))
        if correction is not None and text_cleanup is not None:
            raise ValueError("use correction or text_cleanup, not both")
        self.correction = correction or text_cleanup
        if self.correction is not None:
            builder.use(self.correction)
        self.engine = builder.build()

    async def run(self, wav: str | Path, *, filename: str = "speech.wav") -> WorkflowResult:
        return await self.engine.run(wav, filename=filename)

    def stage_status(self) -> dict[str, object]:
        result = self.engine.stage_status()
        if self.correction is None:
            result["correction"] = {"enabled": False, "ok": True}
        if self.diarization is None:
            result["sd"] = {"enabled": False, "ok": True}
        return result
