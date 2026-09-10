"""Astra 会议处理管线: 音频 Workflow → LLM 纪要。

引擎组合(全部本地, 复用已有依赖):
  - Workflow: Silero VAD → 配置的 ASR → 标点恢复 → resemblyzer SD
  - 纪要: OpenAICompatLLMClient -> omlx (Qwen3-8B / Hy-MT2)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .audio_adapter import audio_buffer_to_wav_bytes, decode_audio_file
from .audio_enhancement import AudioEnhancementPipeline, EnhancementContext
from .workflow import (
    AudioWorkflow,
    PassthroughPunctuation,
    ResemblyzerDiarizationStage,
    Segment,
    SileroVADStage,
    WorkflowEngine,
    WorkflowStage,
)
from .meeting_prompts import DEFAULT_MEETING_PROMPT_ID, get_meeting_prompt_template

logger = logging.getLogger(__name__)

_DEFAULT_DIARIZATION = object()


@dataclass
class MeetingResult:
    filename: str
    duration_s: float
    language: str
    segments: list[Segment] = field(default_factory=list)
    speaker_labels: dict[str, str] = field(default_factory=dict)  # 簇id -> 标签
    summary: str = ""
    translation: str = ""
    enhancement_metrics: list[dict[str, object]] = field(default_factory=list)

    def timeline_text(self) -> str:
        """时间轴逐字稿 (每行: [mm:ss] S# 文本)。"""
        lines = []
        for s in self.segments:
            mm, ss = int(s.start // 60), int(s.start % 60)
            who = s.speaker_name or s.speaker or "?"
            lines.append(f"[{mm:02d}:{ss:02d}] {who} {s.text.strip()}")
        return "\n".join(lines)


@dataclass
class MeetingContext:
    """纪要阶段之间传递的上下文。"""

    segments: list[Segment]
    meeting_topic: str = ""
    target_language: str = "简体中文"
    chunk_tokens: int = 6000
    prompt_template_id: str = DEFAULT_MEETING_PROMPT_ID
    translate: bool = True
    summary: str = ""
    translation: str = ""


class MeetingStage(Protocol):
    name: str

    async def run(self, context: MeetingContext) -> None: ...


class SummaryStage:
    name = "summary"

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    async def run(self, context: MeetingContext) -> None:
        context.summary = await self.generator(
            context.segments,
            meeting_topic=context.meeting_topic,
            chunk_tokens=context.chunk_tokens,
            prompt_template_id=context.prompt_template_id,
        )


class TranslationStage:
    name = "translation"

    def __init__(self, translator: Any) -> None:
        self.translator = translator

    async def run(self, context: MeetingContext) -> None:
        if not context.translate or not context.summary:
            return
        context.translation = await self.translator(
            context.summary,
            target_language=context.target_language,
        )


class MeetingPipeline:
    """离线会议处理: transcribe -> diarize -> summarize(可选)。"""

    def __init__(
        self,
        llm: Any = None,
        mt_llm: Any = None,
        asr: Any = None,
        vad_model: str = "",
        *,
        min_speaker_segments: int = 3,
        punctuation: Any = None,
        diarization: Any = _DEFAULT_DIARIZATION,
        workflow: WorkflowEngine | None = None,
        summary_stage: MeetingStage | None = None,
        translation_stage: MeetingStage | None = None,
        correction_stage: WorkflowStage | None = None,
        text_cleanup: WorkflowStage | None = None,
        prompt_templates_path: str | Path | None = None,
        enhancement: AudioEnhancementPipeline | None = None,
    ) -> None:
        # VAD 提供时间戳，ASR 负责文本。
        self.vad_model = vad_model or str(
            Path(__file__).resolve().parents[3] / "models" / "silero_vad.onnx"
        )
        self.asr = asr  # MlxAudioAsrClient (Qwen3), 逐段转写
        self.llm = llm          # 摘要/推理 (如 Qwen3-8B)
        self.mt_llm = mt_llm    # 翻译 (可选专用 MT, 如 Hy-MT2; 空则回落 llm)
        self.min_speaker_segments = min_speaker_segments
        self.vad = SileroVADStage(self.vad_model)
        self.punctuation = punctuation or PassthroughPunctuation()
        self.diarization = (
            ResemblyzerDiarizationStage()
            if diarization is _DEFAULT_DIARIZATION
            else diarization
        )
        if correction_stage is not None and text_cleanup is not None:
            raise ValueError("use correction_stage or text_cleanup, not both")
        self.correction_stage = correction_stage or text_cleanup
        self.prompt_templates_path = prompt_templates_path
        self.enhancement = enhancement
        self.workflow = workflow or AudioWorkflow(
            self.vad,
            self.asr,
            self.punctuation,
            self.diarization,
            correction=self.correction_stage,
        )
        self.summary_stage = summary_stage or SummaryStage(self._generate_summary)
        self.translation_stage = translation_stage or TranslationStage(
            self._translate_summary
        )

    # ------------------------------------------------------------------ #
    # 1. 音频解码
    # ------------------------------------------------------------------ #
    @staticmethod
    def decode_to_wav(src: str | Path) -> tuple[str, float]:
        """Decode any supported input to a 16k mono PCM16 WAV."""
        src = Path(src)
        audio = decode_audio_file(src, target_sample_rate=16_000)
        wav = Path(tempfile.mkdtemp(prefix="astra_meet_")) / f"{src.stem}.wav"
        wav.write_bytes(audio_buffer_to_wav_bytes(audio))
        duration = len(audio.samples) / audio.sample_rate
        return str(wav), float(duration)

    # ------------------------------------------------------------------ #
    # 2. VAD 切段 + Qwen3 逐段转写 (方案A: 时间戳来自 Silero VAD)
    # ------------------------------------------------------------------ #
    async def transcribe(self, wav: str, *, progress: Any = None) -> tuple[str, list[Segment]]:
        """执行 VAD -> ASR -> 标点恢复 -> SD，返回结构化段列表。"""
        result = await self.workflow.run(wav, filename=Path(wav).name)
        return result.language, result.segments

    # ------------------------------------------------------------------ #
    # 3. 说话人分离 (VAD 段级 resemblyzer embed + 自动簇数余弦聚类)
    # ------------------------------------------------------------------ #
    async def diarize(self, wav: str, segments: list[Segment]) -> None:
        """兼容旧调用方；新的 transcribe() 已在 Workflow 内完成 SD。"""
        await self.diarization.assign(wav, segments)

    # ------------------------------------------------------------------ #
    # 4. LLM 纪要 (摘要 + 翻译)
    # ------------------------------------------------------------------ #
    async def _generate_summary(
        self,
        segments: list[Segment],
        *,
        meeting_topic: str = "",
        chunk_tokens: int = 6000,
        prompt_template_id: str = DEFAULT_MEETING_PROMPT_ID,
    ) -> str:
        """生成中文纪要正文；无 llm 时返回空。

        逐字稿超过 chunk_tokens 时切块: 各块先独立出要点, 再合并成
        最终纪要——避免单请求 prefill 超 16GB 机型的 KV 内存上限。
        """
        prompt_template = get_meeting_prompt_template(
            prompt_template_id,
            custom_templates_path=self.prompt_templates_path,
        )
        if self.llm is None or not getattr(self.llm, "model", ""):
            return ""
        transcript = "\n".join(
            f"[{s.speaker_name or s.speaker or '?'}] {s.text.strip()}"
            for s in segments
            if s.text.strip()
        )
        if not transcript.strip():
            return ""
        # ---- 切块: 按字符粗估(中文1字≈1 token), 在段落边界断开 ----
        chunks: list[str] = []
        current: list[str] = []
        budget = 0
        for line in transcript.split("\n"):
            if current and budget + len(line) > chunk_tokens:
                chunks.append("\n".join(current))
                current, budget = [], 0
            current.append(line)
            budget += len(line)
        if current:
            chunks.append("\n".join(current))

        topic_hint = f"\n会议主题: {meeting_topic}" if meeting_topic else ""

        # ---- 各块独立要点 ----
        per_chunk: list[str] = []
        for i, chunk in enumerate(chunks):
            sys_prompt = (
                prompt_template.chunk_system_prompt
                + (f"\n这是第{i+1}段/共{len(chunks)}段。" if len(chunks) > 1 else "")
                + topic_hint
            )
            part_tokens: list[str] = []
            async for tok in self.llm.stream_chat(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.2,
                max_tokens=1024,
                chat_template_kwargs={"enable_thinking": False},
            ):
                part_tokens.append(tok)
            per_chunk.append("".join(part_tokens).strip())

        # ---- 合并成最终纪要 ----
        if len(per_chunk) == 1:
            summary = per_chunk[0]
        else:
            joined = "\n\n---\n\n".join(
                f"### 第{i+1}段要点\n{text}" for i, text in enumerate(per_chunk) if text
            )
            sys_merge = prompt_template.merge_system_prompt + topic_hint
            merged: list[str] = []
            async for tok in self.llm.stream_chat(
                [
                    {"role": "system", "content": sys_merge},
                    {"role": "user", "content": joined},
                ],
                temperature=0.2,
                max_tokens=2048,
                chat_template_kwargs={"enable_thinking": False},
            ):
                merged.append(tok)
            summary = "".join(merged).strip()

        return summary

    async def _translate_summary(
        self, summary: str, *, target_language: str = "简体中文"
    ) -> str:
        """只翻译最终纪要，不翻译逐字稿。"""
        mt = self.mt_llm or self.llm
        if not summary or mt is None or not getattr(mt, "model", ""):
            return ""
        trans_tokens: list[str] = []
        async for tok in mt.stream_chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"你是专业翻译。把会议纪要全文翻译成{target_language}，"
                        "保留 markdown 结构与 [ ] 行动项格式。只输出译文。"
                    ),
                },
                {"role": "user", "content": summary},
            ],
            temperature=0.0,
            max_tokens=4096,
            chat_template_kwargs={"enable_thinking": False},
        ):
            trans_tokens.append(tok)
        return "".join(trans_tokens).strip()

    async def summarize(
        self,
        segments: list[Segment],
        *,
        meeting_topic: str = "",
        target_language: str = "简体中文",
        chunk_tokens: int = 6000,
        translate: bool = True,
        prompt_template_id: str = DEFAULT_MEETING_PROMPT_ID,
    ) -> tuple[str, str]:
        """按可插拔的纪要、翻译阶段生成结果。"""
        context = MeetingContext(
            segments=segments,
            meeting_topic=meeting_topic,
            target_language=target_language,
            chunk_tokens=chunk_tokens,
            prompt_template_id=prompt_template_id,
            translate=translate,
        )
        await self.summary_stage.run(context)
        await self.translation_stage.run(context)
        return context.summary, context.translation

    # ------------------------------------------------------------------ #
    # 5. 一键处理
    # ------------------------------------------------------------------ #
    async def process(
        self,
        audio: bytes | str | Path,
        *,
        filename: str = "meeting.m4a",
        summarize: bool = True,
        do_translate: bool = True,
        topic: str = "",
        prompt_template_id: str = DEFAULT_MEETING_PROMPT_ID,
        progress: Any = None,
    ) -> MeetingResult:
        t0 = time.time()
        tmp_parent: Path | None = None
        wav: str | None = None
        src_is_path = isinstance(audio, (str, Path)) and Path(audio).exists()

        if src_is_path:
            src = Path(audio)
            wav, dur = await asyncio.to_thread(self.decode_to_wav, src)
        else:
            tmp_parent = Path(tempfile.mkdtemp(prefix="astra_meet_"))
            src = tmp_parent / os.path.basename(filename)
            src.write_bytes(bytes(audio))
            wav, dur = await asyncio.to_thread(self.decode_to_wav, src)

        try:
            enhancement_metrics: list[dict[str, object]] = []
            if self.enhancement is not None:
                audio_buffer = await asyncio.to_thread(decode_audio_file, wav)
                audio_buffer, metrics = await self.enhancement.process(
                    audio_buffer,
                    EnhancementContext(realtime=False),
                )
                enhancement_metrics = [item.to_dict() for item in metrics]
                if any(item["status"] == "applied" for item in enhancement_metrics):
                    Path(wav).write_bytes(audio_buffer_to_wav_bytes(audio_buffer))
            lang, segs = await self.transcribe(wav, progress=progress)
            result = MeetingResult(
                filename=os.path.basename(filename), duration_s=dur,
                language=lang, segments=segs,
                enhancement_metrics=enhancement_metrics,
            )
            if summarize:
                result.summary, result.translation = await self.summarize(
                    segs,
                    meeting_topic=topic,
                    translate=do_translate,
                    prompt_template_id=prompt_template_id,
                )
            logger.info(
                "meeting done: %.1fs audio, %d segs, %.1fs wall",
                dur, len(segs), time.time() - t0,
            )
            return result
        finally:
            if wav is not None:
                try:
                    Path(wav).unlink(missing_ok=True)
                except OSError:
                    pass
            for d in [Path(wav).parent if wav else None, tmp_parent]:
                if d is None:
                    continue
                try:
                    d.rmdir()
                except OSError:
                    pass
