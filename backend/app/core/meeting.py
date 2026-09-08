"""Astra 会议处理管线: 音频 Workflow → LLM 纪要。

引擎组合(全部本地, 复用已有依赖):
  - Workflow: Silero VAD → 配置的 ASR → 标点恢复 → resemblyzer SD
  - 纪要: OpenAICompatLLMClient -> omlx (Qwen3-8B / Hy-MT2)
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .workflow import (
    AudioWorkflow,
    PassthroughPunctuation,
    ResemblyzerDiarizationStage,
    Segment,
    SileroVADStage,
    WorkflowEngine,
    WorkflowStage,
)

logger = logging.getLogger(__name__)

SEGMENT_MIN_SECONDS = 1.0      # 短于它的 whisper 段并入前段
SILENCE_PAD_SECONDS = 0.25     # embed 前每段前后补一点, 避免切边爆音
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
        whisper_model: str = "",
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
    ) -> None:
        # VAD 提供时间戳，ASR 负责文本；旧 whisper_model 参数保留兼容。
        self.vad_model = vad_model or str(
            Path(__file__).resolve().parents[3] / "models" / "silero_vad.onnx"
        )
        self.asr = asr  # MlxAudioAsrClient (Qwen3), 逐段转写
        self.whisper_model = whisper_model or (
            "/Users/kbsonlong/.cache/modelscope/hub/models/mlx-community/"
            "whisper-large-v3-turbo"
        )
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
        """m4a/mp3/wav -> 16k mono PCM16 wav (afconvert, 无 ffmpeg 依赖)。

        afconvert 产出 WAVE_FORMAT_EXTENSIBLE, 本项目统一走 scipy 读取。
        """
        src = Path(src)
        wav = Path(tempfile.mkdtemp(prefix="astra_meet_")) / f"{src.stem}.wav"
        subprocess.run(
            [
                "/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                str(src), str(wav),
            ],
            check=True, capture_output=True,
        )
        # 时长: 用 wave/scipy 读帧数换算
        import numpy as np
        from scipy.io import wavfile
        sr, data = wavfile.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        dur = data.shape[0] / sr
        return str(wav), float(dur)

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
    ) -> str:
        """生成中文纪要正文；无 llm 时返回空。

        逐字稿超过 chunk_tokens 时切块: 各块先独立出要点, 再合并成
        最终纪要——避免单请求 prefill 超 16GB 机型的 KV 内存上限。
        """
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
                "你是严谨的会议纪要助手。基于带说话人标签(S1/S2/...)的逐字稿片段，"
                "只提取本片段明确说出的信息，不补充常识，不修正无法确认的专有名词，不臆造事实。"
                "区分‘已确认结论’、‘明确提出的行动项’和‘讨论中/待确认事项’："
                "只有明确承诺、明确要求或明确决定的内容才能列为行动项；没有明确负责人的行动项写‘负责人：待确认’，"
                "不要根据谁说得多、谁赞同来推断负责人。保留原始 S1/S2 标签和原文中的人名。"
                "同一事项只能出现一次：已列为行动项的内容不要再放入关键结论或待确认事项；"
                "关键结论只写已经形成的结论，不写‘需要做/应当做’的行动句。"
                "关键结论最多 8 条、明确行动项最多 8 条、待确认事项最多 5 条，优先保留最具体的内容。"
                "输出简体中文 markdown，严格使用以下三个小节："
                "### 关键结论、### 明确行动项、### 待确认事项。"
                "没有内容的小节写‘无’。不要输出发言人性格评价，不要使用 ``` 代码块。"
            ) + (f"\n这是第{i+1}段/共{len(chunks)}段。" if len(chunks) > 1 else "") + topic_hint
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
            sys_merge = (
                "你是严谨的会议纪要总编。下面是同一场长会议分段生成的要点。"
                "请合并为一份简体中文 markdown 纪要，去除重复内容，但不得新增逐字稿中没有的事实、"
                "决定、日期、负责人、团队或工具名称。只有原文明确提出、明确承诺或明确决定的事项才保留为行动项；"
                "负责人不明确时写‘负责人：待确认’，不得根据 S1/S2 的发言多少进行推断。"
                "不确定或仅为讨论/建议的内容放入待确认事项。保留原始人名和 S1/S2 标签。"
                "同一事项只能归入一个栏目：会议要点只放已形成的结论，明确行动项只放明确要求/承诺，"
                "待确认事项只放尚未决定的问题；行动项不得重复出现在其他栏目，待确认事项也不得重复行动项。"
                "同义或高度相似的条目只保留一条，确认过的结论不能再次列为待确认事项。"
                "会议要点最多 8 条、明确行动项最多 8 条、待确认事项最多 5 条。"
                "发言概览只概括各 S# 实际讨论的主题，不评价性格，不虚构分工。"
                "严格只输出以下四个小节，不要输出 ``` 代码块，不要重复任何小节："
                "## 会议要点、## 明确行动项、## 待确认事项、## 发言概览。"
                "每个小节都必须存在；没有内容写‘无’。"
            ) + topic_hint
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
    ) -> tuple[str, str]:
        """按可插拔的纪要、翻译阶段生成结果。"""
        context = MeetingContext(
            segments=segments,
            meeting_topic=meeting_topic,
            target_language=target_language,
            chunk_tokens=chunk_tokens,
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
            lang, segs = await self.transcribe(wav, progress=progress)
            result = MeetingResult(
                filename=os.path.basename(filename), duration_s=dur,
                language=lang, segments=segs,
            )
            if summarize:
                result.summary, result.translation = await self.summarize(
                    segs, meeting_topic=topic, translate=do_translate,
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
