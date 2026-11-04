"""Astra 会议处理管线: 音频 Workflow → LLM 纪要。

引擎组合(全部本地, 复用已有依赖):
  - Workflow: Silero VAD → 配置的 ASR → 标点恢复 → resemblyzer SD
  - 纪要: OpenAICompatLLMClient -> omlx (Qwen3-8B / Hy-MT2)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .audio_adapter import (
    audio_buffer_to_wav_bytes,
    decode_audio_file,
    resample_audio_buffer,
)
from .audio_enhancement import AudioEnhancementPipeline, EnhancementContext
from .audio_separation import (
    AudioSeparationStage,
    OverlapDetection,
    OverlapDetector,
    SpectralOverlapDetector,
)
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

# 纪要引用锚点：[seg-XXXXXX]，与 timeline 域 segment_id 一致。
_CITATION_RE = re.compile(r"\[(seg-\d{6})\]")
_CITATION_INSTRUCTION = (
    "引用要求：每个要点末尾标注来源段落 ID，格式 [seg-XXXXXX]（可多个），"
    "只能使用逐字稿中出现的段落 ID，不得编造。"
)
_ACTION_ITEM_PROMPT = (
    "你是会议记录员。从逐字稿中提取待办行动项，每行一条，"
    "格式 `- [ ] 事项 — 负责人(如有) — 截止时间(如有)`。"
    "只记录明确指派或承诺的事项；没有则只输出“无明确待办”。"
)
_DECISION_PROMPT = (
    "你是会议记录员。从逐字稿中提取关键决策，每行一条，"
    "格式 `- 决策 — 依据 — 主要反对意见(如有)`。"
    "只记录已达成或明确拍板的决定；没有则只输出“无明确决策”。"
)
# 纪要 fan-out 子任务（互相独立、各自超时；translation 依赖 summary）。
NOTE_TASKS: tuple[str, ...] = ("summary", "actions", "decisions")


def extract_citations(summary: str) -> dict[str, int]:
    """统计纪要文本中 [seg-XXXXXX] 引用出现次数（按出现顺序）。"""
    citations: dict[str, int] = {}
    for match in _CITATION_RE.finditer(summary or ""):
        segment_id = match.group(1)
        citations[segment_id] = citations.get(segment_id, 0) + 1
    return citations


def _assign_segment_ids(segments: list[Segment]) -> None:
    """按 timeline 顺序赋稳定锚点 seg-XXXXXX（幂等：已赋值不覆盖）。"""
    for index, segment in enumerate(segments, 1):
        if not segment.segment_id:
            segment.segment_id = f"seg-{index:06d}"


class CheckpointError(ValueError):
    """segments.json checkpoint 缺失、损坏或版本不受支持。"""


_CHECKPOINT_VERSION = 1


def write_segments_checkpoint(
    checkpoint_dir: str | Path,
    *,
    filename: str,
    duration_s: float,
    language: str,
    segments: list[Segment],
) -> Path:
    """ASR 完成后原子落盘分段 checkpoint（不含音频字节），供断点续跑。"""
    directory = Path(checkpoint_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _CHECKPOINT_VERSION,
        "filename": filename,
        "duration_s": duration_s,
        "language": language,
        "segments": [
            {
                "segment_id": s.segment_id,
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker,
                "speaker_id": s.speaker_id,
                "speaker_name": s.speaker_name,
                "speaker_similarity": s.speaker_similarity,
                "speaker_confidence": s.speaker_confidence,
                "text": s.text,
                "raw_text": s.raw_text,
            }
            for s in segments
        ],
    }
    path = directory / "segments.json"
    temporary = path.with_name(".segments.json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def load_segments_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    """读取并校验 checkpoint；缺失/损坏/版本不符抛 CheckpointError。"""
    path = Path(checkpoint_dir).expanduser() / "segments.json"
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"checkpoint is corrupt: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be an object")
    if payload.get("version") != _CHECKPOINT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint version: {payload.get('version')!r}"
        )
    if not isinstance(payload.get("segments"), list):
        raise CheckpointError("checkpoint has no segments list")
    return payload


def segments_from_checkpoint(payload: dict[str, Any]) -> list[Segment]:
    """从 checkpoint payload 重建 Segment 列表（不含音频）。"""
    segments: list[Segment] = []
    for row in payload["segments"]:
        if not isinstance(row, dict):
            raise CheckpointError("checkpoint segment row must be an object")
        segments.append(
            Segment(
                start=float(row.get("start", 0.0)),
                end=float(row.get("end", 0.0)),
                text=str(row.get("text", "")),
                speaker=str(row.get("speaker", "")),
                speaker_id=row.get("speaker_id"),
                speaker_name=str(row.get("speaker_name", "")),
                speaker_similarity=row.get("speaker_similarity"),
                speaker_confidence=str(row.get("speaker_confidence", "")),
                audio=None,
                raw_text=str(row.get("raw_text", "")),
                segment_id=str(row.get("segment_id", "")),
            )
        )
    return segments


@dataclass
class MeetingResult:
    filename: str
    duration_s: float
    language: str
    segments: list[Segment] = field(default_factory=list)
    speaker_labels: dict[str, str] = field(default_factory=dict)  # 簇id -> 标签
    summary: str = ""
    translation: str = ""
    note_tasks: dict[str, dict[str, object]] = field(default_factory=dict)
    citations: dict[str, int] = field(default_factory=dict)  # seg id -> 引用次数
    resumed: bool = False  # 从 segments.json checkpoint 续跑（跳过 ASR）
    enhancement_metrics: list[dict[str, object]] = field(default_factory=list)
    separation_metrics: list[dict[str, object]] = field(default_factory=list)
    separation_artifacts: list[str] = field(default_factory=list)
    overlap_detection: dict[str, object] | None = None

    def timeline_text(self) -> str:
        """时间轴逐字稿 (每行: [mm:ss] [seg-XXXXXX] S# 文本)。"""
        lines = []
        for s in self.segments:
            mm, ss = int(s.start // 60), int(s.start % 60)
            who = s.speaker_name or s.speaker or "?"
            anchor = f" [{s.segment_id}]" if s.segment_id else ""
            lines.append(f"[{mm:02d}:{ss:02d}]{anchor} {who} {s.text.strip()}")
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
    note_tasks: dict[str, dict[str, object]] = field(default_factory=dict)


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
        separation: AudioSeparationStage | None = None,
        separation_trigger: str = "manual",
        overlap_detector: OverlapDetector | None = None,
        inline_asr: Any = None,
        speaker_store: Any = None,
        note_task_timeout_seconds: float = 300.0,
        summary_timeout_seconds: float = 30 * 60,
        note_task_max_concurrency: int = 3,
    ) -> None:
        # VAD 提供时间戳，ASR 负责文本。
        self.vad_model = vad_model or str(
            Path(__file__).resolve().parents[3] / "models" / "silero_vad.onnx"
        )
        self.note_task_timeout_seconds = note_task_timeout_seconds
        # 摘要/翻译依赖全量逐字稿, 分块串行耗时远超行式子任务; 单独放宽超时。
        self.summary_timeout_seconds = summary_timeout_seconds
        # fan-out 子任务各自发起一次 LLM 请求; 本地 16GB 机型同时预填多路请求
        # 会撑爆 KV 缓存, 用信号量限制真正并发的 LLM 请求数(默认串行)。
        self.note_task_max_concurrency = max(1, note_task_max_concurrency)
        # F3: 默认 stage 组合走 fan-out 并行；外部注入 stage 保持旧串行语义。
        self._stages_default = summary_stage is None and translation_stage is None
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
        if separation_trigger not in {"manual", "overlap", "always"}:
            raise ValueError(f"unsupported separation trigger: {separation_trigger}")
        self.separation = separation
        self.separation_trigger = separation_trigger
        self.overlap_detector = overlap_detector or SpectralOverlapDetector()
        self.inline_asr = inline_asr
        self.speaker_store = speaker_store
        self.workflow = workflow or AudioWorkflow(
            self.vad,
            self.asr,
            self.punctuation,
            self.diarization,
            correction=self.correction_stage,
            inline_asr=inline_asr,
            speaker_store=speaker_store,
        )
        self.separated_workflow = AudioWorkflow(
            self.vad,
            self.asr,
            self.punctuation,
            None,
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

    async def _transcribe_separated(
        self,
        audio_buffer: Any,
        *,
        artifact_dir: str | Path | None = None,
    ) -> tuple[str, list[Segment], dict[str, object], list[str]]:
        if self.separation is None:
            raise RuntimeError("separation stage is not configured")
        separated_input = await asyncio.to_thread(
            resample_audio_buffer, audio_buffer, self.separation.input_sample_rate
        )
        tracks, metrics = await self.separation.process(
            separated_input,
            EnhancementContext(realtime=False),
        )
        metric_dict = metrics.to_dict()
        if metrics.status != "applied" or not tracks:
            return "zh", [], metric_dict, []

        output_dir = Path(artifact_dir).expanduser() if artifact_dir else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []
        segments: list[Segment] = []
        language = "zh"
        with tempfile.TemporaryDirectory(prefix="astra_separated_") as directory:
            for source_index, track in enumerate(tracks):
                track_for_asr = await asyncio.to_thread(
                    resample_audio_buffer, track, 16_000
                )
                track_wav = audio_buffer_to_wav_bytes(track_for_asr)
                if output_dir is not None:
                    artifact = output_dir / f"source-{source_index}.wav"
                    artifact.write_bytes(track_wav)
                    artifacts.append(str(artifact.resolve()))
                track_path = Path(directory) / f"source-{source_index}.wav"
                track_path.write_bytes(track_wav)
                result = await self.separated_workflow.run(
                    track_path,
                    filename=track_path.name,
                )
                language = result.language
                for segment in result.segments:
                    segment.speaker = f"S{source_index + 1}"
                    segment.speaker_id = None
                    segment.speaker_name = ""
                    segment.speaker_similarity = None
                    segment.speaker_confidence = "separated"
                    segments.append(segment)
        segments.sort(key=lambda item: (item.start, item.speaker, item.end))
        return language, segments, metric_dict, artifacts

    # ------------------------------------------------------------------ #
    # 4. LLM 纪要 (摘要 / 待办 / 决策 并行 fan-out + 翻译)
    # ------------------------------------------------------------------ #
    async def _map_reduce(
        self,
        transcript: str,
        *,
        sys_prompt: str,
        topic_hint: str,
        chunk_tokens: int,
        max_tokens: int = 1024,
        merge_sys_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        """逐字稿切块: 各块独立出要点, 超一块时用 LLM 合并。

        避免单请求 prefill 超 16GB 机型的 KV 内存上限。
        """
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

        # ---- 各块独立要点 ----
        per_chunk: list[str] = []
        for i, chunk in enumerate(chunks):
            sys = (
                sys_prompt
                + (f"\n这是第{i+1}段/共{len(chunks)}段。" if len(chunks) > 1 else "")
                + topic_hint
            )
            part_tokens: list[str] = []
            async for tok in self.llm.stream_chat(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": chunk},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                chat_template_kwargs={"enable_thinking": False},
            ):
                part_tokens.append(tok)
            per_chunk.append("".join(part_tokens).strip())

        # ---- 合并成最终纪要 ----
        if len(per_chunk) == 1:
            return per_chunk[0]
        if not merge_sys_prompt:
            # 行式产物(待办/决策)直接拼接, 不额外烧一次 LLM 合并
            return "\n".join(text for text in per_chunk if text)
        joined = "\n\n---\n\n".join(
            f"### 第{i+1}段要点\n{text}" for i, text in enumerate(per_chunk) if text
        )
        sys_merge = merge_sys_prompt + topic_hint
        merged: list[str] = []
        async for tok in self.llm.stream_chat(
            [
                {"role": "system", "content": sys_merge},
                {"role": "user", "content": joined},
            ],
            temperature=temperature,
            max_tokens=2048,
            chat_template_kwargs={"enable_thinking": False},
        ):
            merged.append(tok)
        return "".join(merged).strip()

    def _transcript_with_ids(self, segments: list[Segment]) -> tuple[str, str]:
        """返回 (带 seg id 的逐字稿文本, 引用规则提示)。"""
        transcript = "\n".join(
            f"[{s.segment_id or '?'}][{s.speaker_name or s.speaker or '?'}] {s.text.strip()}"
            for s in segments
            if s.text.strip()
        )
        has_ids = bool(transcript) and all(
            s.segment_id for s in segments if s.text.strip()
        )
        return transcript, (_CITATION_INSTRUCTION if has_ids else "")

    async def _generate_summary(
        self,
        segments: list[Segment],
        *,
        meeting_topic: str = "",
        chunk_tokens: int = 6000,
        prompt_template_id: str = DEFAULT_MEETING_PROMPT_ID,
    ) -> str:
        """生成中文纪要正文；无 llm 时返回空。"""
        prompt_template = get_meeting_prompt_template(
            prompt_template_id,
            custom_templates_path=self.prompt_templates_path,
        )
        if self.llm is None or not getattr(self.llm, "model", ""):
            return ""
        transcript, citation_rule = self._transcript_with_ids(segments)
        if not transcript.strip():
            return ""
        topic_hint = f"\n会议主题: {meeting_topic}" if meeting_topic else ""
        sys_prompt = prompt_template.chunk_system_prompt
        if citation_rule:
            sys_prompt += f"\n{citation_rule}"
        merge_prompt = prompt_template.merge_system_prompt
        if citation_rule:
            merge_prompt += f"\n合并时保留每条要点原有的 [seg-XXXXXX] 引用标注。{citation_rule}"
        return await self._map_reduce(
            transcript,
            sys_prompt=sys_prompt,
            topic_hint=topic_hint,
            chunk_tokens=chunk_tokens,
            merge_sys_prompt=merge_prompt,
        )

    async def _generate_actions(
        self, segments: list[Segment], *, meeting_topic: str = "", chunk_tokens: int = 6000
    ) -> str:
        """提取待办行动项（行式产物，多块直接拼接不二次合并）。"""
        if self.llm is None or not getattr(self.llm, "model", ""):
            return ""
        transcript, citation_rule = self._transcript_with_ids(segments)
        if not transcript.strip():
            return ""
        topic_hint = f"\n会议主题: {meeting_topic}" if meeting_topic else ""
        sys_prompt = _ACTION_ITEM_PROMPT + (f"\n{citation_rule}" if citation_rule else "")
        return await self._map_reduce(
            transcript,
            sys_prompt=sys_prompt,
            topic_hint=topic_hint,
            chunk_tokens=chunk_tokens,
            max_tokens=512,
        )

    async def _generate_decisions(
        self, segments: list[Segment], *, meeting_topic: str = "", chunk_tokens: int = 6000
    ) -> str:
        """提取关键决策（行式产物，多块直接拼接不二次合并）。"""
        if self.llm is None or not getattr(self.llm, "model", ""):
            return ""
        transcript, citation_rule = self._transcript_with_ids(segments)
        if not transcript.strip():
            return ""
        topic_hint = f"\n会议主题: {meeting_topic}" if meeting_topic else ""
        sys_prompt = _DECISION_PROMPT + (f"\n{citation_rule}" if citation_rule else "")
        return await self._map_reduce(
            transcript,
            sys_prompt=sys_prompt,
            topic_hint=topic_hint,
            chunk_tokens=chunk_tokens,
            max_tokens=512,
        )

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
        on_note_task: Any = None,
        note_tasks: dict[str, dict[str, object]] | None = None,
    ) -> tuple[str, str]:
        """生成纪要结果。

        默认 stage 组合: summary/actions/decisions 三个子任务并发 fan-out,
        各自独立超时; translation 依赖 summary, 在 summary 落定后执行。
        外部注入 summary_stage/translation_stage 时保持旧串行语义(可插拔)。
        note_tasks 传入 dict 时原位填充各子任务状态。
        """
        context = MeetingContext(
            segments=segments,
            meeting_topic=meeting_topic,
            target_language=target_language,
            chunk_tokens=chunk_tokens,
            prompt_template_id=prompt_template_id,
            translate=translate,
            note_tasks=note_tasks if note_tasks is not None else {},
        )
        if not self._stages_default:
            await self.summary_stage.run(context)
            await self.translation_stage.run(context)
            return context.summary, context.translation
        await self._run_note_tasks(context, on_note_task=on_note_task)
        return context.summary, context.translation

    async def _run_note_tasks(
        self, context: MeetingContext, *, on_note_task: Any = None
    ) -> None:
        """并发执行纪要子任务; 单任务超时/失败不拖垮其他任务。"""
        import functools

        # 限制同时在途的 LLM 请求数, 避免多路 prefill 撑爆 16GB 机型的 KV 缓存。
        semaphore = asyncio.Semaphore(self.note_task_max_concurrency)
        generators: dict[str, Any] = {
            "summary": functools.partial(
                self._generate_summary,
                context.segments,
                meeting_topic=context.meeting_topic,
                chunk_tokens=context.chunk_tokens,
                prompt_template_id=context.prompt_template_id,
            ),
            "actions": functools.partial(
                self._generate_actions,
                context.segments,
                meeting_topic=context.meeting_topic,
                chunk_tokens=context.chunk_tokens,
            ),
            "decisions": functools.partial(
                self._generate_decisions,
                context.segments,
                meeting_topic=context.meeting_topic,
                chunk_tokens=context.chunk_tokens,
            ),
        }
        await asyncio.gather(
            *(
                self._run_single_note_task(
                    context,
                    name,
                    factory(),
                    on_note_task,
                    timeout=self._timeout_for(name),
                    semaphore=semaphore,
                )
                for name, factory in generators.items()
            )
        )
        summary_entry = context.note_tasks.get("summary", {})
        context.summary = str(summary_entry.get("text", ""))
        if context.translate and context.summary:
            await self._run_single_note_task(
                context,
                "translation",
                self._translate_summary(
                    context.summary, target_language=context.target_language
                ),
                on_note_task,
                timeout=self.summary_timeout_seconds,
            )
            context.translation = str(
                context.note_tasks.get("translation", {}).get("text", "")
            )

    def _timeout_for(self, name: str) -> float:
        """摘要走更宽松的超时; 行式待办/决策保持短超时。"""
        return (
            self.summary_timeout_seconds
            if name == "summary"
            else self.note_task_timeout_seconds
        )

    async def _run_single_note_task(
        self,
        context: MeetingContext,
        name: str,
        coro: Any,
        on_note_task: Any = None,
        *,
        timeout: float | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        effective_timeout = (
            timeout if timeout is not None else self.note_task_timeout_seconds
        )
        entry: dict[str, object] = {"status": "processing", "text": "", "elapsed_s": 0.0}
        context.note_tasks[name] = entry
        if on_note_task is not None:
            on_note_task(name, dict(entry))
        start = time.time()

        def _finish(status: str, **extra: object) -> None:
            entry["status"] = status
            entry["elapsed_s"] = round(time.time() - start, 1)
            entry.update(extra)
            if on_note_task is not None:
                on_note_task(name, dict(entry))

        try:
            if semaphore is not None:
                async with semaphore:
                    text = await asyncio.wait_for(coro, timeout=effective_timeout)
            else:
                text = await asyncio.wait_for(coro, timeout=effective_timeout)
            _finish("done", text=str(text).strip())
        except asyncio.TimeoutError:
            _finish(
                "timeout",
                error=f"note task timed out after {effective_timeout:g}s",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 单任务失败不影响其他子任务
            _finish("failed", error=str(exc)[:500])

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
        separate: bool = False,
        separation_artifact_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
        on_note_task: Any = None,
    ) -> MeetingResult:
        t0 = time.time()
        tmp_parent: Path | None = None
        wav: str | None = None

        # ---- F2 断点续跑：checkpoint 有效时跳过解码/增强/ASR 全部音频阶段 ----
        checkpoint_payload: dict[str, Any] | None = None
        if resume:
            checkpoint_payload = load_segments_checkpoint(checkpoint_dir or Path("."))
        src_is_path = isinstance(audio, (str, Path)) and Path(audio).exists()

        if checkpoint_payload is not None:
            segs = segments_from_checkpoint(checkpoint_payload)
            _assign_segment_ids(segs)
            result = MeetingResult(
                filename=os.path.basename(filename),
                duration_s=float(checkpoint_payload.get("duration_s", 0.0)),
                language=str(checkpoint_payload.get("language", "zh")),
                segments=segs,
                resumed=True,
            )
            if summarize:
                result.summary, result.translation = await self.summarize(
                    segs,
                    meeting_topic=topic,
                    translate=do_translate,
                    prompt_template_id=prompt_template_id,
                    on_note_task=on_note_task,
                    note_tasks=result.note_tasks,
                )
                result.citations = extract_citations(result.summary)
            logger.info(
                "meeting resumed: %d segs from checkpoint, %.1fs wall",
                len(segs), time.time() - t0,
            )
            return result

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
            separation_metrics: list[dict[str, object]] = []
            separation_artifacts: list[str] = []
            overlap_detection: dict[str, object] | None = None
            separation_requested = separate or self.separation_trigger == "always"
            needs_overlap_detection = (
                self.separation is not None
                and self.separation_trigger == "overlap"
                and not separate
            )
            audio_buffer = None
            if self.enhancement is not None or separation_requested or needs_overlap_detection:
                audio_buffer = await asyncio.to_thread(decode_audio_file, wav)
            if self.enhancement is not None and audio_buffer is not None:
                audio_buffer, metrics = await self.enhancement.process(
                    audio_buffer,
                    EnhancementContext(realtime=False),
                )
                enhancement_metrics = [item.to_dict() for item in metrics]
                if any(item["status"] == "applied" for item in enhancement_metrics):
                    Path(wav).write_bytes(audio_buffer_to_wav_bytes(audio_buffer))
            if needs_overlap_detection and audio_buffer is not None:
                try:
                    detection = await asyncio.to_thread(
                        self.overlap_detector.detect, audio_buffer
                    )
                except Exception as exc:
                    detector_name = getattr(
                        self.overlap_detector, "name", "overlap_detector"
                    )
                    logger.warning("overlap detector failed, keep mixed audio: %s", exc)
                    detection = OverlapDetection(
                        status="failed",
                        suspected=False,
                        score=0.0,
                        active_frames=0,
                        candidate_frames=0,
                        threshold=1.0,
                        details={"detector": detector_name, "error": str(exc)},
                    )
                overlap_detection = detection.to_dict()
                separation_requested = detection.suspected
            use_separation = self.separation is not None and separation_requested
            if use_separation:
                try:
                    if audio_buffer is None:
                        raise RuntimeError("separation input was not decoded")
                    lang, segs, metrics, separation_artifacts = (
                        await self._transcribe_separated(
                            audio_buffer,
                            artifact_dir=separation_artifact_dir,
                        )
                    )
                    separation_metrics = [metrics]
                    if metrics.get("status") != "applied":
                        lang, segs = await self.transcribe(wav, progress=progress)
                except Exception as exc:
                    logger.warning("separation failed, keep mixed audio: %s", exc)
                    separation_metrics = [{
                        "stage_name": getattr(self.separation, "name", "separation"),
                        "status": "failed",
                        "fallback_reason": str(exc),
                    }]
                    lang, segs = await self.transcribe(wav, progress=progress)
            else:
                lang, segs = await self.transcribe(wav, progress=progress)
            # 引用锚点：所有路径统一按 timeline 顺序赋 seg-XXXXXX。
            _assign_segment_ids(segs)
            # F2: ASR(最耗时阶段)完成即落盘 checkpoint，失败后可断点续跑。
            if checkpoint_dir is not None:
                write_segments_checkpoint(
                    checkpoint_dir,
                    filename=os.path.basename(filename),
                    duration_s=dur,
                    language=lang,
                    segments=segs,
                )
            result = MeetingResult(
                filename=os.path.basename(filename), duration_s=dur,
                language=lang, segments=segs,
                enhancement_metrics=enhancement_metrics,
                separation_metrics=separation_metrics,
                separation_artifacts=separation_artifacts,
                overlap_detection=overlap_detection,
            )
            if summarize:
                result.summary, result.translation = await self.summarize(
                    segs,
                    meeting_topic=topic,
                    translate=do_translate,
                    prompt_template_id=prompt_template_id,
                    on_note_task=on_note_task,
                    note_tasks=result.note_tasks,
                )
                result.citations = extract_citations(result.summary)
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
