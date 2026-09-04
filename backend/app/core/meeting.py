"""Astra 会议处理管线: 录音文件 → 时间戳转写 → 说话人分离 → LLM 纪要。

引擎组合(全部本地, 复用已有依赖):
  - 转写: mlx-whisper large-v3-turbo (ModelScope 本地缓存, 段级时间戳)
  - 分离: resemblyzer VoiceEncoder (torch CPU) + spectralcluster
  - 纪要: OpenAICompatLLMClient -> omlx (Qwen3-8B / Hy-MT2)

Speaker 对齐策略: 每个 whisper segment 单独 embed, 聚类后打标,
相邻同簇段合并。clusterer 自动估计说话人数。
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
from typing import Any, Sequence

logger = logging.getLogger(__name__)

SEGMENT_MIN_SECONDS = 1.0      # 短于它的 whisper 段并入前段
SILENCE_PAD_SECONDS = 0.25     # embed 前每段前后补一点, 避免切边爆音


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


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
            who = s.speaker or "?"
            lines.append(f"[{mm:02d}:{ss:02d}] {who} {s.text.strip()}")
        return "\n".join(lines)


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
    ) -> None:
        # 方案A: VAD 提供时间戳, Qwen3(asr) 提供文本; whisper 保留为
        # 可选对照引擎但默认不启用。
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
        self._whisper_lazy: Any = None

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
        """返回 (language, segments)。

        方案A 架构(时间戳与文本解耦):
        - Silero VAD 切语音段 -> 段级时间戳(when);
        - Qwen3-ASR 逐段转写(what), 中英混说强于 whisper;
        - VAD 段即自然语音单元, 天然适合段级声纹 embed(who)。

        VAD 参数: min_speech 0.25s 滤语气词碎片, min_silence 0.5s
        合并句间短停顿, max_speech 30s 对齐 Qwen3 训练窗口。
        """
        import numpy as np

        def _vad_blocking() -> list[tuple[float, float, np.ndarray]]:
            import sherpa_onnx
            from scipy.io import wavfile as _wf

            sr, data = _wf.read(wav)
            if data.ndim > 1:
                data = data.mean(axis=1)
            f32 = data.astype(np.float32) / (
                np.iinfo(data.dtype).max + 1 if np.issubdtype(data.dtype, np.integer) else 1
            )
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = self.vad_model
            cfg.silero_vad.min_speech_duration = 0.25
            cfg.silero_vad.min_silence_duration = 0.5
            cfg.silero_vad.max_speech_duration = 30.0
            vad = sherpa_onnx.VoiceActivityDetector(cfg)
            for i in range(0, len(f32), 512):
                vad.accept_waveform(f32[i : i + 512])
            vad.flush()
            out: list[tuple[float, float, np.ndarray]] = []
            while not vad.empty():
                seg = vad.front
                start = seg.start / sr
                dur = len(seg.samples) / sr
                out.append((start, start + dur, np.asarray(seg.samples, dtype=np.float32)))
                vad.pop()
            return out

        speech = await asyncio.to_thread(_vad_blocking)
        if not speech:
            return "zh", []

        # ---- Qwen3 逐段转写 (asr_client 自带 chunk/热词/系统提示) ----
        segments: list[Segment] = []
        for start, end, samples in speech:
            import soundfile as sf
            import tempfile as _tf
            import os as _os
            with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, samples, 16000, subtype="PCM_16")
                seg_wav = tf.name
            try:
                pcm = open(seg_wav, "rb").read()
                text = await self.asr.transcribe(pcm, filename=seg_wav)
            finally:
                _os.unlink(seg_wav)
            text = (text or "").strip()
            if not text:
                continue
            segments.append(Segment(float(start), float(end), text))
        return "zh", segments

    # ------------------------------------------------------------------ #
    # 3. 说话人分离 (VAD 段级 resemblyzer embed + ward 聚 2 簇)
    # ------------------------------------------------------------------ #
    async def diarize(self, wav: str, segments: list[Segment]) -> None:
        """原地给每个 segment 打 speaker 标签。短音频(<2段)直接返回。"""
        if len(segments) < 2:
            return
        try:
            labels = await asyncio.to_thread(self._diarize_blocking, wav, segments)
        except Exception as exc:
            logger.warning("diarization failed, skip: %s", exc)
            return
        for seg, label in zip(segments, labels):
            seg.speaker = label

    def _diarize_blocking(self, wav: str, segments: list[Segment]) -> list[str]:
        import numpy as np
        from scipy.io import wavfile
        from resemblyzer import VoiceEncoder

        sr, data = wavfile.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1)

        # 短音频(< 2 个有效段)不分离
        if len(segments) < 2:
            return ["S1"] * len(segments)

        # ---- 方案A: VAD 段级 embed (段=自然语音单元, 直接 embed) ----
        enc = VoiceEncoder(device="cpu")
        embs: list[np.ndarray] = []
        for seg in segments:
            i0 = max(0, int(seg.start * sr))
            i1 = min(data.shape[0], int(seg.end * sr))
            chunk = data[i0:i1]
            # <0.4s 无法可靠 embed -> 全零占位, 聚类前过滤
            if chunk.shape[0] < int(0.4 * sr):
                embs.append(np.zeros(256, dtype=np.float32))
                continue
            try:
                embs.append(enc.embed_utterance(chunk))
            except Exception:
                embs.append(np.zeros(256, dtype=np.float32))

        valid = [i for i, e in enumerate(embs) if e.any()]
        if len(valid) < 2:
            return ["S1"] * len(segments)
        E = np.stack([embs[i] for i in valid])
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

        # ---- 聚类: ward 层次聚类, 恒分 2 簇(主/次) ----
        from scipy.cluster.hierarchy import fcluster, linkage

        Z = linkage(E, method="ward")
        raw = fcluster(Z, 2, criterion="maxclust")  # 1/2
        # 簇 id -> 时间序稳定标签 (按首次出现)
        order: dict[int, str] = {}
        mapped: dict[int, str] = {}
        for idx_pos, seg_idx in enumerate(valid):
            cid = int(raw[idx_pos])
            if cid not in order:
                order[cid] = f"S{len(order) + 1}"
            mapped[seg_idx] = order[cid]
        # 无效段(全零)继承最近有效段标签
        labels: list[str] = []
        last = "S1"
        for i in range(len(segments)):
            if i in mapped:
                last = mapped[i]
            labels.append(mapped.get(i, last))
        return labels

    # ------------------------------------------------------------------ #
    # 4. LLM 纪要 (摘要 + 翻译)
    # ------------------------------------------------------------------ #
    async def summarize(
        self,
        segments: list[Segment],
        *,
        meeting_topic: str = "",
        target_language: str = "简体中文",
        chunk_tokens: int = 6000,
    ) -> tuple[str, str]:
        """返回 (summary_md, translation_md)。无 llm 时返回空。

        逐字稿超过 chunk_tokens 时切块: 各块先独立出要点, 再合并成
        最终纪要——避免单请求 prefill 超 16GB 机型的 KV 内存上限。
        """
        if self.llm is None or not getattr(self.llm, "model", ""):
            return "", ""
        transcript = "\n".join(
            f"[{s.speaker or '?'}] {s.text.strip()}" for s in segments if s.text.strip()
        )
        if not transcript.strip():
            return "", ""

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
                "你是专业会议纪要助手。基于带说话人标签(S1/S2/...)的逐字稿片段，"
                "输出 markdown 要点，覆盖: 关键决策与结论、行动项([ ] 事项 @负责人或S#)、"
                "发言人倾向。只基于本片段,不要臆造。"
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
                "你是会议纪要整理助手。下面是长会议分段生成的要点，把它们合并成一份"
                "结构清晰的完整 markdown 纪要: ## 会议要点(去重、按主题归类) / "
                "## 行动项(汇总去重) / ## 说话人分工(推断)。语言: 简体中文。"
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

        # ---- 翻译: 只翻最终纪要(一次请求), 不逐字翻全文 ----
        mt = self.mt_llm or self.llm
        translation = ""
        if summary and mt is not None and getattr(mt, "model", ""):
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
            translation = "".join(trans_tokens).strip()
        return summary, translation

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
            await self.diarize(wav, segs)
            if summarize:
                result.summary, result.translation = await self.summarize(
                    segs, meeting_topic=topic,
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
