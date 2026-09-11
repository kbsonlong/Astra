"""Astra 会议 worker —— 独立进程入口(由 API 以子进程方式调用)。

设计动因:
  MLX GPU stream 是 thread-local, 会议整条管线(Qwen3 逐段转写 + 声纹
  分离 + LLM 纪要)若在 API 进程内跑会长时间独占事件循环, 语音会话 /
  health / 其他请求全部被卡住。拆独立进程后:
    - 子进程拥有独立 MLX/Metal 上下文, 自己 load 一次模型并串行执行;
    - API 事件循环零阻塞, 语音的 MLX 不受影响;
    - 会议低频批量, 子进程启动多 load 一次模型(~2s)可接受。

用法(父进程):
  python -m app.core.meeting_cli --audio <file> --out <dir>
      [--topic T] [--prompt-template ID] [--prompt-templates-path PATH]
      [--no-summarize] [--no-translate]

产物写到 --out/: status.json(processing|done|failed) + report.md +
transcript.txt + meta.json + job.log(stdout/stderr)。
stdout 末行输出单行 JSON: {"status": "done|failed", "task_id": ...}。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo 根 (backend/app/core -> x4)

# 使 `python -m app.core.meeting_cli` 在未设 PYTHONPATH 时也能找到 app 包
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

# 复用 HTTP 层的渲染与引擎标签, 单一来源防漂移
from app.api.meeting_routes import ENGINE_LABEL, _render_markdown  # noqa: E402
from app.core.correction import parse_correction_rules  # noqa: E402
from app.core.meeting_prompts import (  # noqa: E402
    DEFAULT_MEETING_PROMPT_ID,
    get_meeting_prompt_template,
)
from app.core.speaker_registry import SpeakerProfileStore  # noqa: E402
from app.core.workflow import CorrectionStage, ResemblyzerDiarizationStage  # noqa: E402
from app.models.punctuation_client import build_punctuation_client  # noqa: E402


def _build_pipeline(prompt_templates_path: str = ""):
    """与 main.py 中会议管线一致的干净配置(无热词/无 system_prompt)。"""
    from app.config import Settings
    from app.core.meeting import MeetingPipeline
    from app.core.zipenhancer import build_audio_enhancement_pipeline
    from app.core.audio_separation import build_audio_separation_stage
    from app.core.audio_separation import build_overlap_detector
    from app.models.asr_client import MlxAudioAsrClient
    from app.models.llm_client import OpenAICompatLLMClient

    s = Settings.from_env()
    meeting_asr = MlxAudioAsrClient(
        s.asr_model,
        s.asr_language,
        max_tokens=s.asr_max_tokens,
        repetition_penalty=s.asr_repetition_penalty,
        repetition_context_size=s.asr_repetition_context_size,
        hotwords=(),
        system_prompt="",
    )
    llm = OpenAICompatLLMClient(
        s.llm_base_url,
        s.llm_model,
        s.llm_api_key,
        request_timeout_seconds=600.0,
        connect_timeout_seconds=5.0,
        stream_idle_timeout_seconds=120.0,
        chat_path=s.llm_chat_path,
        models_path=s.llm_models_path,
    )
    correction_stage = CorrectionStage(
        llm,
        rules_enabled=s.meeting_rule_correction_enabled,
        llm_enabled=s.meeting_llm_correction_enabled,
        candidate_rules=parse_correction_rules(s.meeting_llm_correction_candidates),
        system_prompt=s.llm_correction_system_prompt,
        max_tokens=s.llm_correction_max_tokens,
    )
    punctuation = build_punctuation_client(
        enabled=s.punctuation_enabled,
        engine=s.punctuation_engine,
        model=s.punctuation_model,
        device=s.punctuation_device,
    )
    sd_engine = s.sd_engine.lower()
    speaker_store = SpeakerProfileStore(
        s.speaker_store_path,
        match_threshold=s.speaker_match_threshold,
        match_margin=s.speaker_match_margin,
        duplicate_threshold=s.speaker_duplicate_threshold,
        sample_dir=s.speaker_sample_dir,
    )
    if sd_engine in {"", "none", "noop"}:
        diarization = None
    elif sd_engine in {"resemblyzer", "resemblyzer-ward"}:
        diarization = ResemblyzerDiarizationStage(
            profile_store=speaker_store, max_speakers=s.speaker_max_speakers
        )
    else:
        raise ValueError(f"unsupported SD engine: {s.sd_engine}")
    return MeetingPipeline(
        llm=llm,
        asr=meeting_asr,
        vad_model=s.vad_model,
        punctuation=punctuation,
        diarization=diarization,
        correction_stage=correction_stage,
        prompt_templates_path=prompt_templates_path,
        enhancement=build_audio_enhancement_pipeline(
            enabled=s.audio_enhancement_enabled,
            ans_model=s.audio_ans_model,
            model_dir=s.audio_enhancement_model_dir,
        ),
        separation=build_audio_separation_stage(
            enabled=s.audio_enhancement_enabled,
            separation_model=s.audio_separation_model,
            model_dir=s.audio_enhancement_model_dir,
            window_seconds=s.audio_separation_window_seconds,
        ),
        separation_trigger=s.audio_separation_trigger,
        overlap_detector=build_overlap_detector(
            model=s.audio_overlap_detector_model,
            model_dir=s.audio_overlap_model_dir,
        ),
    )


def _write_status(out_dir: Path, payload: dict) -> None:
    (out_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _merge_training_segments(segments: list[object], *, max_gap_s: float = 1.5) -> list[object]:
    """Merge adjacent same-speaker segments into complete training utterances."""
    if not segments:
        return []
    import io

    import numpy as np
    import soundfile as sf

    merged: list[object] = []
    for segment in segments:
        if not getattr(segment, "audio", None):
            continue
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        speaker_key = (
            getattr(previous, "speaker_id", None)
            or getattr(previous, "speaker_name", "")
            or getattr(previous, "speaker", "")
        )
        current_key = (
            getattr(segment, "speaker_id", None)
            or getattr(segment, "speaker_name", "")
            or getattr(segment, "speaker", "")
        )
        gap = float(segment.start) - float(previous.end)
        if speaker_key != current_key or not speaker_key or gap > max_gap_s:
            merged.append(segment)
            continue

        previous_audio, previous_sr = sf.read(io.BytesIO(previous.audio), dtype="float32", always_2d=False)
        current_audio, current_sr = sf.read(io.BytesIO(segment.audio), dtype="float32", always_2d=False)
        if previous_sr != current_sr:
            merged.append(segment)
            continue
        if previous_audio.ndim > 1:
            previous_audio = np.mean(previous_audio, axis=1)
        if current_audio.ndim > 1:
            current_audio = np.mean(current_audio, axis=1)
        silence = np.zeros(max(0, round(gap * previous_sr)), dtype=np.float32)
        combined = np.concatenate((previous_audio, silence, current_audio))
        buffer = io.BytesIO()
        sf.write(buffer, combined, previous_sr, subtype="PCM_16", format="WAV")
        previous.audio = buffer.getvalue()
        previous.end = segment.end
        text_separator = " " if previous.text[-1:].isascii() and segment.text[:1].isascii() else ""
        raw_separator = " " if previous.raw_text[-1:].isascii() and segment.raw_text[:1].isascii() else ""
        previous.text = f"{previous.text.rstrip()}{text_separator}{segment.text.lstrip()}".strip()
        previous.raw_text = f"{previous.raw_text.rstrip()}{raw_separator}{segment.raw_text.lstrip()}".strip()
    return merged


def _write_training_artifacts(out_dir: Path, result: object) -> dict[str, str | int]:
    """Export reviewable segment WAVs and Qwen3-ASR JSONL candidates."""
    clips_dir = out_dir / "asr_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript_segments.jsonl"
    training_path = out_dir / "qwen3-asr-candidates.jsonl"
    transcript_lines: list[str] = []
    training_lines: list[str] = []

    training_segments = _merge_training_segments(result.segments)  # type: ignore[attr-defined]
    for index, segment in enumerate(training_segments, 1):
        if not segment.audio:
            continue
        clip_path = clips_dir / f"seg-{index:06d}.wav"
        clip_path.write_bytes(segment.audio)
        raw_text = segment.raw_text or segment.text
        row = {
            "segment_id": f"seg-{index:06d}",
            "start": segment.start,
            "end": segment.end,
            "speaker_id": segment.speaker_id,
            "speaker_name": segment.speaker_name,
            "speaker_confidence": segment.speaker_confidence,
            "raw_text": raw_text,
            "corrected_text": segment.text,
            "correction_source": "asr_or_rules",
            "review_status": "pending",
            "term_candidates": [],
            "audio": str(clip_path.resolve()),
            "text": segment.text,
        }
        transcript_lines.append(json.dumps(row, ensure_ascii=False))
        training_lines.append(
            json.dumps(
                {
                    "audio": row["audio"],
                    "text": row["text"],
                    "segment_id": row["segment_id"],
                    "review_status": row["review_status"],
                },
                ensure_ascii=False,
            )
        )

    transcript_path.write_text("\n".join(transcript_lines) + ("\n" if transcript_lines else ""), encoding="utf-8")
    training_path.write_text("\n".join(training_lines) + ("\n" if training_lines else ""), encoding="utf-8")
    return {
        "clips_dir": str(clips_dir.resolve()),
        "transcript_segments": str(transcript_path.resolve()),
        "training_candidates": str(training_path.resolve()),
        "candidate_segments": len(training_lines),
    }


async def _run(
    audio: Path,
    out_dir: Path,
    topic: str,
    summarize: bool,
    translate: bool,
    separate: bool,
    prompt_template_id: str,
    prompt_templates_path: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_status(out_dir, {"status": "processing", "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    t0 = time.time()
    try:
        prompt_template = get_meeting_prompt_template(
            prompt_template_id,
            custom_templates_path=prompt_templates_path,
        )
        pipeline = _build_pipeline(prompt_templates_path)
        result = await pipeline.process(
            audio,
            filename=audio.name,
            summarize=summarize,
            do_translate=translate,
            topic=topic,
            prompt_template_id=prompt_template.id,
            separate=separate,
            separation_artifact_dir=out_dir / "separated",
        )
    except Exception:
        traceback.print_exc()
        _write_status(out_dir, {
            "status": "failed",
            "error": traceback.format_exc()[-2000:],
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(json.dumps({"status": "failed", "error": "see status.json"}, ensure_ascii=False))
        return 1

    meta = {
        "task_id": out_dir.name,
        "filename": result.filename,
        "topic": topic,
        "prompt_template": prompt_template.id,
        "prompt_template_name": prompt_template.name,
        "llm_model": getattr(getattr(pipeline, "llm", None), "model", "") or "",
        "engine": ENGINE_LABEL,
        "enhancement": {
            "enabled": bool(result.enhancement_metrics),
            "stages": result.enhancement_metrics,
        },
        "separation": {
            "enabled": bool(result.separation_metrics),
            "stages": result.separation_metrics,
            "artifacts": result.separation_artifacts,
        },
        "overlap_detection": result.overlap_detection,
    }
    (out_dir / "report.md").write_text(_render_markdown(result, meta), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(result.timeline_text(), encoding="utf-8")
    artifacts = _write_training_artifacts(out_dir, result)
    (out_dir / "meta.json").write_text(
        json.dumps({"task_id": out_dir.name, **meta, "training_artifacts": artifacts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_status(out_dir, {
        "status": "done",
        "prompt_template": prompt_template.id,
        "prompt_template_name": prompt_template.name,
        "duration_s": round(result.duration_s, 1),
        "segments": len(result.segments),
        "speakers": sorted({s.speaker for s in result.segments if s.speaker}),
        "training_artifacts": artifacts,
        "enhancement": result.enhancement_metrics,
        "separation": result.separation_metrics,
        "separation_artifacts": result.separation_artifacts,
        "overlap_detection": result.overlap_detection,
        "elapsed_s": round(time.time() - t0, 1),
    })
    print(json.dumps({
        "status": "done",
        "task_id": out_dir.name,
        "duration_s": round(result.duration_s, 1),
        "segments": len(result.segments),
        "elapsed_s": round(time.time() - t0, 1),
    }, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Astra 会议 worker(独立进程)")
    parser.add_argument("--audio", required=True, help="输入音频文件(m4a/wav/mp3...)")
    parser.add_argument("--out", required=True, help="输出目录(报告落盘 + status.json)")
    parser.add_argument("--topic", default="", help="会议主题提示")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_MEETING_PROMPT_ID,
        help="纪要系统提示词模板 ID",
    )
    parser.add_argument(
        "--prompt-templates-path",
        default="",
        help="自定义纪要提示词模板 JSON 文件路径",
    )
    parser.add_argument("--no-summarize", action="store_true", help="跳过 LLM 纪要")
    parser.add_argument("--no-translate", action="store_true", help="跳过翻译")
    parser.add_argument(
        "--separate",
        action="store_true",
        help="离线启用 FLASepformer 双说话人分离",
    )
    args = parser.parse_args()
    from app.config import Settings

    prompt_templates_path = args.prompt_templates_path or Settings.from_env().meeting_prompt_templates_path
    return asyncio.run(
        _run(
            Path(args.audio),
            Path(args.out),
            args.topic,
            summarize=not args.no_summarize,
            translate=not args.no_translate,
            separate=args.separate,
            prompt_template_id=args.prompt_template,
            prompt_templates_path=prompt_templates_path,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
