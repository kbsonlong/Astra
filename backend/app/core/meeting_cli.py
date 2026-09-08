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
      [--topic T] [--no-summarize] [--no-translate]

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
from app.core.workflow import ResemblyzerDiarizationStage  # noqa: E402
from app.models.punctuation_client import build_punctuation_client  # noqa: E402


def _build_pipeline():
    """与 main.py 中会议管线一致的干净配置(无热词/无 system_prompt)。"""
    from app.config import Settings
    from app.core.meeting import MeetingPipeline
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
    )
    punctuation = build_punctuation_client(
        enabled=s.punctuation_enabled,
        engine=s.punctuation_engine,
        model=s.punctuation_model,
        device=s.punctuation_device,
    )
    sd_engine = s.sd_engine.lower()
    if sd_engine in {"", "none", "noop"}:
        diarization = None
    elif sd_engine in {"resemblyzer", "resemblyzer-ward"}:
        diarization = ResemblyzerDiarizationStage()
    else:
        raise ValueError(f"unsupported SD engine: {s.sd_engine}")
    return MeetingPipeline(
        llm=llm,
        asr=meeting_asr,
        vad_model=s.vad_model,
        punctuation=punctuation,
        diarization=diarization,
    )


def _write_status(out_dir: Path, payload: dict) -> None:
    (out_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


async def _run(audio: Path, out_dir: Path, topic: str, summarize: bool, translate: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_status(out_dir, {"status": "processing", "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    t0 = time.time()
    try:
        pipeline = _build_pipeline()
        result = await pipeline.process(
            audio,
            filename=audio.name,
            summarize=summarize,
            do_translate=translate,
            topic=topic,
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
        "llm_model": getattr(getattr(pipeline, "llm", None), "model", "") or "",
        "engine": ENGINE_LABEL,
    }
    (out_dir / "report.md").write_text(_render_markdown(result, meta), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(result.timeline_text(), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"task_id": out_dir.name, **meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_status(out_dir, {
        "status": "done",
        "duration_s": round(result.duration_s, 1),
        "segments": len(result.segments),
        "speakers": sorted({s.speaker for s in result.segments if s.speaker}),
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
    parser.add_argument("--no-summarize", action="store_true", help="跳过 LLM 纪要")
    parser.add_argument("--no-translate", action="store_true", help="跳过翻译")
    args = parser.parse_args()
    return asyncio.run(
        _run(
            Path(args.audio),
            Path(args.out),
            args.topic,
            summarize=not args.no_summarize,
            translate=not args.no_translate,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
