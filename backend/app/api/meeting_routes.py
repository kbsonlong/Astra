"""会议处理 HTTP API: 上传录音 -> 转写+分离+纪要 报告落盘。

POST /api/meeting/process   multipart: file(录音), topic(可选会议主题)
                            -> {task_id}  (同步处理, 返回完整结果)
产出目录: ~/Astra/meetings/<task_id>/  report.md + transcript.txt
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api/meeting", tags=["meeting"])

ALLOWED_SUFFIX = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".mov", ".mp4"}


def _render_markdown(result: Any, meta: dict[str, object]) -> str:
    """将 MeetingResult 渲染为可交付 markdown 报告。"""
    out: list[str] = []
    out.append("# 会议纪要")
    out.append("")
    dur = result.duration_s
    out.append(
        f"- **录音**: {result.filename}  · 时长 {int(dur//60)}分{int(dur%60)}秒  "
        f"· 语言 {result.language}  · 说话人 {len({s.speaker for s in result.segments if s.speaker}) or '未分离'}"
    )
    if meta.get("topic"):
        out.append(f"- **主题**: {meta['topic']}")
    out.append(f"- **生成**: {time.strftime('%Y-%m-%d %H:%M')}  · 引擎: whisper-large-v3-turbo + resemblyzer + {meta.get('llm_model', '-')}")
    out.append("")

    if result.summary:
        out.append("---")
        out.append("")
        out.append("## 🤖 AI 摘要")
        out.append("")
        out.append(result.summary)
        out.append("")

    if result.translation:
        out.append("---")
        out.append("")
        out.append("## 🌐 翻译")
        out.append("")
        out.append(result.translation)
        out.append("")

    out.append("---")
    out.append("")
    out.append("## 🎙️ 逐字稿（按说话人标注）")
    out.append("")
    out.append("```text")
    out.append(result.timeline_text())
    out.append("```")
    out.append("")
    return "\n".join(out)


@router.post("/process")
async def process_meeting(
    request: Request,
    file: UploadFile = File(...),
    topic: str = Form(default=""),
) -> dict[str, object]:
    pipeline = getattr(request.app.state, "meeting_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="meeting pipeline is not configured")

    filename = file.filename or "meeting.m4a"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail=f"unsupported audio type: {suffix}")

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="file is empty")
    if len(audio) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large (max 500MB)")

    try:
        result = await pipeline.process(audio, filename=filename, topic=topic)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"meeting processing failed: {exc}") from exc

    # 落盘: report.md + transcript.txt + meta.json
    settings = request.app.state.settings
    base = Path(settings.meeting_output_dir).expanduser()
    task_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{abs(hash(filename)) % 10000:04d}"
    out_dir = base / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, object] = {
        "topic": topic,
        "llm_model": settings.llm_model,
        "engine": "mlx-whisper-large-v3-turbo + resemblyzer + spectralcluster",
    }
    markdown = _render_markdown(result, meta)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    (out_dir / "transcript.txt").write_text(result.timeline_text(), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"task_id": task_id, "filename": filename, **meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "task_id": task_id,
        "filename": filename,
        "duration_s": round(result.duration_s, 1),
        "language": result.language,
        "segments": len(result.segments),
        "speakers": sorted({s.speaker for s in result.segments if s.speaker}),
        "report_path": str(out_dir / "report.md"),
        "summary": result.summary[:2000],
        "timeline_preview": "\n".join(result.timeline_text().split("\n")[:10]),
    }
