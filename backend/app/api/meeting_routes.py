"""会议处理 HTTP API —— 独立进程 job 化(2026-09-05 重构)。

背景: 会议整条管线(Qwen3 ASR + 声纹 + LLM 纪要)在 API 进程内跑会独占
事件循环(MLX 必须主线程), 且 nginx /api/ 默认 60s 读超时会掐断长任务。
重构为 fire-and-poll:
  POST /api/meeting/process   multipart: file(录音), topic(可选)
                              -> 202 {task_id, status: "processing"}
  GET  /api/meeting/{task_id} -> status: processing | done | failed + 摘要
处理在独立子进程(meeting_cli.py)执行, 自带 MLX 上下文, API 事件循环
零阻塞; 语音会话不受影响。产物: ~/Astra/meetings/<task_id>/
report.md + transcript.txt + meta.json + status.json + job.log
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api/meeting", tags=["meeting"])

ALLOWED_SUFFIX = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".mov", ".mp4"}

# 实际引擎链(meeting.py 方案A): Silero VAD 出时间戳 -> Qwen3-ASR 逐段转写
# -> resemblyzer 声纹 embed + scipy ward 聚类打标。勿写回已弃用的 whisper。
ENGINE_LABEL = "Silero VAD + Qwen3-ASR + resemblyzer (ward 聚类)"

_REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/app/api -> repo 根

# 新格式: 20260905-123456-1a2b3c4d | 旧格式(兼容只读): 20260905-123456-1234
_TASK_ID_RE = re.compile(r"^\d{8}-\d{6}-[\da-f]{4,8}$")


def _new_task_id() -> str:
    # uuid 后缀: 避免同秒同文件名时 hash()%10000 撞车覆盖上一次报告
    # (hash() 对 str 跨进程随机, %10000 只有万分之一空间, 双碰撞风险)。
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


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
    out.append(
        f"- **生成**: {time.strftime('%Y-%m-%d %H:%M')}  · "
        f"引擎: {meta.get('engine', '-')}  · 纪要模型: {meta.get('llm_model', '-')}"
    )
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


def _job_dir(base: Path, task_id: str) -> Path:
    """按 task_id 解析输出目录; 格式校验防路径穿越。"""
    if not _TASK_ID_RE.match(task_id):
        raise HTTPException(status_code=400, detail="invalid task_id format")
    out_dir = (base / task_id).resolve()
    base_resolved = base.resolve()
    if base_resolved not in out_dir.parents:
        raise HTTPException(status_code=400, detail="invalid task_id")
    return out_dir


@router.post("/process")
async def process_meeting(
    request: Request,
    file: UploadFile = File(...),
    topic: str = Form(default=""),
) -> dict[str, object]:
    settings = request.app.state.settings
    base = Path(settings.meeting_output_dir).expanduser()

    filename = file.filename or "meeting.m4a"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail=f"unsupported audio type: {suffix}")

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="file is empty")
    if len(audio) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large (max 500MB)")

    # 落盘输入 + 建 job 目录, 立刻返回 task_id(处理在独立进程, 见 meeting_cli)
    task_id = _new_task_id()
    out_dir = base / task_id
    out_dir.mkdir(parents=True, exist_ok=False)
    in_path = out_dir / f"input{suffix}"
    in_path.write_bytes(audio)
    (out_dir / "status.json").write_text(
        json.dumps(
            {"status": "processing", "started": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    argv = [
        sys.executable, "-m", "app.core.meeting_cli",
        "--audio", str(in_path),
        "--out", str(out_dir),
    ]
    if topic:
        argv += ["--topic", topic]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "backend")
    log_path = out_dir / "job.log"
    try:
        with log_path.open("wb") as log:
            # start_new_session: 脱离进程组, uvicorn 退出/重启不杀会议任务
            await asyncio.create_subprocess_exec(
                *argv, cwd=str(_REPO_ROOT), env=env,
                stdout=log, stderr=log, start_new_session=True,
            )
    except Exception as exc:
        (out_dir / "status.json").write_text(
            json.dumps({"status": "failed", "error": f"spawn failed: {exc}"}),
            encoding="utf-8",
        )
        raise HTTPException(status_code=500, detail=f"failed to start job: {exc}") from exc

    return {
        "task_id": task_id,
        "filename": filename,
        "status": "processing",
        "report_path": str(out_dir / "report.md"),
        "note": "处理在独立进程执行; 用 GET /api/meeting/{task_id} 轮询",
    }


@router.get("/{task_id}")
async def meeting_status(request: Request, task_id: str) -> dict[str, object]:
    settings = request.app.state.settings
    base = Path(settings.meeting_output_dir).expanduser()
    out_dir = _job_dir(base, task_id)
    status_path = out_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="task not found")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    result: dict[str, object] = {"task_id": task_id, **status}
    report = out_dir / "report.md"
    if status.get("status") == "done" and report.exists():
        transcript = out_dir / "transcript.txt"
        result["report_path"] = str(report)
        result["summary_preview"] = report.read_text(encoding="utf-8")[:2000]
        result["transcript_preview"] = "\n".join(
            transcript.read_text(encoding="utf-8").split("\n")[:10]
        )
    elif status.get("status") == "processing":
        result["elapsed_s"] = _elapsed(status.get("started", ""))
    return result


def _elapsed(started: str) -> int:
    try:
        return int(time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S")))
    except (ValueError, OSError):
        return 0
