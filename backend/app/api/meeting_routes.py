"""会议处理 API —— 独立进程 job 化(2026-09-05 重构)。

背景: 会议整条管线(Qwen3 ASR + 声纹 + LLM 纪要)在 API 进程内跑会独占
事件循环(MLX 必须主线程), 且 nginx /api/ 默认 60s 读超时会掐断长任务。
重构为 fire-and-subscribe:
  POST /api/meeting/process   multipart: file(录音), topic(可选)
                              -> 202 {task_id, status: "processing"}
  WS   /api/meeting/{task_id}/events -> processing | done | failed 事件
处理在独立子进程(meeting_cli.py)执行, 自带 MLX 上下文, API 事件循环
零阻塞; 语音会话不受影响。产物: ~/Astra/meetings/<task_id>/
report.md + transcript.txt + meta.json + status.json + job.log
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from ..config import llm_environment_values
from .auth import authorize_websocket
from .upload_limits import UploadTooLargeError, save_upload_limited
from ..core.meeting_prompts import (
    DEFAULT_MEETING_PROMPT_ID,
    MeetingPromptTemplateStore,
    get_meeting_prompt_template,
    list_meeting_prompt_templates,
)

router = APIRouter(prefix="/api/meeting", tags=["meeting"])

ALLOWED_SUFFIX = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".mov", ".mp4"}

# 实际引擎链(meeting.py 方案A): Silero VAD 出时间戳 -> Qwen3-ASR 逐段转写
# -> resemblyzer 声纹 embed + scipy ward 聚类打标。
ENGINE_LABEL = "Silero VAD + Qwen3-ASR + punctuation + resemblyzer (自动簇数余弦聚类)"

_REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/app/api -> repo 根

# 新格式: 20260905-123456-1a2b3c4d | 旧格式(兼容只读): 20260905-123456-1234
_TASK_ID_RE = re.compile(r"^\d{8}-\d{6}-[\da-f]{4,8}$")
_SEGMENT_ID_RE = re.compile(r"^seg-\d{6}$")


class TrainingReviewPayload(BaseModel):
    review_status: Literal["pending", "approved", "rejected"]
    corrected_text: str = Field(min_length=1, max_length=20_000)


class BatchTrainingReviewItem(BaseModel):
    segment_id: str = Field(min_length=1, max_length=32)
    corrected_text: str = Field(min_length=1, max_length=20_000)


class BatchTrainingReviewPayload(BaseModel):
    review_status: Literal["pending", "approved", "rejected"]
    items: list[BatchTrainingReviewItem] = Field(min_length=1, max_length=500)


class MeetingPromptTemplatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=200)
    chunk_system_prompt: str = Field(min_length=1, max_length=20_000)
    merge_system_prompt: str = Field(min_length=1, max_length=20_000)


def _prompt_store(request: Request) -> MeetingPromptTemplateStore:
    return MeetingPromptTemplateStore(request.app.state.settings.meeting_prompt_templates_path)


def _prompt_item(template: Any) -> dict[str, object]:
    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "chunk_system_prompt": template.chunk_system_prompt,
        "merge_system_prompt": template.merge_system_prompt,
        "builtin": not template.id.startswith("custom-"),
        "editable": template.id.startswith("custom-"),
    }


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
        f"引擎: {meta.get('engine', '-')}  · 纪要模型: {meta.get('llm_model', '-')}  · "
        f"提示词模板: {meta.get('prompt_template_name', meta.get('prompt_template', '-'))}"
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


def _status_payload(
    base: Path,
    task_id: str,
    task_store: object | None = None,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    out_dir = _job_dir(base, task_id)
    status_path = out_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="task not found")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if task_store is not None and status.get("status") == "processing":
        try:
            record = task_store.reconcile_task(  # type: ignore[attr-defined]
                task_id, timeout_seconds=timeout_seconds
            )
        except KeyError:
            record = None
        if record is not None and record.status != "processing":
            status = {**status, "status": record.status, **record.detail}
            status_path.write_text(
                json.dumps(status, ensure_ascii=False), encoding="utf-8"
            )
    result: dict[str, object] = {"type": "meeting_status", "task_id": task_id, **status}
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


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="training review data not found")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    temporary.replace(path)


def _review_rows(out_dir: Path, task_id: str) -> list[dict[str, object]]:
    rows = _jsonl_rows(out_dir / "transcript_segments.jsonl")
    for row in rows:
        segment_id = str(row.get("segment_id", ""))
        row["audio_url"] = f"/api/meeting/{task_id}/training-data/{segment_id}/audio"
    return rows


@router.get("/{task_id}/training-data")
async def training_data(
    task_id: str,
    request: Request,
    page: int = Query(default=1, ge=1, le=100_000),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    base = Path(request.app.state.settings.meeting_output_dir).expanduser()
    out_dir = _job_dir(base, task_id)
    rows = _review_rows(out_dir, task_id)
    counts = {status: sum(row.get("review_status") == status for row in rows) for status in ("pending", "approved", "rejected")}
    total = len(rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    current_page = min(page, total_pages)
    start = (current_page - 1) * page_size
    return {
        "task_id": task_id,
        "items": rows[start : start + page_size],
        "counts": counts,
        "pagination": {
            "page": current_page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": current_page < total_pages,
            "has_prev": current_page > 1,
        },
    }


@router.get("/prompt-templates")
async def meeting_prompt_templates(request: Request) -> dict[str, object]:
    return {
        "default": DEFAULT_MEETING_PROMPT_ID,
        "items": list_meeting_prompt_templates(
            request.app.state.settings.meeting_prompt_templates_path
        ),
    }


@router.post("/prompt-templates")
async def create_meeting_prompt_template(
    payload: MeetingPromptTemplatePayload,
    request: Request,
) -> dict[str, object]:
    try:
        template = _prompt_store(request).create(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"item": _prompt_item(template)}


@router.put("/prompt-templates/{template_id}")
async def update_meeting_prompt_template(
    template_id: str,
    payload: MeetingPromptTemplatePayload,
    request: Request,
) -> dict[str, object]:
    try:
        template = _prompt_store(request).update(
            template_id, **payload.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"item": _prompt_item(template)}


@router.delete("/prompt-templates/{template_id}")
async def delete_meeting_prompt_template(
    template_id: str,
    request: Request,
) -> dict[str, str]:
    try:
        _prompt_store(request).delete(template_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": template_id, "status": "deleted"}


@router.post("/{task_id}/training-data/batch-review")
async def batch_review_training_data(
    task_id: str,
    payload: BatchTrainingReviewPayload,
    request: Request,
) -> dict[str, object]:
    segment_ids = [item.segment_id for item in payload.items]
    if any(not _SEGMENT_ID_RE.match(segment_id) for segment_id in segment_ids):
        raise HTTPException(status_code=400, detail="invalid segment_id format")
    if len(set(segment_ids)) != len(segment_ids):
        raise HTTPException(status_code=400, detail="duplicate segment_id")

    base = Path(request.app.state.settings.meeting_output_dir).expanduser()
    out_dir = _job_dir(base, task_id)
    rows = _jsonl_rows(out_dir / "transcript_segments.jsonl")
    rows_by_id = {str(row.get("segment_id", "")): row for row in rows}
    missing = [segment_id for segment_id in segment_ids if segment_id not in rows_by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"segment not found: {missing[0]}")

    for item in payload.items:
        corrected_text = item.corrected_text.strip()
        if not corrected_text:
            raise HTTPException(status_code=400, detail="corrected_text cannot be empty")
        row = rows_by_id[item.segment_id]
        row["corrected_text"] = corrected_text
        row["text"] = corrected_text
        row["review_status"] = payload.review_status
        row["correction_source"] = "human"
    _write_jsonl(out_dir / "transcript_segments.jsonl", rows)

    candidate_path = out_dir / "qwen3-asr-candidates.jsonl"
    if candidate_path.is_file():
        candidates = _jsonl_rows(candidate_path)
        candidates_by_id = {str(row.get("segment_id", "")): row for row in candidates}
        for item in payload.items:
            candidate = candidates_by_id.get(item.segment_id)
            if candidate is not None:
                candidate["text"] = item.corrected_text.strip()
                candidate["review_status"] = payload.review_status
        _write_jsonl(candidate_path, candidates)

    counts = {
        status: sum(row.get("review_status") == status for row in rows)
        for status in ("pending", "approved", "rejected")
    }
    return {
        "task_id": task_id,
        "updated": segment_ids,
        "updated_count": len(segment_ids),
        "counts": counts,
    }


@router.patch("/{task_id}/training-data/{segment_id}")
async def review_training_data(
    task_id: str,
    segment_id: str,
    payload: TrainingReviewPayload,
    request: Request,
) -> dict[str, object]:
    if not _SEGMENT_ID_RE.match(segment_id):
        raise HTTPException(status_code=400, detail="invalid segment_id format")
    base = Path(request.app.state.settings.meeting_output_dir).expanduser()
    out_dir = _job_dir(base, task_id)
    rows = _jsonl_rows(out_dir / "transcript_segments.jsonl")
    row = next((item for item in rows if item.get("segment_id") == segment_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="segment not found")
    row["corrected_text"] = payload.corrected_text.strip()
    row["text"] = payload.corrected_text.strip()
    row["review_status"] = payload.review_status
    row["correction_source"] = "human"
    _write_jsonl(out_dir / "transcript_segments.jsonl", rows)

    candidates = _jsonl_rows(out_dir / "qwen3-asr-candidates.jsonl")
    candidate = next((item for item in candidates if item.get("segment_id") == segment_id), None)
    if candidate is not None:
        candidate["text"] = payload.corrected_text.strip()
        candidate["review_status"] = payload.review_status
        _write_jsonl(out_dir / "qwen3-asr-candidates.jsonl", candidates)
    return {"task_id": task_id, "item": {**row, "audio_url": f"/api/meeting/{task_id}/training-data/{segment_id}/audio"}}


@router.get("/{task_id}/training-data/{segment_id}/audio")
async def training_segment_audio(task_id: str, segment_id: str, request: Request) -> FileResponse:
    if not _SEGMENT_ID_RE.match(segment_id):
        raise HTTPException(status_code=400, detail="invalid segment_id format")
    base = Path(request.app.state.settings.meeting_output_dir).expanduser()
    out_dir = _job_dir(base, task_id)
    audio_path = (out_dir / "asr_clips" / f"{segment_id}.wav").resolve()
    if out_dir.resolve() not in audio_path.parents or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="segment audio not found")
    return FileResponse(audio_path, media_type="audio/wav", filename=f"{segment_id}.wav")


@router.post("/process")
async def process_meeting(
    request: Request,
    file: UploadFile = File(...),
    topic: str = Form(default=""),
    prompt_template: str = Form(default=DEFAULT_MEETING_PROMPT_ID),
) -> dict[str, object]:
    settings = request.app.state.settings
    base = Path(settings.meeting_output_dir).expanduser()
    try:
        selected_template = get_meeting_prompt_template(
            prompt_template,
            custom_templates_path=settings.meeting_prompt_templates_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = file.filename or "meeting.m4a"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail=f"unsupported audio type: {suffix}")

    # 落盘输入 + 建 job 目录, 立刻返回 task_id(处理在独立进程, 见 meeting_cli)。
    # 会议录音可达数百 MiB，必须分块写入，不能整体驻留在 API 进程内存中。
    task_id = _new_task_id()
    out_dir = base / task_id
    out_dir.mkdir(parents=True, exist_ok=False)
    in_path = out_dir / f"input{suffix}"
    try:
        uploaded_bytes = await save_upload_limited(
            file, in_path, settings.meeting_max_upload_bytes
        )
    except UploadTooLargeError as exc:
        out_dir.rmdir()
        raise HTTPException(
            status_code=413,
            detail=f"file too large (max {exc.max_bytes} bytes)",
        ) from exc
    if not uploaded_bytes:
        in_path.unlink(missing_ok=True)
        out_dir.rmdir()
        raise HTTPException(status_code=400, detail="file is empty")
    (out_dir / "status.json").write_text(
        json.dumps(
            {"status": "processing", "started": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    task_store = request.app.state.task_store
    log_path = out_dir / "job.log"
    if not task_store.reserve(
        task_id=task_id,
        kind="meeting",
        max_concurrent=settings.meeting_max_concurrent_jobs,
        output_dir=str(out_dir),
        log_path=str(log_path),
        detail={"filename": filename, "prompt_template": selected_template.id},
    ):
        shutil.rmtree(out_dir)
        raise HTTPException(status_code=429, detail="meeting concurrency limit reached")

    argv = [
        sys.executable, "-m", "app.core.meeting_cli",
        "--audio", str(in_path),
        "--out", str(out_dir),
    ]
    if topic:
        argv += ["--topic", topic]
    argv += [
        "--prompt-template", selected_template.id,
        "--prompt-templates-path", settings.meeting_prompt_templates_path,
    ]
    env = dict(os.environ)
    env.update(llm_environment_values(settings))
    env["PYTHONPATH"] = str(_REPO_ROOT / "backend")
    try:
        with log_path.open("wb") as log:
            # start_new_session: 脱离进程组, uvicorn 退出/重启不杀会议任务
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=str(_REPO_ROOT), env=env,
                stdout=log, stderr=log, start_new_session=True,
            )
        pid = getattr(process, "pid", None)
        try:
            pgid = os.getpgid(pid) if pid is not None else None
        except OSError:
            pgid = None
        request.app.state.task_store.register(
            task_id=task_id,
            kind="meeting",
            status="processing",
            pid=pid,
            pgid=pgid,
            output_dir=str(out_dir),
            log_path=str(log_path),
            detail={"filename": filename, "prompt_template": selected_template.id},
        )
    except Exception as exc:
        failure = {"status": "failed", "error": f"spawn failed: {exc}"}
        (out_dir / "status.json").write_text(
            json.dumps(failure), encoding="utf-8"
        )
        request.app.state.task_store.register(
            task_id=task_id,
            kind="meeting",
            status="failed",
            pid=None,
            pgid=None,
            output_dir=str(out_dir),
            log_path=str(log_path),
            detail=failure,
        )
        raise HTTPException(status_code=500, detail=f"failed to start job: {exc}") from exc

    return {
        "task_id": task_id,
        "filename": filename,
        "status": "processing",
        "prompt_template": selected_template.id,
        "report_path": str(out_dir / "report.md"),
        "note": "处理在独立进程执行; 用 WebSocket /api/meeting/{task_id}/events 订阅状态",
    }


@router.delete("/{task_id}")
async def cancel_meeting(task_id: str, request: Request) -> dict[str, object]:
    base = Path(request.app.state.settings.meeting_output_dir).expanduser()
    out_dir = _job_dir(base, task_id)
    status_path = out_dir / "status.json"
    if not status_path.is_file():
        raise HTTPException(status_code=404, detail="task not found")
    try:
        record = request.app.state.task_store.terminate(
            task_id, reason="cancelled by administrator"
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    status.update({"status": record.status, **record.detail})
    status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    return {"task_id": task_id, "status": record.status, **record.detail}


@router.websocket("/{task_id}/events")
async def meeting_events(websocket: WebSocket, task_id: str) -> None:
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()
    settings = websocket.app.state.settings
    base = Path(settings.meeting_output_dir).expanduser()
    try:
        while True:
            try:
                payload = _status_payload(
                    base,
                    task_id,
                    websocket.app.state.task_store,
                    timeout_seconds=settings.meeting_task_timeout_seconds,
                )
            except HTTPException as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "task_id": task_id,
                        "code": "meeting_status_unavailable",
                        "message": exc.detail,
                    }
                )
                await websocket.close(code=1008)
                return
            await websocket.send_json(payload)
            if payload.get("status") in {"done", "failed", "stopped"}:
                await websocket.close(code=1000)
                return
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return


def _elapsed(started: str) -> int:
    try:
        return int(time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S")))
    except (ValueError, OSError):
        return 0
