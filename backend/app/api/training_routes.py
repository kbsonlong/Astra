"""训练任务 API。"""
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..core.review_store import ReviewStore
from ..main_types import TrainingConfigPayload

router = APIRouter(prefix="/api/training", tags=["training"])
_TASK_ID_RE = re.compile(r"^\d{8}-\d{6}-[\da-f]{4,8}$")


def _approved_datasets(meeting_dir: Path, review_store: ReviewStore) -> list[dict[str, object]]:
    if not meeting_dir.is_dir():
        return []
    datasets: list[dict[str, object]] = []
    for task_dir in sorted(meeting_dir.iterdir(), reverse=True):
        if not task_dir.is_dir() or not _TASK_ID_RE.match(task_dir.name):
            continue
        dataset_path = task_dir / "qwen3-asr-candidates.jsonl"
        source_path = task_dir / "transcript_segments.jsonl"
        if not source_path.is_file():
            source_path = dataset_path
        if not source_path.is_file():
            continue
        if not review_store.has_task(task_dir.name):
            rows = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            review_store.import_if_empty(task_dir.name, rows)
        review_store.export_jsonl(task_dir.name, dataset_path)
        rows = review_store.export_rows(task_dir.name)
        total = len(rows)
        approved = sum(row.get("review_status") == "approved" for row in rows)
        if approved:
            datasets.append({
                "id": task_dir.name,
                "task_id": task_dir.name,
                "path": str(dataset_path.resolve()),
                "approved_count": approved,
                "total_count": total,
                "updated_at": dataset_path.stat().st_mtime,
            })
    return datasets


@router.get("/datasets")
async def training_datasets(request: Request) -> dict[str, object]:
    meeting_dir = Path(request.app.state.settings.meeting_output_dir).expanduser()
    return {"items": _approved_datasets(meeting_dir, request.app.state.review_store)}


@router.post("/start")
async def start_training(request: Request, payload: TrainingConfigPayload | None = None) -> dict[str, object]:
    manager = request.app.state.training_manager
    try:
        config = payload.to_config() if payload else request.app.state.training_config_loader()
        return await manager.start(config)
    except (RuntimeError, ValueError, OSError) as exc:
        if isinstance(exc, RuntimeError) and "concurrency limit reached" in str(exc):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=409 if isinstance(exc, RuntimeError) else 422, detail=str(exc)) from exc


@router.get("/status")
async def training_status(request: Request) -> dict[str, object]:
    return request.app.state.training_manager.status()


@router.post("/stop")
async def stop_training(request: Request) -> dict[str, object]:
    try:
        return await request.app.state.training_manager.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
