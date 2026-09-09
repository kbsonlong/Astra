"""训练任务 API。"""
from fastapi import APIRouter, HTTPException, Request

from ..main_types import TrainingConfigPayload

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start")
async def start_training(request: Request, payload: TrainingConfigPayload | None = None) -> dict[str, object]:
    manager = request.app.state.training_manager
    try:
        config = payload.to_config() if payload else request.app.state.training_config_loader()
        return await manager.start(config)
    except (RuntimeError, ValueError, OSError) as exc:
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
