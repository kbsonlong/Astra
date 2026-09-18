"""Authenticated download endpoint for completed realtime recordings."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..core.realtime_recording import recording_download_path

router = APIRouter(prefix="/api/realtime-recordings", tags=["realtime-recordings"])


@router.get("/{recording_id}")
async def download_realtime_recording(recording_id: str, request: Request) -> FileResponse:
    try:
        path = recording_download_path(
            request.app.state.settings.realtime_recording_dir, recording_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="recording is not ready")
    return FileResponse(path, media_type="audio/wav", filename=f"astra-{recording_id}.wav")
