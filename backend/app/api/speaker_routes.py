"""UUID 声纹档案注册 API。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ..core.speaker_registry import (
    ResemblyzerEnrollmentService,
    SpeakerEnrollmentError,
    SpeakerNotFoundError,
    SpeakerProfile,
)
from ..schemas.speaker import SpeakerCreateRequest, SpeakerUpdateRequest

router = APIRouter(prefix="/api/speakers", tags=["speakers"])
ALLOWED_SUFFIX = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".mov", ".mp4"}
MAX_SAMPLE_BYTES = 20 * 1024 * 1024


def _render_profile(profile: SpeakerProfile) -> dict[str, object]:
    return {
        "speaker_id": profile.speaker_id,
        "display_name": profile.display_name,
        "status": profile.status,
        "sample_count": profile.sample_count,
        "embedding_model": profile.embedding_model,
        "updated_at": profile.updated_at,
    }


@router.post("")
async def create_speaker(
    request: Request, payload: SpeakerCreateRequest
) -> dict[str, object]:
    profile = await asyncio.to_thread(
        request.app.state.speaker_store.create, payload.display_name
    )
    return _render_profile(profile)


@router.get("")
async def list_speakers(request: Request) -> dict[str, object]:
    profiles = await asyncio.to_thread(request.app.state.speaker_store.list)
    return {"items": [_render_profile(profile) for profile in profiles]}


@router.get("/{speaker_id}")
async def get_speaker(request: Request, speaker_id: UUID) -> dict[str, object]:
    try:
        profile = await asyncio.to_thread(
            request.app.state.speaker_store.get, str(speaker_id)
        )
    except SpeakerNotFoundError as exc:
        raise HTTPException(status_code=404, detail="speaker not found") from exc
    return _render_profile(profile)


@router.patch("/{speaker_id}")
async def rename_speaker(
    request: Request, speaker_id: UUID, payload: SpeakerUpdateRequest
) -> dict[str, object]:
    try:
        profile = await asyncio.to_thread(
            request.app.state.speaker_store.rename,
            str(speaker_id),
            payload.display_name,
        )
    except SpeakerNotFoundError as exc:
        raise HTTPException(status_code=404, detail="speaker not found") from exc
    return _render_profile(profile)


@router.delete("/{speaker_id}")
async def disable_speaker(request: Request, speaker_id: UUID) -> dict[str, object]:
    try:
        profile = await asyncio.to_thread(
            request.app.state.speaker_store.disable, str(speaker_id)
        )
    except SpeakerNotFoundError as exc:
        raise HTTPException(status_code=404, detail="speaker not found") from exc
    return _render_profile(profile)


@router.post("/{speaker_id}/samples")
async def enroll_speaker_sample(
    request: Request, speaker_id: UUID, file: UploadFile = File(...)
) -> dict[str, object]:
    filename = file.filename or "sample.wav"
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail="unsupported audio type")
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(audio) > MAX_SAMPLE_BYTES:
        raise HTTPException(status_code=413, detail="audio sample is too large")
    service = ResemblyzerEnrollmentService(request.app.state.settings.vad_model)
    try:
        result = await asyncio.to_thread(
            service.enroll,
            audio,
            filename,
            request.app.state.speaker_store,
            str(speaker_id),
        )
    except SpeakerNotFoundError as exc:
        raise HTTPException(status_code=404, detail="speaker not found") from exc
    except SpeakerEnrollmentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "speaker_id": str(speaker_id),
        "sample_id": result.sample_id,
        "status": "ready",
        "duration_s": result.duration_s,
        "speech_duration_s": result.speech_duration_s,
        "quality_score": result.quality_score,
        "embedding_model": "resemblyzer",
        "embedding_dimension": 256,
    }
