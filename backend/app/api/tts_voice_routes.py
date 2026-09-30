"""CRUD API for local TTS reference voices."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import uuid
import wave
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..core.tts_voice_store import (
    TtsVoice,
    TtsVoiceConflictError,
    TtsVoiceError,
    TtsVoiceNotFoundError,
    TtsVoicePathError,
    TtsVoiceRevision,
    TtsVoiceRevisionNotFoundError,
    TtsVoiceStore,
    TtsVoiceValidationError,
)
from ..schemas.tts_voice import (
    CONSENT_CONFIRMATION,
    ConsentRequest,
    ConsentRevokeRequest,
    VoiceMetadataPatch,
)
from .upload_limits import UploadTooLargeError, read_upload_limited


router = APIRouter(prefix="/api/tts/voices", tags=["tts-voices"])
ALLOWED_SUFFIX = {".wav"}


def _store(request: Request) -> TtsVoiceStore:
    return request.app.state.tts_voice_store


def _raise_store_http(exc: Exception) -> None:
    if isinstance(exc, (TtsVoiceNotFoundError, TtsVoiceRevisionNotFoundError)):
        raise HTTPException(status_code=404, detail="TTS voice or revision not found") from exc
    if isinstance(exc, TtsVoiceConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (TtsVoicePathError, TtsVoiceValidationError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, TtsVoiceError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise exc


def _render_revision(revision: TtsVoiceRevision, voice_id: str | None = None) -> dict[str, object]:
    voice_id = voice_id or revision.voice_id
    return {
        "revision": revision.revision,
        "lifecycle": revision.lifecycle,
        "backend_family": revision.backend_family,
        "language": revision.language,
        "duration_s": revision.duration_s,
        "speech_duration_s": revision.speech_duration_s,
        "sample_rate": revision.sample_rate,
        "channels": revision.channels,
        "sample_width_bytes": revision.sample_width_bytes,
        "quality_score": revision.quality_score,
        "default_params": revision.default_params,
        "consent_status": revision.consent_status,
        "consent_confirmed_at": revision.consent_confirmed_at,
        "original_filename": revision.original_filename,
        "reference_sha256": revision.reference_sha256,
        "created_at": revision.created_at,
        "audio_url": f"/api/tts/voices/{voice_id}/revisions/{revision.revision}/audio",
    }


def _render_voice(voice: TtsVoice) -> dict[str, object]:
    return {
        "voice_id": voice.voice_id,
        "display_name": voice.display_name,
        "status": voice.status,
        "current_revision": voice.current_revision,
        "created_at": voice.created_at,
        "updated_at": voice.updated_at,
        "archived_at": voice.archived_at,
        "revoked_at": voice.revoked_at,
        "revision": _render_revision(voice.revision) if voice.revision else None,
    }


def _parse_default_params(raw: str) -> dict[str, object]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="default_params must be valid JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="default_params must be a JSON object")
    return value


def _canonicalize_wav(payload: bytes) -> tuple[bytes, dict[str, object]]:
    """Validate and rewrite a bounded mono PCM16 WAV without external decoders."""
    try:
        source = wave.open(io.BytesIO(payload), "rb")
    except (EOFError, wave.Error) as exc:
        raise HTTPException(status_code=422, detail="reference audio is not a valid WAV") from exc
    with source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        if channels != 1 or sample_width != 2 or sample_rate <= 0 or frame_count <= 0:
            raise HTTPException(
                status_code=422,
                detail="reference WAV must be non-empty mono PCM16 with a valid sample rate",
            )
        frames = source.readframes(frame_count)
    canonical = io.BytesIO()
    with wave.open(canonical, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)
    return canonical.getvalue(), {
        "duration_s": frame_count / sample_rate,
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_width_bytes": 2,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


async def _read_reference_upload(
    file: UploadFile, *, max_bytes: int, min_seconds: float, max_seconds: float
) -> tuple[bytes, bytes, dict[str, object], str]:
    filename = file.filename or "reference.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=415, detail="only WAV reference audio is supported")
    try:
        original = await read_upload_limited(file, max_bytes)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail="reference audio is too large") from exc
    if not original:
        raise HTTPException(status_code=400, detail="reference audio is empty")
    canonical, metadata = _canonicalize_wav(original)
    duration = float(metadata["duration_s"])
    if duration < min_seconds or duration > max_seconds:
        raise HTTPException(
            status_code=422,
            detail=f"reference audio duration must be between {min_seconds:g} and {max_seconds:g} seconds",
        )
    return original, canonical, metadata, Path(filename).name


def _asset_paths(audio_dir: Path, *, token: str) -> tuple[Path, str, str]:
    relative_dir = Path("uploads")
    original_relative = relative_dir / f"{token}-original.wav"
    reference_relative = relative_dir / f"{token}-reference.wav"
    return (
        audio_dir / original_relative,
        original_relative.as_posix(),
        reference_relative.as_posix(),
    )


def _write_assets(audio_dir: Path, token: str, original: bytes, canonical: bytes) -> tuple[str, str]:
    original_path, original_relative, reference_relative = _asset_paths(audio_dir, token=token)
    reference_path = audio_dir / reference_relative
    _atomic_write(original_path, original)
    try:
        _atomic_write(reference_path, canonical)
    except Exception:
        original_path.unlink(missing_ok=True)
        raise
    return original_relative, reference_relative


def _cleanup_assets(audio_dir: Path, *relative_paths: str) -> None:
    for relative in relative_paths:
        path = audio_dir / relative
        try:
            if path.is_file() and not path.is_symlink():
                path.unlink()
        except OSError:
            pass


def _duration_limits(request: Request) -> tuple[int, float, float, Path]:
    settings = request.app.state.settings
    return (
        settings.tts_voice_max_upload_bytes,
        settings.tts_voice_min_reference_seconds,
        settings.tts_voice_max_reference_seconds,
        Path(settings.tts_voice_audio_dir).expanduser(),
    )


@router.post("")
async def create_voice(
    request: Request,
    name: str = Form(..., min_length=1, max_length=128),
    language: str = Form("zh"),
    backend_family: str = Form("indextts25_mlx"),
    default_params: str = Form("{}"),
    file: UploadFile = File(...),
) -> JSONResponse:
    original, canonical, metadata, filename = await _read_reference_upload(
        file,
        max_bytes=_duration_limits(request)[0],
        min_seconds=_duration_limits(request)[1],
        max_seconds=_duration_limits(request)[2],
    )
    params = _parse_default_params(default_params)
    audio_dir = _duration_limits(request)[3]
    token = uuid.uuid4().hex
    original_relative, reference_relative = _write_assets(audio_dir, token, original, canonical)
    try:
        voice = await asyncio.to_thread(
            _store(request).create_draft,
            name,
            backend_family=backend_family,
            language=language,
            reference_sha256=hashlib.sha256(canonical).hexdigest(),
            original_sha256=hashlib.sha256(original).hexdigest(),
            original_filename=filename,
            original_relpath=original_relative,
            reference_relpath=reference_relative,
            duration_s=metadata["duration_s"],
            speech_duration_s=metadata["duration_s"],
            sample_rate=metadata["sample_rate"],
            channels=metadata["channels"],
            sample_width_bytes=metadata["sample_width_bytes"],
            default_params=params,
        )
    except Exception as exc:
        _cleanup_assets(audio_dir, original_relative, reference_relative)
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return JSONResponse(status_code=201, content=_render_voice(voice))


@router.get("")
async def list_voices(
    request: Request,
    status: str | None = Query(None),
    language: str | None = Query(None),
    backend_family: str | None = Query(None),
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int | None = Query(None, ge=1, le=200),
) -> dict[str, object]:
    effective_page_size = page_size or request.app.state.settings.tts_voice_default_page_size
    try:
        voices, pagination = await asyncio.to_thread(
            _store(request).list,
            status=status,
            language=language,
            backend_family=backend_family,
            query=q,
            page=page,
            page_size=effective_page_size,
        )
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return {"items": [_render_voice(voice) for voice in voices], "pagination": pagination}


@router.get("/{voice_id}")
async def get_voice(request: Request, voice_id: UUID) -> dict[str, object]:
    try:
        voice = await asyncio.to_thread(_store(request).get, str(voice_id))
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_voice(voice)


@router.patch("/{voice_id}")
async def update_voice(
    request: Request, voice_id: UUID, payload: VoiceMetadataPatch
) -> dict[str, object]:
    try:
        voice = await asyncio.to_thread(
            _store(request).update_metadata, str(voice_id), payload.display_name
        )
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_voice(voice)


@router.get("/{voice_id}/revisions")
async def list_revisions(request: Request, voice_id: UUID) -> dict[str, object]:
    try:
        revisions = await asyncio.to_thread(_store(request).list_revisions, str(voice_id))
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return {"items": [_render_revision(item, str(voice_id)) for item in revisions]}


@router.get("/{voice_id}/revisions/{revision}")
async def get_revision(
    request: Request, voice_id: UUID, revision: int
) -> dict[str, object]:
    try:
        result = await asyncio.to_thread(
            _store(request).get_revision, str(voice_id), revision
        )
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_revision(result, str(voice_id))


@router.post("/{voice_id}/revisions")
async def add_voice_revision(
    request: Request,
    voice_id: UUID,
    language: str = Form("zh"),
    backend_family: str = Form("indextts25_mlx"),
    default_params: str = Form("{}"),
    file: UploadFile = File(...),
) -> JSONResponse:
    original, canonical, metadata, filename = await _read_reference_upload(
        file,
        max_bytes=_duration_limits(request)[0],
        min_seconds=_duration_limits(request)[1],
        max_seconds=_duration_limits(request)[2],
    )
    params = _parse_default_params(default_params)
    audio_dir = _duration_limits(request)[3]
    token = uuid.uuid4().hex
    original_relative, reference_relative = _write_assets(audio_dir, token, original, canonical)
    try:
        revision = await asyncio.to_thread(
            _store(request).add_revision,
            str(voice_id),
            backend_family=backend_family,
            language=language,
            reference_sha256=hashlib.sha256(canonical).hexdigest(),
            original_sha256=hashlib.sha256(original).hexdigest(),
            original_filename=filename,
            original_relpath=original_relative,
            reference_relpath=reference_relative,
            duration_s=metadata["duration_s"],
            speech_duration_s=metadata["duration_s"],
            sample_rate=metadata["sample_rate"],
            channels=metadata["channels"],
            sample_width_bytes=metadata["sample_width_bytes"],
            default_params=params,
        )
    except Exception as exc:
        _cleanup_assets(audio_dir, original_relative, reference_relative)
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return JSONResponse(
        status_code=201,
        content=_render_revision(revision, str(voice_id)),
    )


@router.post("/{voice_id}/revisions/{revision}/consent")
async def confirm_consent(
    request: Request,
    voice_id: UUID,
    revision: int,
    payload: ConsentRequest,
) -> dict[str, object]:
    if payload.confirmation != CONSENT_CONFIRMATION:
        raise HTTPException(status_code=422, detail="explicit consent confirmation is required")
    try:
        result = await asyncio.to_thread(
            _store(request).confirm_consent,
            str(voice_id),
            revision,
            note=payload.note,
        )
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_revision(result, str(voice_id))


@router.post("/{voice_id}/revisions/{revision}/revoke-consent")
async def revoke_consent(
    request: Request,
    voice_id: UUID,
    revision: int,
    payload: ConsentRevokeRequest,
) -> dict[str, object]:
    try:
        result = await asyncio.to_thread(
            _store(request).revoke_consent,
            str(voice_id),
            revision,
            note=payload.note,
        )
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_revision(result, str(voice_id))


@router.post("/{voice_id}/activate")
async def activate_voice(
    request: Request, voice_id: UUID, revision: int | None = Query(None, ge=1)
) -> dict[str, object]:
    try:
        voice = await asyncio.to_thread(_store(request).activate, str(voice_id), revision)
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_voice(voice)


@router.post("/{voice_id}/archive")
async def archive_voice(request: Request, voice_id: UUID) -> dict[str, object]:
    try:
        voice = await asyncio.to_thread(_store(request).archive, str(voice_id))
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_voice(voice)


@router.post("/{voice_id}/restore")
async def restore_voice(request: Request, voice_id: UUID) -> dict[str, object]:
    try:
        voice = await asyncio.to_thread(_store(request).restore, str(voice_id))
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return _render_voice(voice)


@router.get("/{voice_id}/revisions/{revision}/audio")
async def get_voice_audio(
    request: Request, voice_id: UUID, revision: int
) -> FileResponse:
    try:
        path = await asyncio.to_thread(
            _store(request).resolve_audio_path, str(voice_id), revision, "reference"
        )
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@router.get("/{voice_id}/audio")
async def get_current_voice_audio(
    request: Request, voice_id: UUID
) -> FileResponse:
    try:
        voice = await asyncio.to_thread(_store(request).get, str(voice_id))
        revision = voice.current_revision
        if revision is None and voice.revision is not None:
            revision = voice.revision.revision
        if revision is None:
            raise TtsVoiceRevisionNotFoundError(str(voice_id))
        path = await asyncio.to_thread(
            _store(request).resolve_audio_path, str(voice_id), revision, "reference"
        )
    except Exception as exc:
        _raise_store_http(exc)
        raise AssertionError("unreachable")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
