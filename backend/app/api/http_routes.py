import asyncio
import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..core.meeting import MeetingPipeline
from ..models.asr_client import ASRClientError
from ..models.llm_client import LLMClientError
from .upload_limits import (
    AudioIPConcurrencyLimiter,
    AudioTooLongError,
    UploadTooLargeError,
    read_upload_limited,
    validate_audio_duration,
)

router = APIRouter(prefix="/api")


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _too_large_response(max_bytes: int) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"audio file too large (max {max_bytes} bytes)",
    )


def _client_ip(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


async def _acquire_audio_slot(request: Request) -> tuple[AudioIPConcurrencyLimiter, str]:
    limiter: AudioIPConcurrencyLimiter = request.app.state.audio_ip_limiter
    ip = _client_ip(request)
    if not await limiter.try_acquire(ip):
        raise HTTPException(status_code=429, detail="audio concurrency limit reached")
    return limiter, ip


async def _read_transcribe_upload(request: Request, file: UploadFile) -> bytes:
    try:
        return await read_upload_limited(
            file, request.app.state.settings.transcribe_max_upload_bytes
        )
    except UploadTooLargeError as exc:
        raise _too_large_response(exc.max_bytes) from exc


async def _validate_transcribe_duration(
    request: Request, audio: bytes, filename: str
) -> None:
    try:
        await validate_audio_duration(
            audio,
            filename,
            request.app.state.settings.transcribe_max_duration_seconds,
        )
    except AudioTooLongError as exc:
        raise HTTPException(
            status_code=413,
            detail=(
                "audio duration too long "
                f"(max {exc.max_seconds:g} seconds)"
            ),
        ) from exc


@router.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    limiter, ip = await _acquire_audio_slot(request)
    try:
        audio = await _read_transcribe_upload(request, file)
        if not audio:
            raise HTTPException(status_code=400, detail="audio file is empty")
        filename = file.filename or "speech.wav"
        await _validate_transcribe_duration(request, audio, filename)

        pipeline = getattr(request.app.state, "pipeline", None)
        asr = getattr(pipeline, "asr", None)
        if asr is None:
            raise HTTPException(status_code=503, detail="ASR SDK is not configured")

        try:
            text = await asr.transcribe(audio, filename=filename)
        except ASRClientError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"filename": filename, "bytes": len(audio), "text": text}
    finally:
        await limiter.release(ip)


@router.post("/transcribe/stream")
async def transcribe_and_correct(
    request: Request, file: UploadFile = File(...)
) -> StreamingResponse:
    limiter, ip = await _acquire_audio_slot(request)
    try:
        audio = await _read_transcribe_upload(request, file)
        if not audio:
            raise HTTPException(status_code=400, detail="audio file is empty")
        filename = file.filename or "speech.wav"
        await _validate_transcribe_duration(request, audio, filename)

        pipeline = getattr(request.app.state, "pipeline", None)
        asr = getattr(pipeline, "asr", None)
        llm = getattr(pipeline, "llm", None)
        settings = request.app.state.settings
        if asr is None:
            raise HTTPException(status_code=503, detail="ASR SDK is not configured")
        if llm is None or not settings.llm_correction_enabled:
            raise HTTPException(status_code=503, detail="LLM correction is disabled")
        if not llm.model:
            raise HTTPException(status_code=503, detail="LLM model is not configured")

        timeline: list[dict[str, object]] = []
        meeting_pipeline = getattr(request.app.state, "meeting_pipeline", None)
        vad = getattr(meeting_pipeline, "vad", None)
        if vad is not None:
            try:
                with tempfile.TemporaryDirectory(prefix="astra_stream_") as directory:
                    source = Path(directory) / Path(filename).name
                    source.write_bytes(audio)
                    wav_path, _ = await asyncio.to_thread(MeetingPipeline.decode_to_wav, source)
                    chunks = await vad.detect(wav_path)
                    for index, chunk in enumerate(chunks):
                        text = await asr.transcribe(chunk.audio, filename=f"stream-{index}.wav")
                        text = (text or "").strip()
                        if text:
                            timeline.append(
                                {"index": index, "start": chunk.start, "end": chunk.end, "text": text}
                            )
            except Exception:
                # VAD/afconvert is an enhancement; keep whole-file transcription usable
                # when local audio conversion or the optional VAD model is unavailable.
                timeline = []

        if not timeline:
            try:
                text = await asr.transcribe(audio, filename=filename)
            except ASRClientError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            timeline = [{"index": 0, "start": 0.0, "end": 0.0, "text": text}]
    except Exception:
        await limiter.release(ip)
        raise

    async def events() -> AsyncIterator[str]:
        try:
            if len(timeline) == 1 and timeline[0]["end"] == 0.0:
                yield _sse({"type": "asr_final", "text": timeline[0]["text"]})
            for segment in timeline:
                index = int(segment["index"])
                start = float(segment["start"])
                end = float(segment["end"])
                segment_text = str(segment["text"])
                yield _sse({"type": "asr_segment", "index": index, "start": start, "end": end, "text": segment_text})
                corrected = ""
                messages = [
                    {"role": "system", "content": settings.llm_correction_system_prompt},
                    {"role": "user", "content": segment_text},
                ]
                try:
                    async for token in llm.stream_chat(
                        messages,
                        temperature=0.0,
                        max_tokens=settings.llm_correction_max_tokens,
                        chat_template_kwargs={"enable_thinking": False},
                    ):
                        corrected += token
                        yield _sse({"type": "correction_token", "index": index, "token": token})
                except LLMClientError as exc:
                    yield _sse({"type": "error", "code": "llm_correction_failed", "message": str(exc)})
                    corrected = segment_text
                    yield _sse({"type": "correction_segment", "index": index, "start": start, "end": end, "text": corrected, "fallback": True})
                else:
                    yield _sse({"type": "correction_segment", "index": index, "start": start, "end": end, "text": corrected.strip()})
            yield "data: [DONE]\n\n"
        finally:
            await limiter.release(ip)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
