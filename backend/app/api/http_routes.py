import asyncio
import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..core.audio_adapter import (
    audio_buffer_to_wav_bytes,
    decode_audio_bytes,
    resample_audio_buffer,
)
from ..core.audio_enhancement import EnhancementContext, EnhancementMetrics
from ..core.audio_separation import SeparationMetrics
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


async def _apply_stream_enhancement(
    audio: bytes, filename: str, meeting_pipeline: object | None
) -> tuple[bytes, list[dict[str, object]]]:
    enhancement = getattr(meeting_pipeline, "enhancement", None)
    if enhancement is None:
        return audio, []

    try:
        decoded = await asyncio.to_thread(
            decode_audio_bytes, audio, filename=filename, source="upload"
        )
        enhanced, metrics = await enhancement.process(
            decoded, EnhancementContext(realtime=False)
        )
        payload = [item.to_dict() for item in metrics]
        if any(item["status"] == "applied" for item in payload):
            return await asyncio.to_thread(audio_buffer_to_wav_bytes, enhanced), payload
        return audio, payload
    except Exception as exc:
        fallback = EnhancementMetrics(
            stage_name="audio_enhancement",
            status="failed",
            input_sample_rate=0,
            output_sample_rate=0,
            latency_ms=0.0,
            reference_present=False,
            fallback_reason=str(exc),
            details={"phase": "decode_or_encode"},
        )
        return audio, [fallback.to_dict()]


async def _transcribe_stream_audio(
    audio: bytes,
    filename: str,
    asr: object,
    vad: object | None,
    *,
    source_index: int | None = None,
    index_offset: int = 0,
) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    if vad is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="astra_stream_") as directory:
                source = Path(directory) / Path(filename).name
                source.write_bytes(audio)
                wav_path, _ = await asyncio.to_thread(MeetingPipeline.decode_to_wav, source)
                chunks = await vad.detect(wav_path)  # type: ignore[attr-defined]
                for chunk_index, chunk in enumerate(chunks):
                    text = await asr.transcribe(  # type: ignore[attr-defined]
                        chunk.audio, filename=f"stream-{index_offset + chunk_index}.wav"
                    )
                    text = (text or "").strip()
                    if text:
                        segment: dict[str, object] = {
                            "index": index_offset + chunk_index,
                            "start": chunk.start,
                            "end": chunk.end,
                            "text": text,
                        }
                        if source_index is not None:
                            segment["source_index"] = source_index
                        timeline.append(segment)
        except Exception:
            # VAD/afconvert is optional; keep whole-file transcription usable.
            timeline = []

    if timeline:
        return timeline
    text = await asr.transcribe(audio, filename=filename)  # type: ignore[attr-defined]
    segment = {"index": index_offset, "start": 0.0, "end": 0.0, "text": text}
    if source_index is not None:
        segment["source_index"] = source_index
    return [segment]


def _stream_separation_failure(reason: str) -> dict[str, object]:
    return SeparationMetrics(
        stage_name="separation",
        status="failed",
        input_sample_rate=0,
        output_sample_rate=0,
        latency_ms=0.0,
        output_count=0,
        fallback_reason=reason,
    ).to_dict()


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
    request: Request,
    file: UploadFile = File(...),
    separate: bool = Query(False),
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
        audio, enhancement_status = await _apply_stream_enhancement(
            audio, filename, meeting_pipeline
        )
        vad = getattr(meeting_pipeline, "vad", None)
        separation_status: dict[str, object] | None = None
        overlap_status: dict[str, object] | None = None
        separation = getattr(meeting_pipeline, "separation", None)
        separation_trigger = getattr(meeting_pipeline, "separation_trigger", "manual")
        separation_requested = separate or separation_trigger == "always"
        decoded = None
        if separation is not None and separation_trigger == "overlap" and not separate:
            try:
                decoded = await asyncio.to_thread(decode_audio_bytes, audio, filename=filename)
                detection = await asyncio.to_thread(
                    meeting_pipeline.overlap_detector.detect, decoded
                )
                overlap_status = detection.to_dict()
                separation_requested = detection.suspected
            except Exception as exc:
                overlap_status = {
                    "status": "failed",
                    "suspected": False,
                    "score": 0.0,
                    "threshold": 1.0,
                    "details": {"error": str(exc)},
                }

        if separation_requested:
            try:
                if separation is None:
                    raise RuntimeError("stream separation is not configured")
                if decoded is None:
                    decoded = await asyncio.to_thread(
                        decode_audio_bytes, audio, filename=filename
                    )
                separated_input = await asyncio.to_thread(
                    resample_audio_buffer, decoded, separation.input_sample_rate
                )
                tracks, metrics = await separation.process(
                    separated_input, EnhancementContext(realtime=False)
                )
                separation_status = metrics.to_dict()
                if metrics.status == "applied" and tracks:
                    for source_index, track in enumerate(tracks):
                        track_for_asr = await asyncio.to_thread(
                            resample_audio_buffer, track, 16_000
                        )
                        track_audio = await asyncio.to_thread(
                            audio_buffer_to_wav_bytes, track_for_asr
                        )
                        timeline.extend(
                            await _transcribe_stream_audio(
                                track_audio,
                                f"stream-source-{source_index}.wav",
                                asr,
                                vad,
                                source_index=source_index,
                                index_offset=len(timeline),
                            )
                        )
            except Exception as exc:
                separation_status = _stream_separation_failure(str(exc))
                timeline = []

        if not timeline:
            try:
                timeline = await _transcribe_stream_audio(audio, filename, asr, vad)
            except ASRClientError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        await limiter.release(ip)
        raise

    async def events() -> AsyncIterator[str]:
        try:
            if enhancement_status:
                yield _sse({"type": "enhancement_status", "stages": enhancement_status})
            if overlap_status is not None:
                yield _sse({"type": "overlap_status", **overlap_status})
            if separation_status is not None:
                yield _sse({"type": "separation_status", "stage": separation_status})
            if len(timeline) == 1 and timeline[0]["end"] == 0.0:
                yield _sse({"type": "asr_final", "text": timeline[0]["text"]})
            for segment in timeline:
                index = int(segment["index"])
                start = float(segment["start"])
                end = float(segment["end"])
                segment_text = str(segment["text"])
                payload: dict[str, object] = {
                    "type": "asr_segment",
                    "index": index,
                    "start": start,
                    "end": end,
                    "text": segment_text,
                }
                if "source_index" in segment:
                    payload["source_index"] = segment["source_index"]
                yield _sse(payload)
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
