import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..models.asr_client import ASRClientError
from ..models.llm_client import LLMClientError

router = APIRouter(prefix="/api")


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="audio file is empty")

    pipeline = getattr(request.app.state, "pipeline", None)
    asr = getattr(pipeline, "asr", None)
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR SDK is not configured")

    try:
        text = await asr.transcribe(audio, filename=file.filename or "speech.wav")
    except ASRClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"filename": file.filename or "speech.wav", "bytes": len(audio), "text": text}


@router.post("/transcribe/stream")
async def transcribe_and_correct(
    request: Request, file: UploadFile = File(...)
) -> StreamingResponse:
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="audio file is empty")

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

    try:
        text = await asr.transcribe(audio, filename=file.filename or "speech.wav")
    except ASRClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def events() -> AsyncIterator[str]:
        yield _sse({"type": "asr_final", "text": text})
        corrected = ""
        messages = [
            {"role": "system", "content": settings.llm_correction_system_prompt},
            {"role": "user", "content": text},
        ]
        try:
            async for token in llm.stream_chat(
                messages,
                temperature=0.0,
                max_tokens=settings.llm_correction_max_tokens,
                chat_template_kwargs={"enable_thinking": False},
            ):
                corrected += token
                yield _sse({"type": "correction_token", "token": token})
        except LLMClientError as exc:
            yield _sse({"type": "error", "code": "llm_correction_failed", "message": str(exc)})
            yield _sse({"type": "correction_final", "text": text, "fallback": True})
        else:
            yield _sse({"type": "correction_final", "text": corrected.strip()})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
