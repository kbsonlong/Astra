from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ..models.asr_client import ASRClientError

router = APIRouter(prefix="/api")


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
