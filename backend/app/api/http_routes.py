from fastapi import APIRouter, File, HTTPException, Request, UploadFile

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

    text = await asr.transcribe(audio, filename=file.filename or "speech.wav")
    return {"filename": file.filename or "speech.wav", "bytes": len(audio), "text": text}
