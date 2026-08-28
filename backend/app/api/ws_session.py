import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..core.session_manager import Session
from ..core.pipeline import VoicePipeline
from ..schemas.ws import ClientMessage, StateChange

router = APIRouter()


async def send_state(websocket: WebSocket, session: Session) -> None:
    payload = StateChange(
        state=session.state,
        generation_id=session.generation_id or None,
    )
    await websocket.send_json(payload.model_dump(exclude_none=True))


async def run_generation(
    websocket: WebSocket,
    session: Session,
    pipeline: VoicePipeline,
    audio: bytes,
    generation_id: int,
) -> None:
    async def emit(event: dict[str, object]) -> None:
        if not session.accepts(generation_id):
            return
        if event.get("type") == "tts_start":
            session.state = "SPEAKING"
            await send_state(websocket, session)
        await websocket.send_json(event)

    try:
        await pipeline.run(audio, [], generation_id, emit)
        if session.accepts(generation_id):
            session.state = "LISTENING"
            await send_state(websocket, session)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - exact model errors vary by provider
        if session.accepts(generation_id):
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "pipeline_failed",
                    "message": str(exc),
                    "generation_id": generation_id,
                }
            )
            session.state = "LISTENING"
            await send_state(websocket, session)


@router.websocket("/ws")
async def session_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = Session()
    pipeline: VoicePipeline | None = getattr(websocket.app.state, "pipeline", None)
    generation_task: asyncio.Task[None] | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                session.append_audio(message["bytes"])
                continue
            if message.get("text") is None:
                continue
            try:
                command = ClientMessage.model_validate(json.loads(message["text"]))
            except (json.JSONDecodeError, ValidationError):
                await websocket.send_json({"type": "error", "code": "invalid_message"})
                continue

            if command.type == "start_session":
                session.start()
            elif command.type == "speech_end":
                audio = session.take_audio()
                generation_id = session.speech_end()
                if pipeline is not None:
                    generation_task = asyncio.create_task(
                        run_generation(websocket, session, pipeline, audio, generation_id)
                    )
            elif command.type == "interrupt":
                if session.interrupt(command.generation_id, command.reason or "manual"):
                    if generation_task is not None:
                        generation_task.cancel()
                        await asyncio.gather(generation_task, return_exceptions=True)
                        generation_task = None
            elif command.type == "end_session":
                session.end()
                if generation_task is not None:
                    generation_task.cancel()
                    await asyncio.gather(generation_task, return_exceptions=True)
                    generation_task = None
            await send_state(websocket, session)
    except WebSocketDisconnect:
        session.end()
        if generation_task is not None:
            generation_task.cancel()
            await asyncio.gather(generation_task, return_exceptions=True)
