import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..core.session_manager import Session
from ..schemas.ws import ClientMessage, StateChange

router = APIRouter()


async def send_state(websocket: WebSocket, session: Session) -> None:
    payload = StateChange(
        state=session.state,
        generation_id=session.generation_id or None,
    )
    await websocket.send_json(payload.model_dump(exclude_none=True))


@router.websocket("/ws")
async def session_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = Session()
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
                session.speech_end()
            elif command.type == "interrupt":
                session.interrupt(command.generation_id, command.reason or "manual")
            elif command.type == "end_session":
                session.end()
            await send_state(websocket, session)
    except WebSocketDisconnect:
        session.end()
