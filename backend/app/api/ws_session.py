import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .auth import authorize_websocket
from ..core.pcm_protocol import decode_pcm_frame
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
    reference: bytes | None,
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
        # 历史上下文最多保留最近 20 条消息(10 轮), 本轮 user/assistant
        # 成功后追加; 失败/打断不写入, 避免半截回复污染上下文。
        context = session.history[-20:]
        if reference is None:
            user_text, reply_text = await pipeline.run(audio, context, generation_id, emit)
        else:
            user_text, reply_text = await pipeline.run(
                audio, context, generation_id, emit, reference=reference
            )
        if session.accepts(generation_id):
            session.history.append({"role": "user", "content": user_text})
            if reply_text:
                session.history.append({"role": "assistant", "content": reply_text})
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
    if not await authorize_websocket(websocket):
        return
    limiter = websocket.app.state.audio_ip_limiter
    client_ip = websocket.client.host if websocket.client is not None else "unknown"
    if not await limiter.try_acquire(client_ip):
        await websocket.accept()
        await websocket.send_json(
            {"type": "error", "code": "concurrency_limit"}
        )
        await websocket.close(code=1013, reason="audio concurrency limit reached")
        return
    await websocket.accept()
    session = Session(max_audio_bytes=websocket.app.state.settings.ws_max_audio_bytes)
    pipeline: VoicePipeline | None = getattr(websocket.app.state, "pipeline", None)
    generation_task: asyncio.Task[None] | None = None
    audio_channel = "microphone"
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if session.pcm_mode:
                    try:
                        frame = decode_pcm_frame(message["bytes"])
                    except ValueError as exc:
                        await websocket.send_json(
                            {"type": "error", "code": "invalid_pcm_frame", "message": str(exc)}
                        )
                        continue
                    accepted = session.append_pcm_frame(frame)
                else:
                    accepted = session.append_audio(message["bytes"], source=audio_channel)
                if not accepted:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "audio_too_large",
                            "max_bytes": session.max_audio_bytes,
                        }
                    )
                    await websocket.close(code=1009)
                    return
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
                audio_channel = "microphone"
            elif command.type == "audio_channel":
                if command.channel is None:
                    await websocket.send_json(
                        {"type": "error", "code": "invalid_audio_channel"}
                    )
                    continue
                audio_channel = command.channel
            elif command.type == "audio_format":
                if (
                    command.format != "pcm16"
                    or command.sample_rate is None
                    or command.frame_samples is None
                    or not session.configure_pcm(
                        sample_rate=command.sample_rate,
                        frame_samples=command.frame_samples,
                    )
                ):
                    await websocket.send_json(
                        {"type": "error", "code": "invalid_audio_format"}
                    )
                    continue
                await websocket.send_json(
                    {
                        "type": "audio_format_ready",
                        "format": "pcm16",
                        "sample_rate": session.pcm_sample_rate,
                        "frame_samples": session.pcm_frame_samples,
                    }
                )
            elif command.type == "speech_end":
                if session.pcm_mode:
                    audio, reference = session.take_pcm_pair()
                else:
                    audio = session.take_audio()
                    reference = session.take_reference()
                generation_id = session.speech_end()
                if pipeline is not None:
                    generation_task = asyncio.create_task(
                        run_generation(
                            websocket,
                            session,
                            pipeline,
                            audio,
                            reference,
                            generation_id,
                        )
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
    finally:
        await limiter.release(client_ip)
