import asyncio
import json
from dataclasses import dataclass
from typing import Union

from fastapi import APIRouter, WebSocket
from pydantic import ValidationError

from .auth import authorize_websocket
from ..core.pcm_protocol import decode_pcm_frame
from ..core.realtime_recording import RecordingLimitExceeded, RealtimeRecording
from ..core.session_manager import Session
from ..core.pipeline import VoicePipeline
from ..core.tts_voice_store import TtsVoiceError, TtsVoiceSnapshot
from ..schemas.ws import (
    AudioFormatReadyEvent,
    ClientMessage,
    ErrorEvent,
    RecordingReadyEvent,
    RecordingStartedEvent,
    StateChange,
)

router = APIRouter()

# Returned by a command handler that has already sent its own reply (an error
# frame) and must not trigger the trailing state broadcast. This preserves the
# original handlers' ``continue`` behavior, which skipped ``send_state``.
SUPPRESS_STATE = object()

# A command handler returns either a trailing event dict to send after the
# state broadcast, ``None`` to broadcast state only, or ``SUPPRESS_STATE``.
HandlerResult = Union[dict[str, object], None, object]


async def send_state(websocket: WebSocket, session: Session) -> None:
    payload = StateChange(
        state=session.state,
        generation_id=session.generation_id or None,
        voice_id=(
            session.tts_voice_snapshot.voice_id
            if session.tts_voice_snapshot is not None
            else None
        ),
        voice_revision=(
            session.tts_voice_snapshot.voice_revision
            if session.tts_voice_snapshot is not None
            else None
        ),
    )
    await websocket.send_json(payload.model_dump(exclude_none=True))


async def run_generation(
    websocket: WebSocket,
    session: Session,
    pipeline: VoicePipeline,
    audio: bytes,
    reference: bytes | None,
    generation_id: int,
    voice_snapshot: TtsVoiceSnapshot | None,
) -> None:
    async def emit(event: dict[str, object] | bytes) -> None:
        if not session.accepts(generation_id):
            return
        # Binary payloads are TTS audio frames; JSON dicts are control events.
        if isinstance(event, (bytes, bytearray)):
            await websocket.send_bytes(event)
            return
        if event.get("type") == "tts_start":
            session.state = "SPEAKING"
            await send_state(websocket, session)
        await websocket.send_json(event)

    try:
        # 历史上下文最多保留最近 20 条消息(10 轮), 本轮 user/assistant
        # 成功后追加; 失败/打断不写入, 避免半截回复污染上下文。
        context = session.history[-20:]
        if reference is None and voice_snapshot is None:
            user_text, reply_text = await pipeline.run(audio, context, generation_id, emit)
        elif voice_snapshot is None:
            user_text, reply_text = await pipeline.run(
                audio, context, generation_id, emit, reference=reference
            )
        else:
            user_text, reply_text = await pipeline.run(
                audio,
                context,
                generation_id,
                emit,
                reference=reference,
                voice_snapshot=voice_snapshot,
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
                ErrorEvent(
                    code="pipeline_failed",
                    message=str(exc),
                    generation_id=generation_id,
                ).to_wire()
            )
            session.state = "LISTENING"
            await send_state(websocket, session)


@dataclass
class WsConnection:
    """Per-connection state plus the command handlers for one ``/ws`` session.

    Extracting the handlers off the read loop keeps the loop responsible only
    for framing (binary vs text) and routing, and lets each command be unit
    tested in isolation. ``_cancel_generation`` and ``send_state`` centralize
    the two actions every handler repeats.
    """

    websocket: WebSocket
    session: Session
    pipeline: VoicePipeline | None
    audio_channel: str = "microphone"
    recording: RealtimeRecording | None = None
    recording_limit_reached: bool = False
    segment_limit_reached: bool = False
    last_recorded_sequence: int | None = None
    generation_task: asyncio.Task[None] | None = None

    async def send_state(self) -> None:
        await send_state(self.websocket, self.session)

    async def _send(self, event: object) -> None:
        await self.websocket.send_json(event)

    async def _cancel_generation(self) -> None:
        if self.generation_task is not None:
            self.generation_task.cancel()
            await asyncio.gather(self.generation_task, return_exceptions=True)
            self.generation_task = None

    async def handle_binary(self, data: bytes) -> bool:
        """Ingest one binary audio frame. Returns False to close the socket."""
        if self.session.pcm_mode:
            try:
                frame = decode_pcm_frame(data)
            except ValueError as exc:
                await self._send(
                    ErrorEvent(code="invalid_pcm_frame", message=str(exc)).to_wire()
                )
                return True
            if (
                frame.sample_rate != self.session.pcm_sample_rate
                or frame.sample_count != self.session.pcm_frame_samples
            ):
                await self._send(
                    ErrorEvent(
                        code="invalid_pcm_frame",
                        message="PCM frame does not match the negotiated format",
                    ).to_wire()
                )
                return True
            should_record = (
                self.recording is not None
                and not self.recording_limit_reached
                and frame.channel == "microphone"
                and frame.sequence != self.last_recorded_sequence
            )
            if should_record and self.recording is not None:
                try:
                    self.recording.append(frame.payload)
                except RecordingLimitExceeded:
                    self.recording_limit_reached = True
                    await self._send(
                        ErrorEvent(
                            code="recording_too_large",
                            max_bytes=self.recording.max_bytes,
                        ).to_wire()
                    )
                else:
                    self.last_recorded_sequence = frame.sequence
            accepted = self.segment_limit_reached or self.session.append_pcm_frame(frame)
        else:
            accepted = self.segment_limit_reached or self.session.append_audio(
                data, source=self.audio_channel
            )
        if not accepted:
            self.segment_limit_reached = True
            await self._send(
                ErrorEvent(
                    code="audio_segment_limit_reached",
                    max_bytes=self.session.max_audio_bytes,
                ).to_wire()
            )
        return True

    async def handle_command(self, command: ClientMessage) -> None:
        handler = self._COMMAND_HANDLERS.get(command.type)
        if handler is None:
            return
        # A handler returns one of: SUPPRESS_STATE (it already replied and the
        # trailing state broadcast is skipped, matching the original
        # ``continue`` paths); None (broadcast state only); or a trailing event
        # that must follow the single state broadcast, preserving the original
        # (state_change, then recording_started/recording_ready) ordering.
        result = await handler(self, command)
        if result is SUPPRESS_STATE:
            return
        await self.send_state()
        if result is not None:
            await self._send(result)

    async def _handle_start_session(self, command: ClientMessage) -> HandlerResult:
        voice_snapshot = None
        if command.voice_id is not None:
            try:
                voice_snapshot = self.websocket.app.state.tts_voice_store.snapshot(
                    command.voice_id,
                    command.voice_revision,
                )
            except TtsVoiceError as exc:
                await self._send(
                    ErrorEvent(
                        code="voice_snapshot_unavailable", message=str(exc)
                    ).to_wire()
                )
                return SUPPRESS_STATE
        elif command.voice_revision is not None:
            await self._send(
                ErrorEvent(
                    code="voice_snapshot_unavailable",
                    message="voice_revision requires voice_id",
                ).to_wire()
            )
            return SUPPRESS_STATE
        self.session.start(tts_voice_snapshot=voice_snapshot)
        self.audio_channel = "microphone"
        self.recording_limit_reached = False
        self.segment_limit_reached = False
        self.last_recorded_sequence = None
        if self.recording is not None:
            self.recording.close()
        self.recording = None
        if command.recording_id:
            settings = self.websocket.app.state.settings
            try:
                self.recording = RealtimeRecording(
                    settings.realtime_recording_dir,
                    command.recording_id,
                    max_bytes=settings.realtime_recording_max_bytes,
                )
            except ValueError as exc:
                await self._send(
                    ErrorEvent(
                        code="invalid_recording_id", message=str(exc)
                    ).to_wire()
                )
            else:
                return RecordingStartedEvent(
                    recording_id=self.recording.recording_id
                ).to_wire()
        return None

    async def _handle_audio_channel(self, command: ClientMessage) -> HandlerResult:
        if command.channel is None:
            await self._send(ErrorEvent(code="invalid_audio_channel").to_wire())
            return SUPPRESS_STATE
        self.audio_channel = command.channel
        return None

    async def _handle_audio_format(self, command: ClientMessage) -> HandlerResult:
        if (
            command.format != "pcm16"
            or command.sample_rate is None
            or command.frame_samples is None
            or not self.session.configure_pcm(
                sample_rate=command.sample_rate,
                frame_samples=command.frame_samples,
            )
        ):
            await self._send(ErrorEvent(code="invalid_audio_format").to_wire())
            return SUPPRESS_STATE
        await self._send(
            AudioFormatReadyEvent(
                sample_rate=self.session.pcm_sample_rate,
                frame_samples=self.session.pcm_frame_samples,
            ).to_wire()
        )
        return None

    async def _handle_heartbeat(self, command: ClientMessage) -> HandlerResult:
        await self._send({"type": "heartbeat_ack"})
        return SUPPRESS_STATE

    async def _handle_speech_end(self, command: ClientMessage) -> HandlerResult:
        self._submit_pending_audio()
        return None

    async def _handle_interrupt(self, command: ClientMessage) -> HandlerResult:
        if self.session.interrupt(command.generation_id, command.reason or "manual"):
            await self._cancel_generation()
        return None

    async def _handle_end_session(self, command: ClientMessage) -> HandlerResult:
        # The browser normally submits speech after VAD detects silence. A
        # manual stop can arrive first, though, leaving the final listening
        # segment in the session buffer. Submit that segment before clearing
        # the session so recording-only stops do not lose their transcription.
        if (
            self.pipeline is not None
            and self.session.state == "LISTENING"
            and self.session.has_pending_audio()
        ):
            self._submit_pending_audio()
            if self.generation_task is not None:
                await asyncio.gather(self.generation_task, return_exceptions=True)
                self.generation_task = None
        self.session.end()
        await self._cancel_generation()
        if self.recording is not None:
            path = self.recording.finish()
            recording_id = self.recording.recording_id
            self.recording = None
            return RecordingReadyEvent(
                recording_id=recording_id,
                download_url=(
                    f"/api/realtime-recordings/{recording_id}"
                    if path is not None
                    else ""
                ),
            ).to_wire()
        return None

    def _take_pending_audio(self) -> tuple[bytes, bytes | None]:
        if self.session.pcm_mode:
            return self.session.take_pcm_pair()
        return self.session.take_audio(), self.session.take_reference()

    def _submit_pending_audio(self) -> None:
        audio, reference = self._take_pending_audio()
        self.segment_limit_reached = False
        generation_id = self.session.speech_end()
        if self.pipeline is not None:
            self.generation_task = asyncio.create_task(
                run_generation(
                    self.websocket,
                    self.session,
                    self.pipeline,
                    audio,
                    reference,
                    generation_id,
                    self.session.tts_voice_snapshot,
                )
            )

    _COMMAND_HANDLERS = {
        "start_session": _handle_start_session,
        "audio_channel": _handle_audio_channel,
        "audio_format": _handle_audio_format,
        "heartbeat": _handle_heartbeat,
        "speech_end": _handle_speech_end,
        "interrupt": _handle_interrupt,
        "end_session": _handle_end_session,
    }

    async def close(self) -> None:
        self.session.end()
        await self._shield_cancel_generation()
        if self.recording is not None:
            # Keep partial PCM so a reconnect can resume the same recording;
            # only end_session finalizes the downloadable WAV.
            self.recording.close()

    async def _shield_cancel_generation(self) -> None:
        if self.generation_task is None:
            return
        self.generation_task.cancel()
        cleanup = asyncio.gather(self.generation_task, return_exceptions=True)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup)
        self.generation_task = None


@router.websocket("/ws")
async def session_websocket(websocket: WebSocket) -> None:
    if not await authorize_websocket(websocket):
        return
    limiter = websocket.app.state.audio_ip_limiter
    client_ip = websocket.client.host if websocket.client is not None else "unknown"
    if not await limiter.try_acquire(client_ip):
        await websocket.accept()
        await websocket.send_json(ErrorEvent(code="concurrency_limit").to_wire())
        await websocket.close(code=1013, reason="audio concurrency limit reached")
        return
    await websocket.accept()
    connection = WsConnection(
        websocket=websocket,
        session=Session(max_audio_bytes=websocket.app.state.settings.ws_max_audio_bytes),
        pipeline=getattr(websocket.app.state, "pipeline", None),
    )
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if not await connection.handle_binary(message["bytes"]):
                    return
                continue
            if message.get("text") is None:
                continue
            try:
                command = ClientMessage.model_validate(json.loads(message["text"]))
            except (json.JSONDecodeError, ValidationError):
                await websocket.send_json(ErrorEvent(code="invalid_message").to_wire())
                continue
            await connection.handle_command(command)
    finally:
        try:
            await connection.close()
        finally:
            release = asyncio.create_task(limiter.release(client_ip))
            try:
                await asyncio.shield(release)
            except asyncio.CancelledError:
                await asyncio.shield(release)
