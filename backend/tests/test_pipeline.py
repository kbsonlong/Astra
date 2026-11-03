import io
import wave
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import pytest

from app.core.pipeline import VoicePipeline
from app.core.tts_voice_store import TtsVoiceSnapshot
from app.core.pcm_protocol import decode_tts_frame


class FakeASR:
    async def transcribe(self, audio: bytes) -> str:
        assert audio == b"wav"
        return "hello"


class FakeLLM:
    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> AsyncIterator[str]:
        assert messages[0]["role"] == "user"
        for token in ("First sentence. ", "Second sentence"):
            yield token


class FakeTTS:
    async def synthesize(self, text: str) -> bytes:
        return text.encode()


class WavTTS:
    async def synthesize(self, text: str) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(b"\x00\x00")
        return output.getvalue()


class SnapshotTTS:
    def __init__(self) -> None:
        self.snapshot_calls: list[tuple[str, TtsVoiceSnapshot]] = []

    async def synthesize(self, text: str) -> bytes:
        raise AssertionError("legacy TTS path must not be used for a voice snapshot")

    async def synthesize_snapshot(self, text: str, snapshot: TtsVoiceSnapshot) -> bytes:
        self.snapshot_calls.append((text, snapshot))
        return b"snapshot:" + text.encode()


@pytest.mark.anyio
async def test_pipeline_emits_ordered_generation_scoped_events() -> None:
    events: list[dict[str, object] | bytes] = []

    async def emit(event: dict[str, object] | bytes) -> None:
        events.append(event)

    pipeline = VoicePipeline(FakeASR(), FakeLLM(), FakeTTS())
    await pipeline.run(b"wav", [{"role": "user", "content": "hi"}], 7, emit)

    # TTS audio is a binary frame; every other event stays a JSON dict.
    kinds = [
        "bytes" if isinstance(event, (bytes, bytearray)) else event["type"]
        for event in events
    ]
    assert kinds == [
        "asr_final",
        "llm_token",
        "tts_start",
        "bytes",
        "llm_token",
        "tts_start",
        "bytes",
        "tts_end",
    ]
    for event in events:
        if isinstance(event, (bytes, bytearray)):
            assert decode_tts_frame(event).generation_id == 7
        else:
            assert event["generation_id"] == 7
    second_chunk = decode_tts_frame(events[6])
    assert second_chunk.payload == b"Second sentence"
    assert second_chunk.sequence == 1


@pytest.mark.anyio
async def test_pipeline_reports_actual_tts_wav_sample_rate() -> None:
    events: list[dict[str, object] | bytes] = []

    async def emit(event: dict[str, object] | bytes) -> None:
        events.append(event)

    pipeline = VoicePipeline(FakeASR(), FakeLLM(), WavTTS())
    await pipeline.run(b"wav", [], 1, emit)

    frame = next(
        decode_tts_frame(event)
        for event in events
        if isinstance(event, (bytes, bytearray))
    )
    assert frame.sample_rate == 24_000


@pytest.mark.anyio
async def test_pipeline_synthesizes_with_session_voice_snapshot() -> None:
    events: list[dict[str, object] | bytes] = []
    tts = SnapshotTTS()
    snapshot = TtsVoiceSnapshot(
        voice_id="voice-1",
        voice_revision=2,
        backend_family="indextts25_mlx",
        language="zh",
        reference_path=Path("/tmp/reference.wav"),
        reference_sha256="b" * 64,
        default_params={},
    )

    async def emit(event: dict[str, object] | bytes) -> None:
        events.append(event)

    pipeline = VoicePipeline(FakeASR(), FakeLLM(), tts)
    await pipeline.run(b"wav", [], 1, emit, voice_snapshot=snapshot)

    assert [text for text, _ in tts.snapshot_calls] == [
        "First sentence.",
        "Second sentence",
    ]
    assert all(received == snapshot for _, received in tts.snapshot_calls)
    chunks = [
        decode_tts_frame(event)
        for event in events
        if isinstance(event, (bytes, bytearray))
    ]
    assert chunks[0].payload == b"snapshot:First sentence."
