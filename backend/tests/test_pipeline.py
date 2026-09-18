import base64
import io
import wave
from collections.abc import AsyncIterator, Mapping, Sequence

import pytest

from app.core.pipeline import VoicePipeline


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


@pytest.mark.anyio
async def test_pipeline_emits_ordered_generation_scoped_events() -> None:
    events: list[dict[str, object]] = []

    async def emit(event: dict[str, object]) -> None:
        events.append(event)

    pipeline = VoicePipeline(FakeASR(), FakeLLM(), FakeTTS())
    await pipeline.run(b"wav", [{"role": "user", "content": "hi"}], 7, emit)

    assert [event["type"] for event in events] == [
        "asr_final",
        "llm_token",
        "tts_start",
        "tts_chunk",
        "llm_token",
        "tts_start",
        "tts_chunk",
        "tts_end",
    ]
    assert all(event["generation_id"] == 7 for event in events)
    chunk = events[6]
    assert base64.b64decode(chunk["audio_b64"]) == b"Second sentence"


@pytest.mark.anyio
async def test_pipeline_reports_actual_tts_wav_sample_rate() -> None:
    events: list[dict[str, object]] = []

    async def emit(event: dict[str, object]) -> None:
        events.append(event)

    pipeline = VoicePipeline(FakeASR(), FakeLLM(), WavTTS())
    await pipeline.run(b"wav", [], 1, emit)

    chunk = next(event for event in events if event["type"] == "tts_chunk")
    assert chunk["sample_rate"] == 24_000
