import base64
from collections.abc import AsyncIterator, Mapping, Sequence

import pytest

from app.core.pipeline import VoicePipeline


class FakeASR:
    async def transcribe(self, audio: bytes) -> str:
        assert audio == b"wav"
        return "hello"


class FakeLLM:
    async def stream_chat(self, messages: Sequence[Mapping[str, str]]) -> AsyncIterator[str]:
        assert messages[0]["role"] == "user"
        for token in ("First sentence. ", "Second sentence"):
            yield token


class FakeTTS:
    async def synthesize(self, text: str) -> bytes:
        return text.encode()


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
