from collections.abc import Mapping, Sequence

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.audio_adapter import audio_buffer_to_wav_bytes
from app.core.audio_enhancement import AudioBuffer, AudioEnhancementPipeline, EnhancementMetrics
from app.core.audio_separation import SeparationMetrics
from app.main import create_app
from app.models.asr_client import ASRClientError


class FakeASR:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        assert audio == b"wav-data"
        assert filename == "test.wav"
        return "测试语音"

    def is_ready(self) -> bool:
        return True


class FakeLLM:
    model = "test-model"

    async def stream_chat(self, messages, **kwargs):
        assert messages[1]["content"] == "测试语音"
        assert kwargs["max_tokens"] == 256
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
        yield "测试"
        yield "语音。"


class FakePipeline:
    asr = FakeASR()
    llm = FakeLLM()


class EnhancedASR:
    def __init__(self) -> None:
        self.audio: bytes | None = None

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.audio = audio
        assert audio.startswith(b"RIFF")
        return "测试语音"

    def is_ready(self) -> bool:
        return True


class EnhancedPipeline:
    def __init__(self) -> None:
        self.asr = EnhancedASR()
        self.llm = FakeLLM()


class FakeEnhancementStage:
    name = "fake_ans"
    input_sample_rate = 16_000
    output_sample_rate = 16_000
    realtime = False

    async def process(self, audio, context):
        return audio, EnhancementMetrics(
            stage_name=self.name,
            status="applied",
            input_sample_rate=audio.sample_rate,
            output_sample_rate=audio.sample_rate,
            latency_ms=1.0,
            reference_present=context.reference is not None,
        )


class FakeStreamSeparationStage:
    name = "fake_separation"
    input_sample_rate = 8_000
    output_sample_rate = 8_000
    realtime = False

    async def process(self, audio, context):
        track = np.zeros(len(audio.samples), dtype=np.float32)
        return [
            AudioBuffer(track, 8_000, 1, source="separated"),
            AudioBuffer(track, 8_000, 1, source="separated"),
        ], SeparationMetrics(
            stage_name=self.name,
            status="applied",
            input_sample_rate=audio.sample_rate,
            output_sample_rate=audio.sample_rate,
            latency_ms=2.0,
            output_count=2,
        )


class FailingASR:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        raise ASRClientError("mlx-audio model is not available locally")

    def is_ready(self) -> bool:
        return False


class FailingPipeline:
    asr = FailingASR()


def test_transcribe_route_returns_sdk_result() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", b"wav-data", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"filename": "test.wav", "bytes": 8, "text": "测试语音"}


def test_transcribe_route_rejects_empty_file() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/api/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "audio file is empty"


def test_transcribe_stream_route_returns_asr_and_correction_events() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/api/transcribe/stream",
        files={"file": ("test.wav", b"wav-data", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "asr_final"' in response.text
    assert '"type": "correction_token"' in response.text
    assert '"text": "测试语音。"' in response.text
    assert "data: [DONE]" in response.text


def test_transcribe_stream_route_applies_enhancement_before_asr() -> None:
    pipeline = EnhancedPipeline()
    app = create_app(
        settings=Settings(transcribe_max_upload_bytes=25 * 1024 * 1024),
        pipeline=pipeline,
    )
    app.state.meeting_pipeline.vad = None
    app.state.meeting_pipeline.enhancement = AudioEnhancementPipeline(
        [FakeEnhancementStage()]
    )
    audio = audio_buffer_to_wav_bytes(
        AudioBuffer(samples=np.zeros(1600, dtype=np.float32), sample_rate=16_000, channels=1)
    )

    response = TestClient(app).post(
        "/api/transcribe/stream",
        files={"file": ("test.wav", audio, "audio/wav")},
    )

    assert response.status_code == 200
    assert '"type": "enhancement_status"' in response.text
    assert response.text.index('"type": "enhancement_status"') < response.text.index(
        '"type": "asr_final"'
    )
    assert pipeline.asr.audio == audio


def test_transcribe_stream_route_can_run_separation_before_asr() -> None:
    pipeline = EnhancedPipeline()
    app = create_app(
        settings=Settings(transcribe_max_upload_bytes=25 * 1024 * 1024),
        pipeline=pipeline,
    )
    app.state.meeting_pipeline.vad = None
    app.state.meeting_pipeline.separation = FakeStreamSeparationStage()
    audio = audio_buffer_to_wav_bytes(
        AudioBuffer(samples=np.zeros(1600, dtype=np.float32), sample_rate=16_000, channels=1)
    )

    response = TestClient(app).post(
        "/api/transcribe/stream?separate=true",
        files={"file": ("test.wav", audio, "audio/wav")},
    )

    assert response.status_code == 200
    assert '"type": "separation_status"' in response.text
    assert '"output_count": 2' in response.text
    assert response.text.count('"type": "asr_segment"') == 2
    assert response.text.count('"source_index":') == 2


def test_transcribe_route_returns_service_unavailable_for_sdk_error() -> None:
    client = TestClient(create_app(pipeline=FailingPipeline()))

    response = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", b"wav-data", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "mlx-audio model is not available locally"


def test_transcribe_routes_reject_oversized_audio() -> None:
    client = TestClient(
        create_app(
            settings=Settings(transcribe_max_upload_bytes=4),
            pipeline=FakePipeline(),
        )
    )

    for endpoint in ("/api/transcribe", "/api/transcribe/stream"):
        response = client.post(
            endpoint,
            files={"file": ("test.wav", b"oversized", "audio/wav")},
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "audio file too large (max 4 bytes)"
