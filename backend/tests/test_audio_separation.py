import asyncio
from pathlib import Path

import numpy as np

from app.core.audio_adapter import audio_buffer_to_wav_bytes, decode_audio_file
from app.core.audio_enhancement import AudioBuffer, EnhancementContext
from app.core.audio_separation import (
    FLASepformerStage,
    SeparationMetrics,
    build_audio_separation_stage,
)
from app.core.meeting import MeetingPipeline
from app.core.workflow import Segment, WorkflowResult


class FakeSeparationBackend:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def __call__(self, path: str) -> dict[str, list[bytes]]:
        self.inputs.append(path)
        audio = decode_audio_file(path, target_sample_rate=8_000)
        pcm = np.zeros(len(audio.samples), dtype="<i2").tobytes()
        return {"output_pcm_list": [pcm, pcm]}


def _audio(samples: int = 400) -> AudioBuffer:
    return AudioBuffer(
        samples=np.linspace(-0.4, 0.4, samples, dtype=np.float32),
        sample_rate=8_000,
        channels=1,
        source="upload",
    )


def test_flasepformer_uses_bounded_windows_and_returns_two_tracks() -> None:
    backend = FakeSeparationBackend()
    tracks, metrics = asyncio.run(
        FLASepformerStage(
            backend=backend,
            window_seconds=0.02,
        ).process(_audio(400), EnhancementContext())
    )

    assert len(backend.inputs) == 3
    assert len(tracks) == 2
    assert all(track.sample_rate == 8_000 for track in tracks)
    assert all(len(track.samples) == 400 for track in tracks)
    assert metrics.status == "applied"
    assert metrics.details["window_count"] == 3
    assert metrics.details["source_count"] == 2


def test_flasepformer_is_not_applicable_to_realtime() -> None:
    stage = FLASepformerStage(backend=FakeSeparationBackend())
    tracks, metrics = asyncio.run(
        stage.process(_audio(), EnhancementContext(realtime=True))
    )

    assert tracks == []
    assert metrics.status == "not_applicable"
    assert "offline" in (metrics.fallback_reason or "")


def test_build_audio_separation_stage_is_disabled_by_default() -> None:
    assert build_audio_separation_stage(
        enabled=False,
        separation_model="flasepformer_8k",
    ) is None
    stage = build_audio_separation_stage(
        enabled=True,
        separation_model="flasepformer_8k",
    )
    assert stage is not None
    assert stage.name == "flasepformer_8k"


class FakeMeetingSeparation:
    name = "flasepformer_8k"
    input_sample_rate = 8_000
    output_sample_rate = 8_000
    realtime = False

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[list[AudioBuffer], SeparationMetrics]:
        tracks = [
            AudioBuffer(
                samples=np.zeros(len(audio.samples), dtype=np.float32),
                sample_rate=8_000,
                channels=1,
                source="separated",
            )
            for _ in range(2)
        ]
        return tracks, SeparationMetrics(
            stage_name=self.name,
            status="applied",
            input_sample_rate=8_000,
            output_sample_rate=8_000,
            latency_ms=1.0,
            output_count=2,
        )


def test_meeting_pipeline_transcribes_separated_tracks_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    pipeline = MeetingPipeline(
        asr=None,
        diarization=None,
        separation=FakeMeetingSeparation(),
    )

    async def fake_run(wav: str | Path, *, filename: str) -> WorkflowResult:
        return WorkflowResult(
            language="zh",
            segments=[Segment(start=0.0, end=0.1, text=Path(wav).stem)],
        )

    pipeline.separated_workflow.run = fake_run  # type: ignore[method-assign]
    result = asyncio.run(
        pipeline.process(
            audio_buffer_to_wav_bytes(_audio(1_600)),
            filename="meeting.wav",
            summarize=False,
            do_translate=False,
            separate=True,
            separation_artifact_dir=tmp_path / "separated",
        )
    )

    assert result.separation_metrics[0]["status"] == "applied"
    assert [segment.speaker for segment in result.segments] == ["S1", "S2"]
    assert len(result.separation_artifacts) == 2
    assert all(Path(path).exists() for path in result.separation_artifacts)
