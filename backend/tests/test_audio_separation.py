import asyncio
from pathlib import Path

import numpy as np

from app.core.audio_adapter import audio_buffer_to_wav_bytes, decode_audio_file
from app.core.audio_enhancement import AudioBuffer, EnhancementContext
from app.core.audio_separation import (
    FLASepformerStage,
    OverlapDetection,
    PyannoteOverlapDetector,
    SpectralOverlapDetector,
    SeparationMetrics,
    build_overlap_detector,
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


def test_spectral_overlap_detector_distinguishes_two_prominent_voice_band_tones() -> None:
    sample_rate = 16_000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    single = np.sin(2 * np.pi * 140 * time_axis).astype(np.float32)
    mixed = (
        0.5 * np.sin(2 * np.pi * 140 * time_axis)
        + 0.5 * np.sin(2 * np.pi * 260 * time_axis)
    ).astype(np.float32)
    harmonic = (
        0.7 * np.sin(2 * np.pi * 140 * time_axis)
        + 0.3 * np.sin(2 * np.pi * 280 * time_axis)
    ).astype(np.float32)
    noise = np.random.default_rng(7).normal(0, 0.2, len(time_axis)).astype(np.float32)
    detector = SpectralOverlapDetector()

    single_result = detector.detect(
        AudioBuffer(samples=single, sample_rate=sample_rate, channels=1)
    )
    mixed_result = detector.detect(
        AudioBuffer(samples=mixed, sample_rate=sample_rate, channels=1)
    )
    harmonic_result = detector.detect(
        AudioBuffer(samples=harmonic, sample_rate=sample_rate, channels=1)
    )
    noise_result = detector.detect(
        AudioBuffer(samples=noise, sample_rate=sample_rate, channels=1)
    )

    assert not single_result.suspected
    assert mixed_result.suspected
    assert mixed_result.candidate_frames > 0
    assert not harmonic_result.suspected
    assert not noise_result.suspected
    assert harmonic_result.details["harmonic_rejections"] > 0
    assert noise_result.details["noise_rejections"] > 0


class FakePyannoteSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakePyannoteTimeline:
    def support(self) -> list[FakePyannoteSegment]:
        return [FakePyannoteSegment(0.25, 0.65)]


class FakePyannoteOutput:
    def get_timeline(self) -> FakePyannoteTimeline:
        return FakePyannoteTimeline()


class FakePyannoteBackend:
    def __call__(self, path: str) -> FakePyannoteOutput:
        assert Path(path).is_file()
        return FakePyannoteOutput()


def test_pyannote_overlap_detector_converts_timeline_to_detection() -> None:
    detector = PyannoteOverlapDetector(
        backend=FakePyannoteBackend(),
        min_overlap_seconds=0.2,
    )
    result = detector.detect(_audio(8_000))

    assert result.status == "ok"
    assert result.suspected
    assert result.score == 0.4
    assert result.details["detector"] == "pyannote_osd"
    assert result.details["overlap_seconds"] == 0.4


def test_build_overlap_detector_defaults_to_heuristic() -> None:
    assert isinstance(build_overlap_detector(), SpectralOverlapDetector)
    assert isinstance(
        build_overlap_detector(model="pyannote_osd", model_dir="/models/osd"),
        PyannoteOverlapDetector,
    )


class FakeOverlapDetector:
    def __init__(self, suspected: bool) -> None:
        self.suspected = suspected

    def detect(self, audio: AudioBuffer) -> OverlapDetection:
        return OverlapDetection(
            suspected=self.suspected,
            score=1.0 if self.suspected else 0.0,
            active_frames=10,
            candidate_frames=10 if self.suspected else 0,
            threshold=0.55,
        )


class FailingOverlapDetector:
    name = "pyannote_osd"

    def detect(self, audio: AudioBuffer) -> OverlapDetection:
        raise RuntimeError("model terms or local pipeline are not ready")


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


def test_meeting_pipeline_overlap_trigger_runs_detector_before_separation() -> None:
    pipeline = MeetingPipeline(
        asr=None,
        diarization=None,
        separation=FakeMeetingSeparation(),
        separation_trigger="overlap",
        overlap_detector=FakeOverlapDetector(suspected=True),
    )

    async def fake_run(wav: str | Path, *, filename: str) -> WorkflowResult:
        return WorkflowResult(
            language="zh",
            segments=[Segment(start=0.0, end=0.1, text="overlap")],
        )

    pipeline.separated_workflow.run = fake_run  # type: ignore[method-assign]
    result = asyncio.run(
        pipeline.process(
            audio_buffer_to_wav_bytes(_audio(1_600)),
            filename="meeting.wav",
            summarize=False,
            do_translate=False,
        )
    )

    assert result.overlap_detection == {
        "status": "ok",
        "suspected": True,
        "score": 1.0,
        "active_frames": 10,
        "candidate_frames": 10,
        "threshold": 0.55,
        "details": {},
    }
    assert result.separation_metrics[0]["status"] == "applied"


def test_meeting_pipeline_overlap_trigger_keeps_mixed_audio_when_clear() -> None:
    pipeline = MeetingPipeline(
        asr=None,
        diarization=None,
        separation=FakeMeetingSeparation(),
        separation_trigger="overlap",
        overlap_detector=FakeOverlapDetector(suspected=False),
    )
    calls: list[str] = []

    async def fake_transcribe(wav: str, *, progress=None):
        calls.append(wav)
        return "zh", [Segment(start=0.0, end=0.1, text="mixed")]

    pipeline.transcribe = fake_transcribe  # type: ignore[method-assign]
    result = asyncio.run(
        pipeline.process(
            audio_buffer_to_wav_bytes(_audio(1_600)),
            filename="meeting.wav",
            summarize=False,
            do_translate=False,
        )
    )

    assert result.overlap_detection["suspected"] is False
    assert result.separation_metrics == []
    assert len(calls) == 1


def test_meeting_pipeline_records_detector_failure_and_falls_back() -> None:
    pipeline = MeetingPipeline(
        asr=None,
        diarization=None,
        separation=FakeMeetingSeparation(),
        separation_trigger="overlap",
        overlap_detector=FailingOverlapDetector(),
    )
    calls: list[str] = []

    async def fake_transcribe(wav: str, *, progress=None):
        calls.append(wav)
        return "zh", [Segment(start=0.0, end=0.1, text="mixed")]

    pipeline.transcribe = fake_transcribe  # type: ignore[method-assign]
    result = asyncio.run(
        pipeline.process(
            audio_buffer_to_wav_bytes(_audio(1_600)),
            filename="meeting.wav",
            summarize=False,
            do_translate=False,
        )
    )

    assert result.overlap_detection["status"] == "failed"
    assert result.overlap_detection["details"]["detector"] == "pyannote_osd"
    assert result.separation_metrics == []
    assert len(calls) == 1
