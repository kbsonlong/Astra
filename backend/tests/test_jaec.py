import asyncio

import numpy as np

from app.core.audio_adapter import decode_audio_bytes
from app.core.audio_enhancement import AudioBuffer, EnhancementContext
from app.core.jaec import JAECStage, build_audio_aec_pipeline


class FakeJAECBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, bytes]] = []

    def __call__(self, inputs: dict[str, bytes]) -> dict[str, bytes]:
        self.calls.append(inputs)
        microphone = decode_audio_bytes(
            inputs["nearend_mic"], filename="nearend.wav"
        )
        return {
            "output_pcm": np.zeros(len(microphone.samples), dtype="<i2").tobytes()
        }


def _audio(value: float, samples: int = 320) -> AudioBuffer:
    return AudioBuffer(
        samples=np.full(samples, value, dtype=np.float32),
        sample_rate=16_000,
        channels=1,
        source="microphone",
    )


def test_jaec_requires_reference_and_preserves_length() -> None:
    backend = FakeJAECBackend()
    output, metrics = asyncio.run(
        JAECStage(backend=backend).process(
            _audio(0.4),
            EnhancementContext(reference=_audio(0.2), realtime=True),
        )
    )

    assert len(backend.calls) == 1
    assert set(backend.calls[0]) == {"nearend_mic", "farend_speech"}
    assert output.sample_rate == 16_000
    assert len(output.samples) == 320
    assert metrics.status == "applied"
    assert metrics.details["algorithmic_delay_ms"] == 22.0
    assert metrics.details["algorithmic_delay_samples"] == 352


def test_jaec_without_reference_is_not_applicable() -> None:
    backend = FakeJAECBackend()
    audio = _audio(0.4)

    output, metrics = asyncio.run(
        JAECStage(backend=backend).process(audio, EnhancementContext(realtime=True))
    )

    assert output is audio
    assert metrics.status == "not_applicable"
    assert "far-end reference" in (metrics.fallback_reason or "")
    assert backend.calls == []


def test_jaec_rejects_mismatched_reference_length() -> None:
    with_error = JAECStage(backend=FakeJAECBackend())

    try:
        asyncio.run(
            with_error.process(
                _audio(0.4, samples=320),
                EnhancementContext(reference=_audio(0.2, samples=160)),
            )
        )
    except ValueError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError("expected JAEC to reject mismatched input lengths")


def test_build_audio_aec_pipeline_is_disabled_by_default() -> None:
    assert build_audio_aec_pipeline(enabled=False, aec_model="jaec_16k") is None
    pipeline = build_audio_aec_pipeline(enabled=True, aec_model="jaec_16k")
    assert pipeline is not None
    assert pipeline.stages[0].name == "jaec_16k"
