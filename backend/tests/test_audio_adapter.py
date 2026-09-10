import io
import wave
from pathlib import Path

import numpy as np
import pytest

from app.core.audio_adapter import (
    AudioFormatError,
    audio_buffer_to_wav_bytes,
    decode_audio_bytes,
    decode_audio_file,
)
from app.core.audio_enhancement import AudioBuffer
from app.core.meeting import MeetingPipeline


def wav_bytes(*, sample_rate: int, channels: int, seconds: float) -> bytes:
    frame_count = int(sample_rate * seconds)
    frames = np.zeros((frame_count, channels), dtype=np.int16)
    if frame_count:
        frames[0, :] = 1000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames.tobytes())
    return buffer.getvalue()


def test_decode_audio_bytes_normalizes_to_mono_float32_and_target_rate() -> None:
    audio = decode_audio_bytes(
        wav_bytes(sample_rate=8000, channels=2, seconds=1),
        filename="recording.wav",
    )

    assert audio.sample_rate == 16_000
    assert audio.channels == 1
    assert isinstance(audio.samples, np.ndarray)
    assert audio.samples.dtype == np.float32
    assert len(audio.samples) == 16_000
    assert audio.source == "upload"


def test_audio_buffer_wav_round_trip(tmp_path) -> None:
    original = decode_audio_bytes(wav_bytes(sample_rate=16_000, channels=1, seconds=0.25))
    path = tmp_path / "round-trip.wav"
    path.write_bytes(audio_buffer_to_wav_bytes(original))

    restored = decode_audio_file(path)

    assert restored.sample_rate == 16_000
    assert restored.channels == 1
    assert len(restored.samples) == len(original.samples)
    assert np.max(np.abs(restored.samples - original.samples)) <= 1 / 32768


def test_meeting_decode_to_wav_uses_the_common_adapter(tmp_path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(wav_bytes(sample_rate=8000, channels=2, seconds=0.5))

    output_path, duration = MeetingPipeline.decode_to_wav(source)
    output = Path(output_path)
    try:
        normalized = decode_audio_file(output)
        assert normalized.sample_rate == 16_000
        assert normalized.channels == 1
        assert len(normalized.samples) == 8_000
        assert duration == pytest.approx(0.5, abs=0.001)
    finally:
        output.unlink(missing_ok=True)
        output.parent.rmdir()


def test_decode_audio_rejects_empty_or_missing_input(tmp_path) -> None:
    with pytest.raises(AudioFormatError, match="empty"):
        decode_audio_bytes(b"")
    with pytest.raises(AudioFormatError, match="not found"):
        decode_audio_file(tmp_path / "missing.wav")


def test_audio_buffer_wav_rejects_inconsistent_channel_metadata() -> None:
    audio = AudioBuffer(
        samples=np.zeros((10, 2), dtype=np.float32),
        sample_rate=16_000,
        channels=1,
    )

    with pytest.raises(AudioFormatError, match="channel metadata"):
        audio_buffer_to_wav_bytes(audio)
