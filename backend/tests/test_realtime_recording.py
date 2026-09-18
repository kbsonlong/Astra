import io
import os
import time
import wave
from uuid import uuid4

from app.core.realtime_recording import (
    RecordingLimitExceeded,
    RealtimeRecording,
    clean_expired_realtime_recordings,
)


def test_realtime_recording_streams_pcm_to_wav_without_buffering_all_audio(tmp_path) -> None:
    recording = RealtimeRecording(tmp_path, str(uuid4()), max_bytes=8)
    recording.append(b"\x01\x00\x02\x00")
    recording.append(b"\x03\x00\x04\x00")

    wav_path = recording.finish()

    assert wav_path is not None
    assert not recording.pcm_path.exists()
    with wave.open(str(wav_path), "rb") as output:
        assert output.getframerate() == 16_000
        assert output.getnframes() == 4
        assert output.readframes(4) == b"\x01\x00\x02\x00\x03\x00\x04\x00"


def test_realtime_recording_keeps_written_prefix_when_at_limit(tmp_path) -> None:
    recording = RealtimeRecording(tmp_path, str(uuid4()), max_bytes=4)
    recording.append(b"\x01\x00\x02\x00")

    try:
        recording.append(b"\x03\x00")
    except RecordingLimitExceeded:
        pass
    else:  # pragma: no cover - test failure guard
        raise AssertionError("recording limit was not enforced")

    assert recording.pcm_path.read_bytes() == b"\x01\x00\x02\x00"


def test_realtime_recording_cleanup_is_scoped_to_stale_recording_files(tmp_path) -> None:
    stale = tmp_path / f"{uuid4()}.pcm"
    retained = tmp_path / "notes.txt"
    stale.write_bytes(b"pcm")
    retained.write_text("keep", encoding="utf-8")
    old = time.time() - 8 * 24 * 60 * 60
    os.utime(stale, (old, old))

    assert clean_expired_realtime_recordings(tmp_path, retention_days=7) == 1
    assert not stale.exists()
    assert retained.exists()
