import struct

import numpy as np

from app.core.audio_adapter import decode_audio_bytes
from app.core.pcm_protocol import (
    PCM_FRAME_HEADER,
    PCM_FRAME_MAGIC,
    PCM_FRAME_VERSION,
    PCM_MICROPHONE,
    decode_pcm_frame,
)
from app.core.session_manager import Session


def _wire_frame(sequence: int, channel: int, samples: np.ndarray) -> bytes:
    payload = np.asarray(samples, dtype="<i2").tobytes()
    return PCM_FRAME_HEADER.pack(
        PCM_FRAME_MAGIC,
        PCM_FRAME_VERSION,
        channel,
        sequence,
        16_000,
        len(samples),
    ) + payload


def test_pcm_frame_decode_and_session_pairing_fills_missing_reference_frames() -> None:
    session = Session(max_audio_bytes=10_000)
    session.start()
    assert session.configure_pcm(sample_rate=16_000, frame_samples=160)

    microphone = _wire_frame(1, PCM_MICROPHONE, np.full(160, 1000, dtype=np.int16))
    microphone_next = _wire_frame(2, PCM_MICROPHONE, np.full(160, 1000, dtype=np.int16))
    reference = _wire_frame(2, 1, np.full(160, 500, dtype=np.int16))
    assert session.append_pcm_frame(decode_pcm_frame(microphone))
    assert session.append_pcm_frame(decode_pcm_frame(microphone_next))
    assert session.append_pcm_frame(decode_pcm_frame(reference))

    audio, far_end = session.take_pcm_pair()
    assert far_end is not None
    decoded_audio = decode_audio_bytes(audio)
    decoded_reference = decode_audio_bytes(far_end, source="reference")
    assert len(decoded_audio.samples) == 320
    assert len(decoded_reference.samples) == 320
    assert float(decoded_audio.samples[0]) == np.float32(1000 / 32768)
    assert float(decoded_reference.samples[0]) == 0.0
    assert float(decoded_reference.samples[-1]) == np.float32(500 / 32768)


def test_pcm_frame_decoder_rejects_payload_length_mismatch() -> None:
    frame = PCM_FRAME_HEADER.pack(
        PCM_FRAME_MAGIC, PCM_FRAME_VERSION, PCM_MICROPHONE, 1, 16_000, 160
    ) + b"\x00\x00"
    try:
        decode_pcm_frame(frame)
    except ValueError as exc:
        assert "payload length" in str(exc)
    else:
        raise AssertionError("expected malformed PCM frame to be rejected")
