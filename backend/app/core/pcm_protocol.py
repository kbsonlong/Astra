"""Binary protocol for realtime 16 kHz PCM audio frames."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal


PCM_FRAME_MAGIC = b"ASTR"
PCM_FRAME_VERSION = 1
PCM_MICROPHONE = 0
PCM_REFERENCE = 1
PCM_FRAME_HEADER = struct.Struct("<4sBBIHH")


@dataclass(frozen=True)
class PcmFrame:
    channel: Literal["microphone", "reference"]
    sequence: int
    sample_rate: int
    sample_count: int
    payload: bytes


def decode_pcm_frame(data: bytes) -> PcmFrame:
    if len(data) < PCM_FRAME_HEADER.size:
        raise ValueError("PCM frame header is truncated")
    magic, version, channel, sequence, sample_rate, sample_count = PCM_FRAME_HEADER.unpack(
        data[: PCM_FRAME_HEADER.size]
    )
    if magic != PCM_FRAME_MAGIC or version != PCM_FRAME_VERSION:
        raise ValueError("unsupported PCM frame header")
    if channel not in {PCM_MICROPHONE, PCM_REFERENCE}:
        raise ValueError("unsupported PCM frame channel")
    if sample_rate <= 0 or sample_count <= 0:
        raise ValueError("PCM frame format is invalid")
    payload = data[PCM_FRAME_HEADER.size :]
    if len(payload) != sample_count * 2:
        raise ValueError("PCM frame payload length does not match sample count")
    return PcmFrame(
        channel="microphone" if channel == PCM_MICROPHONE else "reference",
        sequence=sequence,
        sample_rate=sample_rate,
        sample_count=sample_count,
        payload=payload,
    )


# --- Downstream TTS audio frames ------------------------------------------
# TTS audio previously travelled as base64 inside a JSON ``tts_chunk`` event,
# inflating payloads ~33% and forcing a decode per frame. It now uses a compact
# binary frame mirroring the upstream ``ASTR`` design. The header carries the
# generation id and sequence so the client can drop audio from an interrupted
# turn, exactly as the JSON ``generation_id``/``seq`` fields did.
TTS_FRAME_MAGIC = b"ATTS"
TTS_FRAME_VERSION = 1
# magic(4s) version(B) reserved(B) generation_id(I) sequence(I) sample_rate(I)
TTS_FRAME_HEADER = struct.Struct("<4sBBIII")


@dataclass(frozen=True)
class TtsAudioFrame:
    generation_id: int
    sequence: int
    sample_rate: int
    payload: bytes


def encode_tts_frame(frame: TtsAudioFrame) -> bytes:
    """Serialize one downstream TTS audio frame (header + WAV payload)."""
    if frame.generation_id < 0 or frame.sequence < 0:
        raise ValueError("generation_id and sequence must be non-negative")
    if frame.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    header = TTS_FRAME_HEADER.pack(
        TTS_FRAME_MAGIC,
        TTS_FRAME_VERSION,
        0,
        frame.generation_id,
        frame.sequence,
        frame.sample_rate,
    )
    return header + frame.payload


def decode_tts_frame(data: bytes) -> TtsAudioFrame:
    if len(data) < TTS_FRAME_HEADER.size:
        raise ValueError("TTS frame header is truncated")
    magic, version, _reserved, generation_id, sequence, sample_rate = (
        TTS_FRAME_HEADER.unpack(data[: TTS_FRAME_HEADER.size])
    )
    if magic != TTS_FRAME_MAGIC or version != TTS_FRAME_VERSION:
        raise ValueError("unsupported TTS frame header")
    if sample_rate <= 0:
        raise ValueError("TTS frame sample_rate is invalid")
    return TtsAudioFrame(
        generation_id=generation_id,
        sequence=sequence,
        sample_rate=sample_rate,
        payload=data[TTS_FRAME_HEADER.size :],
    )
