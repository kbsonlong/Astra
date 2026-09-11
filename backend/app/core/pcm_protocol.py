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
