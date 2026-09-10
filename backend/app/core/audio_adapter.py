"""Decode and normalize audio for enhancement stages."""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .audio_enhancement import AudioBuffer, AudioSource


class AudioFormatError(ValueError):
    """Raised when an input cannot be decoded into the enhancement format."""


def _validate_target_rate(target_sample_rate: int) -> None:
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be greater than 0")


def _read_soundfile(source: object) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        with sf.SoundFile(source, mode="r") as handle:
            samples = handle.read(dtype="float32", always_2d=True)
            return np.asarray(samples, dtype=np.float32), int(handle.samplerate)
    except Exception as exc:
        raise AudioFormatError("audio format is not supported by libsndfile") from exc


def _decode_with_afconvert(path: Path) -> tuple[np.ndarray, int]:
    if not Path("/usr/bin/afconvert").is_file():
        raise AudioFormatError(f"cannot decode audio file: {path}")
    with tempfile.TemporaryDirectory(prefix="astra_audio_decode_") as directory:
        decoded = Path(directory) / "decoded.wav"
        try:
            subprocess.run(
                [
                    "/usr/bin/afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@16000",
                    "-c",
                    "1",
                    str(path),
                    str(decoded),
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AudioFormatError(f"cannot decode audio file: {path}") from exc
        try:
            return _read_soundfile(decoded)
        except AudioFormatError as exc:
            raise AudioFormatError(f"decoded audio is unreadable: {path}") from exc


def _normalize(
    samples: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
) -> tuple[np.ndarray, int]:
    if samples.size == 0:
        raise AudioFormatError("audio input is empty")
    if samples.ndim != 2:
        raise AudioFormatError("decoded audio must have frame and channel dimensions")

    mono = samples.mean(axis=1, dtype=np.float32)
    if sample_rate != target_sample_rate:
        mono = resample_poly(mono, target_sample_rate, sample_rate).astype(np.float32, copy=False)
    return np.asarray(mono, dtype=np.float32), target_sample_rate


def _to_buffer(
    samples: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    source: AudioSource,
) -> AudioBuffer:
    normalized, output_rate = _normalize(samples, sample_rate, target_sample_rate)
    return AudioBuffer(
        samples=normalized,
        sample_rate=output_rate,
        channels=1,
        source=source,
    )


def decode_audio_bytes(
    audio: bytes,
    *,
    filename: str = "audio.wav",
    target_sample_rate: int = 16_000,
    source: AudioSource = "upload",
) -> AudioBuffer:
    """Decode bytes into mono float32 audio at the requested sample rate."""
    _validate_target_rate(target_sample_rate)
    if not audio:
        raise AudioFormatError("audio input is empty")
    try:
        samples, sample_rate = _read_soundfile(io.BytesIO(audio))
    except AudioFormatError:
        with tempfile.TemporaryDirectory(prefix="astra_audio_input_") as directory:
            path = Path(directory) / (Path(filename).name or "audio.bin")
            path.write_bytes(audio)
            samples, sample_rate = _decode_with_afconvert(path)
    return _to_buffer(samples, sample_rate, target_sample_rate, source)


def decode_audio_file(
    path: str | Path,
    *,
    target_sample_rate: int = 16_000,
    source: AudioSource = "upload",
) -> AudioBuffer:
    """Decode a local audio file into mono float32 audio."""
    _validate_target_rate(target_sample_rate)
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise AudioFormatError(f"audio file not found: {audio_path}")
    try:
        samples, sample_rate = _read_soundfile(audio_path)
    except AudioFormatError:
        samples, sample_rate = _decode_with_afconvert(audio_path)
    return _to_buffer(samples, sample_rate, target_sample_rate, source)


def audio_buffer_to_wav_bytes(audio: AudioBuffer) -> bytes:
    """Encode an AudioBuffer as mono/stereo PCM16 WAV bytes."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.size == 0:
        raise AudioFormatError("audio input is empty")
    if samples.ndim == 1:
        if audio.channels != 1:
            raise AudioFormatError("channel metadata does not match audio samples")
    elif samples.ndim == 2:
        if samples.shape[1] != audio.channels:
            raise AudioFormatError("channel metadata does not match audio samples")
    else:
        raise AudioFormatError("audio samples must be one or two dimensional")

    try:
        import soundfile as sf

        output = io.BytesIO()
        sf.write(output, samples, audio.sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue()
    except Exception as exc:
        raise AudioFormatError("cannot encode audio as WAV") from exc
