from dataclasses import dataclass, field
from uuid import uuid4

from .audio_adapter import pcm16_frames_to_wav_bytes
from .pcm_protocol import PcmFrame


State = str


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    state: State = "IDLE"
    generation_id: int = 0
    cancelled_generations: set[int] = field(default_factory=set)
    audio_buffer: bytearray = field(default_factory=bytearray)
    reference_buffer: bytearray = field(default_factory=bytearray)
    pcm_mode: bool = False
    pcm_sample_rate: int = 16_000
    pcm_frame_samples: int = 160
    pcm_microphone_frames: dict[int, bytes] = field(default_factory=dict)
    pcm_reference_frames: dict[int, bytes] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    max_audio_bytes: int = 25 * 1024 * 1024

    def start(self) -> None:
        self._clear_audio()
        self.pcm_mode = False
        self.state = "LISTENING"

    def configure_pcm(self, *, sample_rate: int, frame_samples: int) -> bool:
        if self.state != "LISTENING":
            return False
        if sample_rate != 16_000 or frame_samples != 160:
            return False
        if (
            self.audio_buffer
            or self.reference_buffer
            or self.pcm_microphone_frames
            or self.pcm_reference_frames
        ):
            return False
        self.pcm_mode = True
        self.pcm_sample_rate = sample_rate
        self.pcm_frame_samples = frame_samples
        return True

    def append_audio(self, chunk: bytes, *, source: str = "microphone") -> bool:
        """Append one audio frame, rejecting an utterance that exceeds its budget."""
        if self.state != "LISTENING":
            return True
        used_bytes = len(self.audio_buffer) + len(self.reference_buffer)
        if len(chunk) > self.max_audio_bytes - used_bytes:
            self.audio_buffer.clear()
            self.reference_buffer.clear()
            return False
        if source == "reference":
            self.reference_buffer.extend(chunk)
        else:
            self.audio_buffer.extend(chunk)
        return True

    def append_pcm_frame(self, frame: PcmFrame) -> bool:
        """Append a validated PCM frame while preserving its sequence number."""
        if self.state not in {"LISTENING", "SPEAKING"}:
            return True
        if (
            not self.pcm_mode
            or frame.sample_rate != self.pcm_sample_rate
            or frame.sample_count != self.pcm_frame_samples
        ):
            return False
        if frame.channel == "microphone" and self.state != "LISTENING":
            return True
        frames = (
            self.pcm_reference_frames
            if frame.channel == "reference"
            else self.pcm_microphone_frames
        )
        if frame.sequence in frames:
            return True
        used_bytes = self._audio_bytes()
        if len(frame.payload) > self.max_audio_bytes - used_bytes:
            self._clear_audio()
            return False
        frames[frame.sequence] = frame.payload
        return True

    def take_audio(self) -> bytes:
        if self.pcm_mode:
            audio, _ = self.take_pcm_pair()
            return audio
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio

    def take_reference(self) -> bytes | None:
        if self.pcm_mode:
            _, reference = self.take_pcm_pair()
            return reference
        if not self.reference_buffer:
            return None
        reference = bytes(self.reference_buffer)
        self.reference_buffer.clear()
        return reference

    def take_pcm_pair(self) -> tuple[bytes, bytes | None]:
        sequences = sorted(self.pcm_microphone_frames)
        if not sequences:
            self._clear_audio()
            return b"", None
        audio = pcm16_frames_to_wav_bytes(
            self.pcm_microphone_frames,
            sequences,
            sample_rate=self.pcm_sample_rate,
            frame_samples=self.pcm_frame_samples,
        )
        reference = None
        if any(sequence in self.pcm_reference_frames for sequence in sequences):
            reference = pcm16_frames_to_wav_bytes(
                self.pcm_reference_frames,
                sequences,
                sample_rate=self.pcm_sample_rate,
                frame_samples=self.pcm_frame_samples,
            )
        self._clear_audio()
        return audio, reference

    def speech_end(self) -> int:
        self.generation_id += 1
        self.state = "REASONING"
        return self.generation_id

    def interrupt(self, generation_id: int | None, reason: str) -> bool:
        target = self.generation_id if generation_id is None else generation_id
        if target != self.generation_id or target == 0:
            return False
        self.cancelled_generations.add(target)
        self.state = "LISTENING" if reason == "vad" else "IDLE"
        return True

    def end(self) -> None:
        self.state = "IDLE"
        self._clear_audio()

    def _audio_bytes(self) -> int:
        return (
            len(self.audio_buffer)
            + len(self.reference_buffer)
            + sum(len(item) for item in self.pcm_microphone_frames.values())
            + sum(len(item) for item in self.pcm_reference_frames.values())
        )

    def _clear_audio(self) -> None:
        self.audio_buffer.clear()
        self.reference_buffer.clear()
        self.pcm_microphone_frames.clear()
        self.pcm_reference_frames.clear()

    def accepts(self, generation_id: int) -> bool:
        return generation_id == self.generation_id and generation_id not in self.cancelled_generations
