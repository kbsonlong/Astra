from dataclasses import dataclass, field
from uuid import uuid4


State = str


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    state: State = "IDLE"
    generation_id: int = 0
    cancelled_generations: set[int] = field(default_factory=set)
    audio_buffer: bytearray = field(default_factory=bytearray)
    reference_buffer: bytearray = field(default_factory=bytearray)
    history: list[dict[str, str]] = field(default_factory=list)
    max_audio_bytes: int = 25 * 1024 * 1024

    def start(self) -> None:
        self.audio_buffer.clear()
        self.reference_buffer.clear()
        self.state = "LISTENING"

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

    def take_audio(self) -> bytes:
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio

    def take_reference(self) -> bytes | None:
        if not self.reference_buffer:
            return None
        reference = bytes(self.reference_buffer)
        self.reference_buffer.clear()
        return reference

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
        self.audio_buffer.clear()
        self.reference_buffer.clear()

    def accepts(self, generation_id: int) -> bool:
        return generation_id == self.generation_id and generation_id not in self.cancelled_generations
