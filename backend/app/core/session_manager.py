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

    def start(self) -> None:
        self.state = "LISTENING"

    def append_audio(self, chunk: bytes) -> None:
        if self.state == "LISTENING":
            self.audio_buffer.extend(chunk)

    def take_audio(self) -> bytes:
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio

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

    def accepts(self, generation_id: int) -> bool:
        return generation_id == self.generation_id and generation_id not in self.cancelled_generations
