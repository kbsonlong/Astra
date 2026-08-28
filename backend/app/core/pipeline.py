import base64
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence

from ..models.asr_client import MlxAudioAsrClient
from ..models.llm_client import OpenAICompatLLMClient
from ..models.tts_client import PiperSdkTtsClient


Emit = Callable[[dict[str, object]], Awaitable[None]]
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


class VoicePipeline:
    def __init__(
        self,
        asr: MlxAudioAsrClient,
        llm: OpenAICompatLLMClient,
        tts: PiperSdkTtsClient,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts

    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit: Emit,
    ) -> None:
        text = await self.asr.transcribe(audio)
        await emit({"type": "asr_final", "text": text, "generation_id": generation_id})

        sentence_buffer = ""
        sequence = 0
        async for token in self.llm.stream_chat(messages):
            await emit({"type": "llm_token", "token": token, "generation_id": generation_id})
            sentence_buffer += token
            parts = _SENTENCE_END.split(sentence_buffer)
            sentence_buffer = parts.pop()
            for sentence in parts:
                sequence = await self._synthesize(sentence.strip(), generation_id, sequence, emit)

        if sentence_buffer.strip():
            await self._synthesize(sentence_buffer.strip(), generation_id, sequence, emit)
        await emit({"type": "tts_end", "generation_id": generation_id})

    async def _synthesize(
        self,
        sentence: str,
        generation_id: int,
        sequence: int,
        emit: Emit,
    ) -> int:
        if not sentence:
            return sequence
        audio = await self.tts.synthesize(sentence)
        await emit({"type": "tts_start", "generation_id": generation_id, "seq": sequence})
        await emit(
            {
                "type": "tts_chunk",
                "generation_id": generation_id,
                "seq": sequence,
                "mime": "audio/wav",
                "sample_rate": 22050,
                "audio_b64": base64.b64encode(audio).decode("ascii"),
            }
        )
        return sequence + 1
