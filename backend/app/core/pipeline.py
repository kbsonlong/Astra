import base64
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence

from typing import Protocol

from ..models.llm_client import OpenAICompatLLMClient
from ..models.tts_client import PiperSdkTtsClient


class ASRClient(Protocol):
    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str: ...


Emit = Callable[[dict[str, object]], Awaitable[None]]
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


class VoicePipeline:
    def __init__(
        self,
        asr: ASRClient,
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
    ) -> tuple[str, str]:
        """一轮语音对话。返回 (user_asr_text, assistant_reply)。

        ``messages`` 是历史上下文（不含本轮 ASR 文本）——调用方负责
        在成功后把 user/assistant 两条追加进会话历史；失败/打断的轮次
        不应写入，避免上下文被半截回复污染。
        """
        text = await self.asr.transcribe(audio)
        await emit({"type": "asr_final", "text": text, "generation_id": generation_id})

        history = list(messages)
        history.append({"role": "user", "content": text})

        sentence_buffer = ""
        sequence = 0
        reply_parts: list[str] = []
        async for token in self.llm.stream_chat(
            history,
            chat_template_kwargs={"enable_thinking": False},
        ):
            reply_parts.append(token)
            await emit({"type": "llm_token", "token": token, "generation_id": generation_id})
            sentence_buffer += token
            parts = _SENTENCE_END.split(sentence_buffer)
            sentence_buffer = parts.pop()
            for sentence in parts:
                sequence = await self._synthesize(sentence.strip(), generation_id, sequence, emit)

        if sentence_buffer.strip():
            await self._synthesize(sentence_buffer.strip(), generation_id, sequence, emit)
        await emit({"type": "tts_end", "generation_id": generation_id})
        return text, "".join(reply_parts).strip()

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
