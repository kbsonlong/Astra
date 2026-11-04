import asyncio
import io
import re
import wave
from collections.abc import Awaitable, Callable, Mapping, Sequence

from typing import Protocol

from .audio_adapter import audio_buffer_to_wav_bytes, decode_audio_bytes
from .audio_enhancement import AudioEnhancementPipeline, EnhancementContext
from ..models.llm_client import OpenAICompatLLMClient
from ..models.tts_client import PiperSdkTtsClient
from .tts_voice_store import TtsVoiceSnapshot
from .pcm_protocol import TtsAudioFrame, encode_tts_frame
from ..schemas.ws import (
    AsrFinalEvent,
    EnhancementStatusEvent,
    LlmTokenEvent,
    TtsEndEvent,
    TtsStartEvent,
)


class ASRClient(Protocol):
    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str: ...


# An emit target accepts either a JSON control event (dict) or a binary audio
# frame (bytes). TTS audio travels as bytes; everything else stays JSON.
Emit = Callable[[dict[str, object] | bytes], Awaitable[None]]
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


class VoicePipeline:
    def __init__(
        self,
        asr: ASRClient,
        llm: OpenAICompatLLMClient,
        tts: PiperSdkTtsClient,
        enhancement: AudioEnhancementPipeline | None = None,
        enhancement_max_queue: int = 2,
        enhancement_frame_timeout_ms: float = 80.0,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.enhancement = enhancement
        self.enhancement_max_queue = enhancement_max_queue
        self.enhancement_frame_timeout_ms = enhancement_frame_timeout_ms

    async def run(
        self,
        audio: bytes,
        messages: Sequence[Mapping[str, str]],
        generation_id: int,
        emit: Emit,
        reference: bytes | None = None,
        voice_snapshot: TtsVoiceSnapshot | None = None,
    ) -> tuple[str, str]:
        """一轮语音对话。返回 (user_asr_text, assistant_reply)。

        ``messages`` 是历史上下文（不含本轮 ASR 文本）——调用方负责
        在成功后把 user/assistant 两条追加进会话历史；失败/打断的轮次
        不应写入，避免上下文被半截回复污染。
        """
        if self.enhancement is not None:
            microphone = await asyncio.to_thread(
                decode_audio_bytes, audio, filename="microphone.wav", source="microphone"
            )
            far_end = None
            if reference:
                far_end = await asyncio.to_thread(
                    decode_audio_bytes,
                    reference,
                    filename="far-end-reference.wav",
                    source="reference",
                )
            microphone, enhancement_metrics = await self.enhancement.process_realtime(
                microphone,
                EnhancementContext(
                    reference=far_end,
                    realtime=True,
                ),
                max_queue=self.enhancement_max_queue,
                frame_timeout_ms=self.enhancement_frame_timeout_ms,
            )
            await emit(
                EnhancementStatusEvent(
                    generation_id=generation_id,
                    stages=[item.to_dict() for item in enhancement_metrics],
                ).to_wire()
            )
            if any(item.status == "applied" for item in enhancement_metrics):
                audio = audio_buffer_to_wav_bytes(microphone)

        text = await self.asr.transcribe(audio)
        await emit(
            AsrFinalEvent(generation_id=generation_id, text=text).to_wire()
        )

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
            await emit(
                LlmTokenEvent(generation_id=generation_id, token=token).to_wire()
            )
            sentence_buffer += token
            parts = _SENTENCE_END.split(sentence_buffer)
            sentence_buffer = parts.pop()
            for sentence in parts:
                sequence = await self._synthesize(
                    sentence.strip(), generation_id, sequence, emit, voice_snapshot
                )

        if sentence_buffer.strip():
            await self._synthesize(
                sentence_buffer.strip(), generation_id, sequence, emit, voice_snapshot
            )
        await emit(TtsEndEvent(generation_id=generation_id).to_wire())
        return text, "".join(reply_parts).strip()

    async def _synthesize(
        self,
        sentence: str,
        generation_id: int,
        sequence: int,
        emit: Emit,
        voice_snapshot: TtsVoiceSnapshot | None = None,
    ) -> int:
        if not sentence:
            return sequence
        if voice_snapshot is None:
            audio = await self.tts.synthesize(sentence)
        else:
            synthesize_snapshot = getattr(self.tts, "synthesize_snapshot", None)
            if not callable(synthesize_snapshot):
                raise RuntimeError("configured TTS backend does not support voice snapshots")
            audio = await synthesize_snapshot(sentence, voice_snapshot)
        await emit(
            TtsStartEvent(generation_id=generation_id, seq=sequence).to_wire()
        )
        # Audio goes out as a compact binary frame (header + WAV) instead of
        # base64-in-JSON; the header carries generation_id/seq for gating.
        await emit(
            encode_tts_frame(
                TtsAudioFrame(
                    generation_id=generation_id,
                    sequence=sequence,
                    sample_rate=_wav_sample_rate(audio),
                    payload=audio,
                )
            )
        )
        return sequence + 1


def _wav_sample_rate(audio: bytes) -> int:
    """从 TTS 返回的 WAV 读取实际采样率；非 WAV 保持 Piper 兼容默认值。"""
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            rate = handle.getframerate()
            return rate if rate > 0 else 22050
    except wave.Error:
        return 22050
