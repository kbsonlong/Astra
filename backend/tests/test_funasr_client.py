import wave

import pytest

from app.core.workflow import AudioWorkflow, Segment
from app.models.funasr_client import (
    FunAsrClient,
    FunAsrClientError,
    normalize_funasr_segments,
)


def test_normalize_prefers_sentence_info_with_speakers() -> None:
    res = [
        {
            "language": "zh",
            "sentence_info": [
                {"text": "<|zh|><|NEUTRAL|>大家好", "start": 0, "end": 1500, "spk": 0},
                {"text": "你好呀", "start": 1600, "end": 3000, "spk": 1},
            ],
        }
    ]
    language, segments = normalize_funasr_segments(res)
    assert language == "zh"
    assert len(segments) == 2
    assert segments[0].text == "大家好"  # 富标签被清洗
    assert segments[0].start == 0.0 and segments[0].end == 1.5  # ms -> s
    assert segments[0].speaker == "Speaker 1"  # spk 0 -> Speaker 1
    assert segments[0].speaker_confidence == "inline"
    assert segments[1].speaker == "Speaker 2"


def test_normalize_falls_back_to_text_and_timestamp() -> None:
    res = [{"text": "<|en|>hello world", "timestamp": [[0, 500], [500, 1200]]}]
    language, segments = normalize_funasr_segments(res)
    assert language is None
    assert len(segments) == 1
    assert segments[0].text == "hello world"
    assert segments[0].start == 0.0
    assert segments[0].end == 1.2


def test_normalize_handles_empty_and_bad_input() -> None:
    assert normalize_funasr_segments([]) == (None, [])
    assert normalize_funasr_segments([{"sentence_info": []}]) == (None, [])
    lang, segs = normalize_funasr_segments("just a string")
    assert lang is None and segs[0].text == "just a string"


def test_capabilities_and_availability() -> None:
    client = FunAsrClient("iic/SenseVoiceSmall", diarize=True)
    caps = client.capabilities()
    assert caps["engine"] == "funasr"
    assert caps["diarize"] is True
    assert caps["inline_diarization"] is True
    assert "zh" in caps["languages"] and "yue" in caps["languages"]
    assert client.diarizes_inline() is True

    # diarize=False -> 不声明内联分离
    plain = FunAsrClient("iic/SenseVoiceSmall", diarize=False)
    assert plain.diarizes_inline() is False
    assert plain.capabilities()["diarize"] is False

    # 未配置模型 -> 不可用, 原因不含本地路径
    empty = FunAsrClient("")
    ok, reason = empty.is_available()
    assert ok is False
    assert "ASR_FUNASR_MODEL" in reason

    # 注入 model_instance -> 可用
    injected = FunAsrClient("m", model_instance=object())
    assert injected.is_available() == (True, "ready")


class _FakeFunModel:
    def generate(self, **kwargs):
        return [
            {
                "language": "zh",
                "sentence_info": [
                    {"text": "第一句", "start": 0, "end": 1000, "spk": 0},
                    {"text": "第二句", "start": 1000, "end": 2000, "spk": 1},
                ],
            }
        ]


@pytest.mark.anyio
async def test_transcribe_segments_via_injected_model() -> None:
    client = FunAsrClient("m", model_instance=_FakeFunModel())
    language, segments = await client.transcribe_segments("/tmp/x.wav")
    assert language == "zh"
    assert [s.text for s in segments] == ["第一句", "第二句"]
    assert segments[0].speaker == "Speaker 1"
    assert segments[1].speaker == "Speaker 2"


@pytest.mark.anyio
async def test_transcribe_segments_wraps_errors() -> None:
    class Boom:
        def generate(self, **kwargs):
            raise RuntimeError("funasr exploded")

    client = FunAsrClient("m", model_instance=Boom())
    with pytest.raises(FunAsrClientError, match="funasr transcription failed"):
        await client.transcribe_segments("/tmp/x.wav")


@pytest.mark.anyio
async def test_transcribe_asrstage_compat_joins_text() -> None:
    client = FunAsrClient("m", model_instance=_FakeFunModel())
    text = await client.transcribe(b"fake-wav-bytes", filename="chunk.wav")
    assert text == "第一句 第二句"

    with pytest.raises(FunAsrClientError, match="empty"):
        await client.transcribe(b"")


# ── workflow 内联分支 ──────────────────────────────────────────────────


class _InlineASR:
    """自带内联分离的 fake ASR, 供 AudioWorkflow inline 分支测试。"""

    def __init__(self) -> None:
        self.called_with: str | None = None

    def diarizes_inline(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True

    async def transcribe_segments(self, wav: str, *, filename: str = "speech.wav"):
        self.called_with = wav
        return "zh", [
            Segment(0.0, 1.0, "大家好", speaker="Speaker 1", speaker_confidence="inline"),
            Segment(1.0, 2.0, "你好", speaker="Speaker 2", speaker_confidence="inline"),
        ]


class _ExplodingVAD:
    async def detect(self, wav):
        raise AssertionError("VAD must not run in inline-diarization branch")


class _ExplodingASR:
    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        raise AssertionError("chunk ASR must not run in inline-diarization branch")


class _Punct:
    async def restore(self, text: str, *, language: str = "zh") -> str:
        return f"{text}。"

    def is_ready(self) -> bool:
        return True


@pytest.mark.anyio
async def test_audio_workflow_uses_inline_branch_and_materializes_audio(tmp_path) -> None:
    inline = _InlineASR()
    wav = tmp_path / "meeting.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0" * 16_000 * 2 * 2)
    workflow = AudioWorkflow(
        _ExplodingVAD(),
        _ExplodingASR(),
        _Punct(),
        diarization=None,
        inline_asr=inline,
        speaker_store=None,  # 无 store -> 保留 Speaker N 标签
    )
    result = await workflow.run(str(wav), filename=wav.name)

    # 内联分支执行: VAD/chunk-ASR 均未触发(否则 AssertionError)
    assert inline.called_with == str(wav)
    assert result.language == "zh"
    assert [s.text for s in result.segments] == ["大家好。", "你好。"]
    assert result.segments[0].speaker == "Speaker 1"
    assert result.metadata.get("diarized_inline") is True
    # 内联段按时间戳从源 WAV 提取，供会议审校和 Qwen3 微调导出使用。
    assert all(segment.audio and segment.audio[:4] == b"RIFF" for segment in result.segments)

    # stage_status: asr 阶段存在, sd 阶段不再单列为 disabled
    status = workflow.stage_status()
    assert "asr" in status
    assert "segment_audio" in status
    assert "speaker_registry" in status


class _PlainFunASR(_InlineASR):
    """关闭 cam++ 后仍应作为整段 FunASR 后端执行。"""

    def diarizes_inline(self) -> bool:
        return False

    async def transcribe_segments(self, wav: str, *, filename: str = "speech.wav"):
        self.called_with = wav
        return "zh", [Segment(0.0, 1.0, "仅转写，不分离")]


@pytest.mark.anyio
async def test_audio_workflow_uses_funasr_when_inline_diarization_disabled(tmp_path) -> None:
    wav = tmp_path / "plain.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0" * 16_000 * 2)
    plain = _PlainFunASR()
    workflow = AudioWorkflow(
        _ExplodingVAD(),
        _ExplodingASR(),
        _Punct(),
        diarization=None,
        inline_asr=plain,
    )
    result = await workflow.run(str(wav), filename=wav.name)

    assert plain.called_with == str(wav)
    assert [segment.text for segment in result.segments] == ["仅转写，不分离。"]
    assert result.metadata["diarized_inline"] is False
    assert result.segments[0].audio and result.segments[0].audio[:4] == b"RIFF"
    assert workflow.stage_status()["sd"] == {"enabled": False, "ok": True}


@pytest.mark.anyio
async def test_audio_workflow_default_branch_when_asr_not_inline() -> None:
    """inline_asr 未提供时走原有 VAD->ASR 分支 (回归保护)。"""
    from app.core.workflow import SpeechChunk

    class _VAD:
        async def detect(self, wav):
            return [SpeechChunk(0.0, 1.0, b"hi")]

    class _ASR:
        async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
            return audio.decode()

    workflow = AudioWorkflow(_VAD(), _ASR(), _Punct(), diarization=None)
    result = await workflow.run("/tmp/x.wav", filename="x.wav")
    assert [s.text for s in result.segments] == ["hi。"]
    assert result.metadata.get("diarized_inline") is None
