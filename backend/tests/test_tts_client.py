import asyncio
import io
import threading
import time
import wave

import pytest

from app.models.tts_client import (
    CosyVoiceTtsClient,
    IndexTTS25MlxClient,
    MlxAudioTtsClient,
    PiperSdkTtsClient,
    TTSClientError,
)
from app.core.tts_voice_store import TtsVoiceSnapshot


@pytest.mark.anyio
async def test_synthesize_uses_piper_sdk() -> None:
    class FakeVoice:
        def synthesize_wav(self, text: str, output) -> None:
            assert text == "hello"
            output.write(b"RIFFfake-wav")

    client = PiperSdkTtsClient("test.onnx", voice=FakeVoice())

    assert await client.synthesize("hello") == b"RIFFfake-wav"


@pytest.mark.anyio
async def test_synthesize_rejects_empty_text() -> None:
    client = PiperSdkTtsClient("test.onnx", voice=object())
    try:
        with pytest.raises(TTSClientError, match="empty"):
            await client.synthesize("  ")
    finally:
        pass


class _FakeResult:
    def __init__(self, audio, sample_rate=None):
        self.audio = audio
        if sample_rate is not None:
            self.sample_rate = sample_rate


class _FakeMlxModel:
    sample_rate = 24000

    def __init__(self, audio, calls=None):
        self._audio = audio
        self._calls = calls

    def generate(self, text, *, voice, lang_code, speed):
        if self._calls is not None:
            self._calls.append(
                {"text": text, "voice": voice, "lang_code": lang_code, "speed": speed}
            )
        yield _FakeResult(self._audio)


def _read_wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sampwidth": handle.getsampwidth(),
            "framerate": handle.getframerate(),
            "frames": handle.getnframes(),
        }


@pytest.mark.anyio
async def test_mlx_synthesize_returns_wav_bytes() -> None:
    calls: list[dict] = []
    model = _FakeMlxModel([0.0, 0.5, -0.5, 1.0, -1.0], calls=calls)
    client = MlxAudioTtsClient(
        "mlx-community/Kokoro-82M-4bit",
        voice="zf_xiaobei",
        lang_code="z",
        speed=1.0,
        model_instance=model,
    )

    assert client.is_ready() is True
    audio = await client.synthesize("你好，世界")

    meta = _read_wav(audio)
    assert audio[:4] == b"RIFF"
    assert meta["channels"] == 1
    assert meta["sampwidth"] == 2
    assert meta["framerate"] == 24000
    assert meta["frames"] == 5
    # 合成参数正确透传给 model.generate
    assert calls == [
        {"text": "你好，世界", "voice": "zf_xiaobei", "lang_code": "z", "speed": 1.0}
    ]


@pytest.mark.anyio
async def test_mlx_synthesize_uses_injected_loader() -> None:
    loaded: list[str] = []

    def loader(model_id: str):
        loaded.append(model_id)
        return _FakeMlxModel([0.1, -0.1])

    MlxAudioTtsClient.clear_model_cache()
    client = MlxAudioTtsClient(
        "mlx-community/Kokoro-82M-4bit", model_loader=loader
    )

    audio = await client.synthesize("测试")
    assert audio[:4] == b"RIFF"
    assert loaded == ["mlx-community/Kokoro-82M-4bit"]
    MlxAudioTtsClient.clear_model_cache()


@pytest.mark.anyio
async def test_mlx_synthesize_rejects_empty_text() -> None:
    client = MlxAudioTtsClient("m", model_instance=_FakeMlxModel([0.1]))
    with pytest.raises(TTSClientError, match="empty"):
        await client.synthesize("   ")


@pytest.mark.anyio
async def test_mlx_synthesize_wraps_generate_errors() -> None:
    class _BoomModel:
        sample_rate = 24000

        def generate(self, text, *, voice, lang_code, speed):
            raise RuntimeError("mlx exploded")

    client = MlxAudioTtsClient("m", model_instance=_BoomModel())
    with pytest.raises(TTSClientError, match="mlx-audio synthesis failed"):
        await client.synthesize("你好")


@pytest.mark.anyio
async def test_mlx_synthesize_errors_on_no_samples() -> None:
    class _EmptyModel:
        sample_rate = 24000

        def generate(self, text, *, voice, lang_code, speed):
            return iter(())

    client = MlxAudioTtsClient("m", model_instance=_EmptyModel())
    with pytest.raises(TTSClientError):
        await client.synthesize("你好")


def test_piper_capabilities_and_availability() -> None:
    client = PiperSdkTtsClient("voice.onnx")
    caps = client.capabilities()
    assert caps["engine"] == "piper"
    assert caps["clone"] is False
    assert "zh" in caps["languages"]
    # 注入 voice 时立即可用
    ready = PiperSdkTtsClient("voice.onnx", voice=object())
    available, reason = ready.is_available()
    assert available is True and reason == "ready"
    # 未配置路径时不可用, 原因不含路径
    empty = PiperSdkTtsClient("")
    ok, why = empty.is_available()
    assert ok is False
    assert "TTS_MODEL_PATH" in why


def test_mlx_tts_capabilities_and_availability() -> None:
    model = _FakeMlxModel([0.1])
    client = MlxAudioTtsClient("mlx-community/Kokoro-82M-4bit", model_instance=model)
    caps = client.capabilities()
    assert caps["engine"] == "mlx_audio"
    assert caps["model"] == "mlx-community/Kokoro-82M-4bit"
    assert "zh" in caps["languages"]
    # 注入 model_instance -> 可用
    available, reason = client.is_available()
    assert available is True and reason == "ready"
    # 未配置 model -> 不可用
    no_model = MlxAudioTtsClient("", model_instance=None)
    ok, why = no_model.is_available()
    assert ok is False
    assert "TTS_MLX_MODEL" in why

@pytest.mark.anyio
async def test_mlx_synthesis_runs_on_dedicated_worker_without_blocking_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_ids: list[int] = []

    class _SlowModel(_FakeMlxModel):
        def generate(self, text, *, voice, lang_code, speed):
            worker_ids.append(threading.get_ident())
            time.sleep(0.12)
            yield _FakeResult([0.1, -0.1])

    monkeypatch.setattr(
        MlxAudioTtsClient,
        "_load_model_from_sdk",
        staticmethod(lambda model: _SlowModel([0.1, -0.1])),
    )
    MlxAudioTtsClient.clear_model_cache()
    client = MlxAudioTtsClient("fake-production-model")
    try:
        started = time.perf_counter()
        synthesis = asyncio.create_task(client.synthesize("测试"))
        await asyncio.sleep(0.01)
        # 若 production generate 在事件循环执行，此处会被 120ms sleep 拖住。
        assert time.perf_counter() - started < 0.08
        await synthesis
        assert worker_ids and worker_ids[0] != threading.get_ident()
    finally:
        await client.aclose()
        MlxAudioTtsClient.clear_model_cache()


@pytest.mark.anyio
async def test_cosyvoice_zero_shot_synthesis_emits_wav_and_language_tag() -> None:
    calls: list[dict] = []

    class _FakeCosyVoice:
        sample_rate = 22_050

        def inference_zero_shot(self, text, prompt_text, prompt, *, stream, speed):
            calls.append(
                {
                    "text": text,
                    "prompt_text": prompt_text,
                    "prompt": prompt,
                    "stream": stream,
                    "speed": speed,
                }
            )
            return iter(({"tts_speech": [[0.0, 0.5, -0.5]]},))

    client = CosyVoiceTtsClient(
        "CosyVoice2-0.5B",
        prompt_wav="reference.wav",
        prompt_text="这是一段参考音色。",
        language="yue",
        speed=1.1,
        model_instance=_FakeCosyVoice(),
        prompt_loader=lambda path: f"loaded:{path}",
    )
    try:
        audio = await client.synthesize("你好")
    finally:
        await client.aclose()

    assert _read_wav(audio) == {
        "channels": 1,
        "sampwidth": 2,
        "framerate": 22_050,
        "frames": 3,
    }
    assert calls == [
        {
            "text": "<|yue|>你好",
            "prompt_text": "这是一段参考音色。",
            "prompt": "loaded:reference.wav",
            "stream": False,
            "speed": 1.1,
        }
    ]


def test_cosyvoice_capabilities_and_missing_prompt_are_explicit() -> None:
    client = CosyVoiceTtsClient(
        "CosyVoice2-0.5B", prompt_wav="", prompt_text="", model_instance=object()
    )
    capabilities = client.capabilities()
    assert capabilities["clone"] is True
    assert capabilities["clone_configured"] is False
    assert "yue" in capabilities["languages"]
    available, reason = client.is_available()
    assert available is False
    assert "PROMPT_WAV/TEXT" in reason


def _indextts_snapshot(tmp_path, *, params=None) -> TtsVoiceSnapshot:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    return TtsVoiceSnapshot(
        voice_id="voice-1",
        voice_revision=2,
        backend_family="indextts25_mlx",
        language="zh",
        reference_path=reference,
        reference_sha256="a" * 64,
        default_params=params or {"greedy": True},
    )


@pytest.mark.anyio
async def test_indextts_snapshot_synthesis_caches_speaker_and_emits_wav(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class FakeIndexTTS:
        sample_rate = 22_050

        def build_speaker(self, path: str):
            calls.append({"op": "build_speaker", "path": path})
            return "speaker-context"

        def synthesize(self, text, *, lang, spk, **generation):
            calls.append(
                {
                    "op": "synthesize",
                    "text": text,
                    "lang": lang,
                    "spk": spk,
                    "generation": generation,
                }
            )
            return [-32768, 0, 32767]

    client = IndexTTS25MlxClient(model_instance=FakeIndexTTS())
    snapshot = _indextts_snapshot(tmp_path)
    try:
        first = await client.synthesize_snapshot("你好", snapshot)
        second = await client.synthesize_snapshot("世界", snapshot, duration_factor=1.1)
    finally:
        await client.aclose()

    assert first[:4] == b"RIFF"
    assert second[:4] == b"RIFF"
    with wave.open(io.BytesIO(first), "rb") as handle:
        assert handle.getframerate() == 22_050
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 3
    assert [item["op"] for item in calls] == [
        "build_speaker",
        "synthesize",
        "synthesize",
    ]
    assert calls[1]["generation"] == {"greedy": True}
    assert calls[2]["generation"] == {"greedy": True, "duration_factor": 1.1}


@pytest.mark.anyio
async def test_indextts_snapshot_rejects_wrong_backend_and_unknown_options(tmp_path) -> None:
    client = IndexTTS25MlxClient(model_instance=object())
    wrong = _indextts_snapshot(tmp_path)
    wrong = TtsVoiceSnapshot(
        voice_id=wrong.voice_id,
        voice_revision=wrong.voice_revision,
        backend_family="cosyvoice",
        language=wrong.language,
        reference_path=wrong.reference_path,
        reference_sha256=wrong.reference_sha256,
        default_params=wrong.default_params,
    )
    with pytest.raises(TTSClientError, match="not for IndexTTS"):
        await client.synthesize_snapshot("测试", wrong)
    with pytest.raises(TTSClientError, match="unsupported IndexTTS generation option"):
        await client.synthesize_snapshot("测试", _indextts_snapshot(tmp_path), bad_option=True)
    await client.aclose()


def test_indextts_capabilities_and_lazy_availability(tmp_path) -> None:
    client = IndexTTS25MlxClient(model_dir=str(tmp_path / "missing"))
    assert client.capabilities()["backend"] == "indextts_mlx"
    available, reason = client.is_available()
    assert available is False
    assert "模型目录" in reason
