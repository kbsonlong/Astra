import asyncio
import io
import threading
import time
import wave

import pytest

from app.models.tts_client import (
    MlxAudioTtsClient,
    PiperSdkTtsClient,
    TTSClientError,
)


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
