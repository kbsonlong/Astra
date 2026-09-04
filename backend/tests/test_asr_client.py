import pytest

from app.models.asr_client import ASRClientError, MlxAudioAsrClient


@pytest.mark.anyio
async def test_transcribe_uses_mlx_audio_sdk_with_temp_wav() -> None:
    observed: dict[str, object] = {}

    class Result:
        text = "  hello world  "

    def fake_load_model(model: str) -> object:
        observed["model_id"] = model
        return object()

    def fake_load_audio(path: str) -> list[float]:
        return [0.0]

    def fake_generate_transcription(**kwargs: object) -> Result:
        with open(kwargs["audio"], "rb") as audio:  # type: ignore[arg-type]
            observed["audio_bytes"] = audio.read()
        observed.update(kwargs)
        return Result()

    client = MlxAudioAsrClient(
        "test-model",
        hotwords=("host", "host 网络模式"),
        system_prompt="只输出实际说出的内容。",
        load_audio=fake_load_audio,
        load_model=fake_load_model,
        generate_transcription=fake_generate_transcription,
    )

    assert await client.transcribe(b"wav", filename="test.wav") == "hello world"
    assert observed["audio_bytes"] == b"wav"
    assert observed["model_id"] == "test-model"
    assert str(observed["audio"]).endswith("/test.wav")
    assert observed["language"] == "Chinese"
    assert observed["max_tokens"] == 512
    assert observed["temperature"] == 0.0
    assert observed["repetition_penalty"] == 1.08
    assert observed["repetition_context_size"] == 100
    assert observed["chunk_duration"] == 30.0
    assert observed["hotwords"] == ["host", "host 网络模式"]
    assert observed["system_prompt"] == "只输出实际说出的内容。"
    assert observed["format"] == "txt"


@pytest.mark.anyio
async def test_transcribe_includes_sdk_error_detail() -> None:
    def fake_load_model(model: str) -> object:
        raise FileNotFoundError(model)

    client = MlxAudioAsrClient(
        "missing-model",
        load_audio=lambda path: [0.0],
        load_model=fake_load_model,
        generate_transcription=lambda **kwargs: None,
    )

    with pytest.raises(ASRClientError, match="FileNotFoundError: missing-model"):
        await client.transcribe(b"wav")


@pytest.mark.anyio
async def test_long_audio_is_transcribed_in_independent_chunks() -> None:
    observed: list[object] = []

    class Result:
        def __init__(self, text: str) -> None:
            self.text = text

    def fake_generate_transcription(**kwargs: object) -> Result:
        observed.append(kwargs["audio"])
        return Result(f"chunk-{len(observed)}")

    client = MlxAudioAsrClient(
        "test-model",
        chunk_duration=1.0,
        long_audio_threshold=1.0,
        load_audio=lambda path: [0.0] * 32000,
        load_model=lambda model: object(),
        generate_transcription=fake_generate_transcription,
    )

    assert await client.transcribe(b"audio", filename="meeting.m4a") == "chunk-1 chunk-2"
    assert len(observed) == 2


def test_strip_prompt_leak_removes_injected_hotwords_and_system_prompt() -> None:
    client = MlxAudioAsrClient(
        "test-model",
        hotwords=("host", "host 网络模式", "网络模式", "大佬", "CI/CD"),
        system_prompt="你是一个专业的中文语音转写器。只输出音频中实际说出的内容，不要补充、解释或改写。",
    )

    dirty = (
        "就是大家还是对这一块要有一个警惕性，就是机密的东西一定不要往上面去扔。"
        "你是一个专业的中文语音转写器。只输出音频中实际说出的内容，不要补充、解释或改写。"
        "host, host 网络模式, 网络模式, 大佬, CI/CD"
    )
    assert "host 网络模式" not in client._strip_prompt_leak(dirty)
    assert "中文语音转写器" not in client._strip_prompt_leak(dirty)
    # 切除后保留正文主体
    assert "警惕性" in client._strip_prompt_leak(dirty)

    # 正文里真实出现的短词不误伤
    normal = "我们那个环境用的是 host 模式部署，感觉还行。"
    assert client._strip_prompt_leak(normal) == normal

    # 无泄漏场景原样返回
    plain = client._strip_prompt_leak("今天讨论了资源优化的事情。")
    assert plain == "今天讨论了资源优化的事情。"


@pytest.mark.anyio
async def test_sdk_model_loaded_once_across_transcribes(monkeypatch) -> None:
    """生产 SDK 路径: 同线程多次 transcribe 只加载一次模型。

    会议管线每 VAD 段一次 transcribe, 若不缓存则 131 段 x 0.6-2s 纯重载开销。
    """
    loads: list[str] = []

    class Result:
        text = "ok"

    class FakeModel:
        def generate(self, audio: object, **kwargs: object) -> Result:
            return Result()

    def fake_load_model(model: str) -> object:
        loads.append(model)
        return FakeModel()

    monkeypatch.setattr(
        MlxAudioAsrClient, "_load_model_from_sdk", staticmethod(fake_load_model)
    )
    monkeypatch.setattr(
        MlxAudioAsrClient,
        "_load_audio_from_sdk",
        staticmethod(lambda path: [0.0] * 16000),
    )
    client = MlxAudioAsrClient("cached-model")
    try:
        assert await client.transcribe(b"wav1", filename="a.wav") == "ok"
        assert await client.transcribe(b"wav2", filename="b.wav") == "ok"
    finally:
        MlxAudioAsrClient.clear_model_cache()
    assert loads == ["cached-model"]  # 第二次命中缓存, 不再加载


@pytest.mark.anyio
async def test_injected_loaders_are_not_cached() -> None:
    """注入路径(测试): 每次 transcribe 都走注入的 loader, 不经过模块缓存。"""

    loads: list[str] = []

    class Result:
        text = "hi"

    def fake_load_model(model: str) -> object:
        loads.append(model)
        return object()

    client = MlxAudioAsrClient(
        "uncached-model",
        load_model=fake_load_model,
        load_audio=lambda path: [0.0],
        generate_transcription=lambda **kwargs: Result(),
    )
    assert await client.transcribe(b"wav") == "hi"
    assert await client.transcribe(b"wav") == "hi"
    assert loads == ["uncached-model", "uncached-model"]
