import pytest

from app.core.tts_router import TtsRouter
from app.core.tts_voice_store import TtsVoiceSnapshot
from app.models.tts_client import TTSClientError


class FakeEngine:
    def __init__(self, name: str, *, available: bool = True) -> None:
        self.name = name
        self.available = available
        self.default_calls: list[str] = []
        self.snapshot_calls: list[tuple[str, TtsVoiceSnapshot]] = []
        self.closed = False

    def is_ready(self) -> bool:
        return self.available

    def is_available(self) -> tuple[bool, str]:
        return self.available, "ready" if self.available else "not ready"

    def capabilities(self) -> dict[str, object]:
        return {"engine": self.name, "clone": self.name == "indextts_mlx"}

    async def synthesize(self, text: str) -> bytes:
        self.default_calls.append(text)
        return f"{self.name}:{text}".encode()

    async def synthesize_snapshot(self, text: str, snapshot: TtsVoiceSnapshot) -> bytes:
        self.snapshot_calls.append((text, snapshot))
        return f"{self.name}:{text}".encode()

    async def aclose(self) -> None:
        self.closed = True


def snapshot() -> TtsVoiceSnapshot:
    return TtsVoiceSnapshot(
        voice_id="voice-1",
        voice_revision=1,
        backend_family="indextts25_mlx",
        language="zh",
        reference_path=__file__,
        reference_sha256="a" * 64,
        default_params={},
    )


@pytest.mark.anyio
async def test_router_uses_default_and_snapshot_engine() -> None:
    piper = FakeEngine("piper")
    index = FakeEngine("indextts_mlx")
    router = TtsRouter(
        {"piper": piper, "indextts_mlx": index},
        default_backend="piper",
        indextts_validated=True,
    )

    assert await router.synthesize("默认") == b"piper:\xe9\xbb\x98\xe8\xae\xa4"
    assert await router.synthesize_snapshot("音色", snapshot()) == b"indextts_mlx:\xe9\x9f\xb3\xe8\x89\xb2"
    assert piper.default_calls == ["默认"]
    assert len(index.snapshot_calls) == 1


@pytest.mark.anyio
async def test_snapshot_engine_never_falls_back_to_default() -> None:
    piper = FakeEngine("piper")
    router = TtsRouter({"piper": piper}, default_backend="piper")

    with pytest.raises(TTSClientError, match="not enabled"):
        await router.synthesize_snapshot("音色", snapshot())
    assert piper.default_calls == []


@pytest.mark.anyio
async def test_default_can_fall_back_but_snapshot_cannot() -> None:
    unavailable = FakeEngine("piper", available=False)
    fallback = FakeEngine("mlx_audio")
    router = TtsRouter(
        {"piper": unavailable, "mlx_audio": fallback},
        default_backend="piper",
        fallback_backend="mlx_audio",
    )

    assert await router.synthesize("默认") == b"mlx_audio:\xe9\xbb\x98\xe8\xae\xa4"
    with pytest.raises(TTSClientError, match="not enabled"):
        await router.synthesize_snapshot("音色", snapshot())


@pytest.mark.anyio
async def test_router_reports_engines_and_closes_them() -> None:
    piper = FakeEngine("piper")
    index = FakeEngine("indextts_mlx")
    router = TtsRouter({"piper": piper, "indextts_mlx": index}, default_backend="piper")

    caps = router.capabilities()
    assert caps["default_backend"] == "piper"
    assert caps["engines"]["indextts_mlx"]["capabilities"]["clone"] is True
    await router.aclose()
    assert piper.closed and index.closed
    with pytest.raises(TTSClientError, match="closed"):
        await router.synthesize("again")


def test_router_keeps_user_enablement_separate_from_technical_enablement() -> None:
    router = TtsRouter(
        {"piper": FakeEngine("piper"), "indextts_mlx": FakeEngine("indextts_mlx")},
        default_backend="piper",
        user_enabled_backends=("piper",),
    )

    engines = router.capabilities()["engines"]
    assert engines["piper"]["user_enabled"] is True
    assert engines["indextts_mlx"]["available"] is True
    assert engines["indextts_mlx"]["user_enabled"] is False


@pytest.mark.anyio
async def test_router_requires_explicit_indextts_device_validation() -> None:
    router = TtsRouter(
        {"piper": FakeEngine("piper"), "indextts_mlx": FakeEngine("indextts_mlx")},
        default_backend="piper",
        user_enabled_backends=("piper", "indextts_mlx"),
    )

    caps = router.capabilities()["engines"]["indextts_mlx"]
    assert caps["available"] is True
    assert caps["validated"] is False
    assert caps["user_enabled"] is False
    assert caps["user_reason"] == "device_validation_required"
    with pytest.raises(TTSClientError, match="not approved"):
        await router.synthesize_snapshot("音色", snapshot())


def test_router_rejects_unvalidated_indextts_as_default_or_fallback() -> None:
    engines = {"piper": FakeEngine("piper"), "indextts_mlx": FakeEngine("indextts_mlx")}
    with pytest.raises(ValueError, match="default TTS backend is not approved"):
        TtsRouter(
            engines,
            default_backend="indextts_mlx",
            user_enabled_backends=("piper", "indextts_mlx"),
        )
    with pytest.raises(ValueError, match="fallback TTS backend is not approved"):
        TtsRouter(
            engines,
            default_backend="piper",
            fallback_backend="indextts_mlx",
            user_enabled_backends=("piper", "indextts_mlx"),
        )
