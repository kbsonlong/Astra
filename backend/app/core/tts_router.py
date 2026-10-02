"""Route realtime TTS requests to a default or snapshot-compatible engine."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .tts_voice_store import TtsVoiceSnapshot
from ..models.tts_client import TTSClientError


_SNAPSHOT_ENGINES = {
    "indextts25_mlx": "indextts_mlx",
    "indextts_mlx": "indextts_mlx",
    "cosyvoice": "cosyvoice",
}


class TtsRouter:
    """A bounded, per-engine router that preserves an explicitly selected voice.

    Engines themselves own model/thread lifecycle. The router only selects an
    engine and applies a small admission limit so an unavailable clone engine
    cannot starve the default voice path.
    """

    def __init__(
        self,
        engines: Mapping[str, object],
        *,
        default_backend: str,
        fallback_backend: str = "",
        max_queue: int = 2,
        user_enabled_backends: tuple[str, ...] | None = None,
        indextts_validated: bool = False,
    ) -> None:
        self.engines = dict(engines)
        if default_backend not in self.engines:
            raise ValueError("default TTS backend is not enabled")
        if fallback_backend and fallback_backend not in self.engines:
            raise ValueError("fallback TTS backend is not enabled")
        if max_queue < 1:
            raise ValueError("TTS queue size must be positive")
        self.user_enabled_backends = set(
            tuple(self.engines) if user_enabled_backends is None else user_enabled_backends
        )
        self.indextts_validated = indextts_validated
        if not self.user_enabled_backends.issubset(self.engines):
            raise ValueError("user-enabled TTS backend is not enabled")
        if not self._is_user_enabled(default_backend):
            raise ValueError("default TTS backend is not approved for user access")
        if fallback_backend and not self._is_user_enabled(fallback_backend):
            raise ValueError("fallback TTS backend is not approved for user access")
        self.default_backend = default_backend
        self.fallback_backend = fallback_backend
        self._limits = {key: asyncio.Semaphore(1) for key in self.engines}
        self._pending = {key: 0 for key in self.engines}
        self._pending_lock = asyncio.Lock()
        self._max_queue = max_queue
        self._closed = False

    async def synthesize(self, text: str) -> bytes:
        try:
            return await self._run(self.default_backend, "synthesize", text)
        except TTSClientError:
            # Default voices may use an explicitly configured fallback before
            # anything has been emitted. Snapshot voices never take this path.
            if not self.fallback_backend or self.fallback_backend == self.default_backend:
                raise
            return await self._run(self.fallback_backend, "synthesize", text)

    async def synthesize_snapshot(self, text: str, snapshot: TtsVoiceSnapshot) -> bytes:
        backend = _SNAPSHOT_ENGINES.get(snapshot.backend_family)
        if backend is None:
            raise TTSClientError("TTS voice backend is unsupported")
        if backend not in self.engines:
            raise TTSClientError("TTS voice backend is not enabled")
        if not self._is_user_enabled(backend):
            raise TTSClientError(f"TTS backend {backend} is not approved for user access")
        return await self._run(backend, "synthesize_snapshot", text, snapshot)

    def _is_user_enabled(self, backend: str) -> bool:
        return backend in self.user_enabled_backends and (
            backend != "indextts_mlx" or self.indextts_validated
        )

    def _user_reason(self, backend: str) -> str:
        if backend not in self.user_enabled_backends:
            return "user access disabled"
        if backend == "indextts_mlx" and not self.indextts_validated:
            return "device_validation_required"
        return "ready"

    async def _run(self, backend: str, method: str, *args: object) -> bytes:
        if self._closed:
            raise TTSClientError("TTS router is closed")
        client = self.engines[backend]
        available = getattr(client, "is_available", None)
        if callable(available):
            ok, reason = available()
            if not ok:
                raise TTSClientError(f"TTS backend {backend} is unavailable: {reason}")
        async with self._pending_lock:
            if self._pending[backend] >= self._max_queue:
                raise TTSClientError(f"TTS backend {backend} queue is full")
            self._pending[backend] += 1
        try:
            async with self._limits[backend]:
                target = getattr(client, method, None)
                if not callable(target):
                    raise TTSClientError(f"TTS backend {backend} does not support voice snapshots")
                result = await target(*args)
                if not isinstance(result, bytes) or not result:
                    raise TTSClientError(f"TTS backend {backend} returned empty audio")
                return result
        finally:
            async with self._pending_lock:
                self._pending[backend] -= 1

    def is_ready(self) -> bool:
        default = self.engines[self.default_backend]
        ready = getattr(default, "is_ready", None)
        return bool(ready and ready())

    def is_available(self) -> tuple[bool, str]:
        client = self.engines[self.default_backend]
        probe = getattr(client, "is_available", None)
        if not callable(probe):
            return True, "ready"
        return probe()

    def capabilities(self) -> dict[str, Any]:
        engines: dict[str, object] = {}
        for key, client in self.engines.items():
            probe = getattr(client, "is_available", None)
            available, reason = probe() if callable(probe) else (True, "ready")
            caps_fn = getattr(client, "capabilities", None)
            caps = dict(caps_fn()) if callable(caps_fn) else {}
            engines[key] = {
                "available": bool(available),
                "reason": str(reason),
                "default": key == self.default_backend,
                "user_enabled": self._is_user_enabled(key),
                "user_reason": self._user_reason(key),
                "validated": key != "indextts_mlx" or self.indextts_validated,
                "pending": self._pending[key],
                "capabilities": caps,
            }
        return {
            "engine": "router",
            "backend": "router",
            "default_backend": self.default_backend,
            "fallback_backend": self.fallback_backend or None,
            "engines": engines,
        }

    async def aclose(self) -> None:
        self._closed = True
        for client in self.engines.values():
            close = getattr(client, "aclose", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result
