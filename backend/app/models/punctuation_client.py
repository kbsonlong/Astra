"""标点恢复阶段的模型适配器。"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..core.workflow import PassthroughPunctuation, clean_repeated_punctuation


class PunctuationClientError(RuntimeError):
    pass


class FunASRPunctuationClient:
    """FunASR ct-punc-c 的懒加载适配器。

    FunASR 是可选依赖，避免导入阶段阻塞 Astra 启动；模型只在首次文本
    到达时加载。`model_factory` 仅用于测试或接入兼容实现。
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "mps",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model
        self.device = device
        self._model_factory = model_factory
        self._model: Any = None

    async def restore(self, text: str, *, language: str = "zh") -> str:
        if not text.strip():
            return text
        return await asyncio.to_thread(self._restore_blocking, text, language)

    def _restore_blocking(self, text: str, language: str) -> str:
        model = self._get_model()
        try:
            result = model.generate(
                input=text,
                batch_size_s=300,
                disable_pbar=True,
            )
        except TypeError:
            result = model.generate(input=text)
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            result = result.get("text", "")
        result_text = getattr(result, "text", result)
        if not isinstance(result_text, str):
            raise PunctuationClientError("FunASR punctuation response does not contain text")
        return clean_repeated_punctuation(result_text.strip())

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._model_factory or self._default_model_factory
        try:
            self._model = factory(
                model=self.model_name,
                device=self.device,
                disable_update=True,
            )
        except Exception as exc:
            raise PunctuationClientError(
                f"failed to load punctuation model {self.model_name}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        return self._model

    @staticmethod
    def _default_model_factory(**kwargs: Any) -> Any:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise PunctuationClientError("funasr is not installed") from exc
        return AutoModel(**kwargs)

    def is_ready(self) -> bool:
        if self._model is not None or self._model_factory is not None:
            return bool(self.model_name)
        try:
            import funasr  # noqa: F401
        except ImportError:
            return False
        return bool(self.model_name)


def build_punctuation_client(
    *,
    enabled: bool,
    engine: str,
    model: str,
    device: str,
) -> Any:
    if not enabled or engine.lower() in {"", "none", "passthrough", "noop"}:
        return PassthroughPunctuation()
    if engine.lower() in {"funasr", "ct-punc-c", "ct_punc_c"}:
        return FunASRPunctuationClient(model, device=device)
    raise ValueError(f"unsupported punctuation engine: {engine}")
