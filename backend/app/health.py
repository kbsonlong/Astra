from dataclasses import dataclass

import httpx

from .config import Settings


@dataclass(frozen=True)
class DependencyStatus:
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"ok": self.ok}
        if self.detail:
            result["detail"] = self.detail
        return result


async def probe_url(client: httpx.AsyncClient, url: str) -> DependencyStatus:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return DependencyStatus(ok=True)
    except httpx.HTTPStatusError as exc:
        return DependencyStatus(ok=False, detail=f"HTTP {exc.response.status_code}")
    except httpx.HTTPError as exc:
        return DependencyStatus(ok=False, detail=exc.__class__.__name__)


async def collect_health(
    settings: Settings,
    pipeline: object | None = None,
    meeting_pipeline: object | None = None,
) -> dict[str, object]:
    timeout = httpx.Timeout(
        settings.llm_request_timeout_seconds,
        connect=settings.llm_connect_timeout_seconds,
    )
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        llm = await probe_url(client, f"{settings.llm_base_url}{settings.llm_models_path}")
    asr_client = getattr(pipeline, "asr", None)
    tts_client = getattr(pipeline, "tts", None)
    asr_ok = bool(asr_client and asr_client.is_ready())
    tts_ok = bool(tts_client and tts_client.is_ready())
    asr_engine = _engine_status(asr_client, default_mode="mlx-sdk")
    tts_engine = _engine_status(tts_client, default_mode="piper-sdk")
    meeting_ok = bool(
        meeting_pipeline
        and meeting_pipeline.llm is not None
        and getattr(meeting_pipeline.llm, "model", "")
    )
    workflow = getattr(meeting_pipeline, "workflow", None)
    workflow_status = (
        workflow.stage_status() if workflow is not None else {"enabled": False}
    )
    return {
        "ok": llm.ok and asr_ok and tts_ok,
        "llm": {
            **llm.as_dict(),
            "base_url": settings.llm_base_url,
            "runtime": "unknown",
            "models_ok": llm.ok,
            "stream_ok": False,
        },
        "asr": {"ok": asr_ok, "mode": asr_engine["mode"], **asr_engine["extra"]},
        "tts": {"ok": tts_ok, "mode": tts_engine["mode"], **tts_engine["extra"]},
        "meeting": {
            "ok": meeting_ok,
            "workflow": workflow_status,
            "output_dir": settings.meeting_output_dir,
        },
        "version": settings.version,
    }


def _engine_status(client: object | None, *, default_mode: str) -> dict[str, object]:
    """从引擎 client 提取 capabilities/is_available (方向二), 对旧 client 优雅降级。

    返回 {"mode": str, "extra": {...}}, extra 含 available/reason/capabilities。
    """
    extra: dict[str, object] = {}
    mode = default_mode
    if client is None:
        return {"mode": mode, "extra": {"available": False, "reason": "not configured"}}

    caps_fn = getattr(client, "capabilities", None)
    if callable(caps_fn):
        try:
            caps = dict(caps_fn())
        except Exception:  # pragma: no cover - defensive
            caps = {}
        if caps:
            extra["capabilities"] = caps
            engine = caps.get("engine") or caps.get("backend")
            if isinstance(engine, str) and engine:
                mode = engine

    avail_fn = getattr(client, "is_available", None)
    if callable(avail_fn):
        try:
            available, reason = avail_fn()
        except Exception:  # pragma: no cover - defensive
            available, reason = False, "availability probe failed"
        extra["available"] = bool(available)
        extra["reason"] = str(reason)

    return {"mode": mode, "extra": extra}
