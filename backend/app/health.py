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


async def collect_health(settings: Settings) -> dict[str, object]:
    timeout = httpx.Timeout(
        settings.llm_request_timeout_seconds,
        connect=settings.llm_connect_timeout_seconds,
    )
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        llm = await probe_url(client, f"{settings.llm_base_url}{settings.llm_models_path}")
        asr = await probe_url(client, f"{settings.asr_endpoint}/health")
        tts = await probe_url(client, f"{settings.tts_endpoint}/health")

    return {
        "ok": llm.ok and asr.ok and tts.ok,
        "llm": {
            **llm.as_dict(),
            "base_url": settings.llm_base_url,
            "runtime": "unknown",
            "models_ok": llm.ok,
            "stream_ok": False,
        },
        "asr": {**asr.as_dict(), "mode": "docker-cpu"},
        "tts": tts.as_dict(),
        "version": settings.version,
    }
