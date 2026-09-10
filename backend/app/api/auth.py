"""同源管理员令牌认证与 HttpOnly 会话 Cookie。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import Settings

SESSION_COOKIE_NAME = "astra_admin_session"
_SESSION_VERSION = "v1"
router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginPayload(BaseModel):
    token: str = Field(min_length=1, max_length=4096)


def auth_required(settings: Settings) -> bool:
    """An empty ADMIN_TOKEN preserves legacy LAN deployments during migration."""
    return bool(settings.admin_token)


def _session_key(settings: Settings) -> bytes:
    # ADMIN_SESSION_SECRET lets operators rotate sessions separately from the login
    # credential. Falling back keeps a single-token deployment operable.
    return (settings.admin_session_secret or settings.admin_token).encode("utf-8")


def _signature(settings: Settings, payload: str) -> str:
    return hmac.new(_session_key(settings), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_session_cookie(settings: Settings, *, now: int | None = None) -> str:
    issued_at = int(time.time() if now is None else now)
    expires_at = issued_at + settings.admin_session_ttl_seconds
    nonce = base64.urlsafe_b64encode(secrets.token_bytes(18)).decode("ascii").rstrip("=")
    payload = f"{_SESSION_VERSION}.{expires_at}.{nonce}"
    return f"{payload}.{_signature(settings, payload)}"


def verify_session_cookie(
    settings: Settings, cookie: str | None, *, now: int | None = None
) -> bool:
    if not auth_required(settings):
        return True
    if not cookie:
        return False
    try:
        version, expiry, nonce, signature = cookie.split(".", 3)
        expires_at = int(expiry)
    except (TypeError, ValueError):
        return False
    if version != _SESSION_VERSION or not nonce or expires_at <= int(time.time() if now is None else now):
        return False
    payload = f"{version}.{expiry}.{nonce}"
    return hmac.compare_digest(signature, _signature(settings, payload))


def is_authorized_request(request: Request) -> bool:
    return verify_session_cookie(
        request.app.state.settings, request.cookies.get(SESSION_COOKIE_NAME)
    )


async def authorize_websocket(websocket: WebSocket) -> bool:
    if verify_session_cookie(
        websocket.app.state.settings, websocket.cookies.get(SESSION_COOKIE_NAME)
    ):
        return True
    await websocket.accept()
    await websocket.send_json({"type": "error", "code": "authentication_required"})
    await websocket.close(code=1008)
    return False


def _session_response(settings: Settings, *, authenticated: bool) -> dict[str, object]:
    return {
        "auth_required": auth_required(settings),
        "authenticated": authenticated,
        "session_ttl_seconds": settings.admin_session_ttl_seconds if auth_required(settings) else 0,
    }


@router.get("/status")
async def status(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    return _session_response(settings, authenticated=is_authorized_request(request))


@router.post("/login")
async def login(payload: LoginPayload, request: Request) -> JSONResponse:
    settings = request.app.state.settings
    if not auth_required(settings):
        return JSONResponse(_session_response(settings, authenticated=True))
    if not hmac.compare_digest(payload.token, settings.admin_token):
        raise HTTPException(status_code=401, detail="invalid administrator token")
    response = JSONResponse(_session_response(settings, authenticated=True))
    response.set_cookie(
        SESSION_COOKIE_NAME,
        create_session_cookie(settings),
        max_age=settings.admin_session_ttl_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="strict",
        path="/",
    )
    return response


@router.post("/logout")
async def logout(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    response = JSONResponse(_session_response(settings, authenticated=False))
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        secure=settings.auth_cookie_secure,
        samesite="strict",
        path="/",
    )
    return response
