import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.auth import (
    SESSION_COOKIE_NAME,
    create_session_cookie,
    verify_session_cookie,
)
from app.config import Settings
from app.main import create_app


def _protected_client() -> TestClient:
    return TestClient(
        create_app(
            settings=Settings(
                admin_token="correct-admin-token",
                admin_session_secret="session-signing-secret",
                admin_session_ttl_seconds=60,
            ),
            enable_pipeline=False,
            enable_meeting=False,
        )
    )


def test_unconfigured_token_keeps_compatible_open_mode() -> None:
    client = TestClient(create_app(enable_pipeline=False, enable_meeting=False))

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_required"] is False
    assert status.json()["authenticated"] is True
    assert client.get("/api/config").status_code == 200


def test_http_authentication_login_logout_and_protected_health() -> None:
    client = _protected_client()

    assert client.get("/api/config").status_code == 401
    assert client.get("/api/health").status_code == 401

    invalid = client.post("/api/auth/login", json={"token": "wrong-token"})
    assert invalid.status_code == 401
    assert "set-cookie" not in invalid.headers

    logged_in = client.post(
        "/api/auth/login", json={"token": "correct-admin-token"}
    )
    assert logged_in.status_code == 200
    assert logged_in.json() == {
        "auth_required": True,
        "authenticated": True,
        "session_ttl_seconds": 60,
    }
    cookie = logged_in.headers["set-cookie"]
    assert f"{SESSION_COOKIE_NAME}=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert client.get("/api/config").status_code == 200

    logged_out = client.post("/api/auth/logout")
    assert logged_out.status_code == 200
    assert client.get("/api/config").status_code == 401


def test_signed_cookie_rejects_tampering_and_expiration() -> None:
    settings = Settings(
        admin_token="correct-admin-token",
        admin_session_secret="session-signing-secret",
        admin_session_ttl_seconds=60,
    )
    cookie = create_session_cookie(settings, now=100)
    assert verify_session_cookie(settings, cookie, now=159) is True
    assert verify_session_cookie(settings, cookie, now=160) is False
    assert verify_session_cookie(settings, f"{cookie}x", now=101) is False

    client = _protected_client()
    client.cookies.set(SESSION_COOKIE_NAME, f"{cookie}x")
    assert client.get("/api/config").status_code == 401


def test_websocket_rejects_unauthenticated_and_accepts_authenticated_client() -> None:
    client = _protected_client()

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {
            "type": "error",
            "code": "authentication_required",
        }
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert client.post(
        "/api/auth/login", json={"token": "correct-admin-token"}
    ).status_code == 200
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        assert websocket.receive_json() == {"type": "state_change", "state": "LISTENING"}
