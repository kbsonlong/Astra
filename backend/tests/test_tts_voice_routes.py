import io
import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas.tts_voice import CONSENT_CONFIRMATION
from app.models.tts_client import TTSClientError


def _wav_bytes(duration_s: float = 6.0) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x01\x00" * int(16_000 * duration_s))
    return output.getvalue()


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(
        tts_voice_store_path=str(tmp_path / "tts-voices.sqlite3"),
        tts_voice_audio_dir=str(tmp_path / "tts-voices"),
        tts_voice_min_reference_seconds=5,
        tts_voice_max_reference_seconds=15,
        tts_voice_max_upload_bytes=2 * 1024 * 1024,
        task_store_path=str(tmp_path / "tasks.sqlite3"),
        speaker_store_path=str(tmp_path / "speakers.sqlite3"),
        speaker_sample_dir=str(tmp_path / "speaker-samples"),
    )
    return TestClient(create_app(settings, enable_pipeline=False, enable_meeting=False))


def _create(client: TestClient, filename: str = "reference.wav") -> dict:
    response = client.post(
        "/api/tts/voices",
        data={
            "name": "小雅",
            "language": "zh",
            "backend_family": "indextts25_mlx",
            "default_params": json.dumps({"greedy": True}),
        },
        files={"file": (filename, _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_voice_crud_and_audio_download(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = _create(client)
        voice_id = created["voice_id"]
        assert created["status"] == "draft"
        assert created["current_revision"] is None
        assert created["revision"]["consent_status"] == "unknown"

        listed = client.get("/api/tts/voices?status=draft&language=zh")
        assert listed.status_code == 200
        assert listed.json()["pagination"]["total"] == 1

        renamed = client.patch(
            f"/api/tts/voices/{voice_id}", json={"display_name": "小雅-会议"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["display_name"] == "小雅-会议"

        consent = client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/consent",
            json={"confirmation": CONSENT_CONFIRMATION, "note": "本人授权"},
        )
        assert consent.status_code == 200

        activated = client.post(f"/api/tts/voices/{voice_id}/activate?revision=1")
        assert activated.status_code == 200
        assert activated.json()["status"] == "active"

        audio = client.get(f"/api/tts/voices/{voice_id}/revisions/1/audio")
        assert audio.status_code == 200
        assert audio.headers["content-type"].startswith("audio/wav")
        assert audio.content[:4] == b"RIFF"
        current_audio = client.get(f"/api/tts/voices/{voice_id}/audio")
        assert current_audio.status_code == 200

        archived = client.post(f"/api/tts/voices/{voice_id}/archive")
        assert archived.status_code == 200
        assert archived.json()["status"] == "archived"
        assert client.post(f"/api/tts/voices/{voice_id}/restore").json()["status"] == "active"


def test_revision_upload_and_consent_validation(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = _create(client)
        voice_id = created["voice_id"]

        revision = client.post(
            f"/api/tts/voices/{voice_id}/revisions",
            data={"language": "zh", "default_params": "{}"},
            files={"file": ("replacement.wav", _wav_bytes(7), "audio/wav")},
        )
        assert revision.status_code == 201
        assert revision.json()["revision"] == 2

        bad_consent = client.post(
            f"/api/tts/voices/{voice_id}/revisions/2/consent",
            json={"confirmation": "yes"},
        )
        assert bad_consent.status_code == 422

        revisions = client.get(f"/api/tts/voices/{voice_id}/revisions")
        assert revisions.status_code == 200
        assert [item["revision"] for item in revisions.json()["items"]] == [2, 1]
        detail = client.get(f"/api/tts/voices/{voice_id}/revisions/2")
        assert detail.status_code == 200
        assert detail.json()["revision"] == 2


def test_invalid_uploads_and_revoked_audio(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        too_short = client.post(
            "/api/tts/voices",
            data={"name": "短音频"},
            files={"file": ("short.wav", _wav_bytes(1), "audio/wav")},
        )
        assert too_short.status_code == 422

        created = _create(client)
        voice_id = created["voice_id"]
        assert client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/consent",
            json={"confirmation": CONSENT_CONFIRMATION},
        ).status_code == 200
        assert client.post(f"/api/tts/voices/{voice_id}/activate").status_code == 200
        revoked = client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/revoke-consent",
            json={"note": "撤销"},
        )
        assert revoked.status_code == 200
        assert client.get(f"/api/tts/voices/{voice_id}/revisions/1/audio").status_code == 409


def test_revoked_noncurrent_revision_audio_is_not_readable(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = _create(client)
        voice_id = created["voice_id"]
        assert client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/consent",
            json={"confirmation": CONSENT_CONFIRMATION},
        ).status_code == 200
        assert client.post(f"/api/tts/voices/{voice_id}/activate?revision=1").status_code == 200
        assert client.post(
            f"/api/tts/voices/{voice_id}/revisions",
            data={"language": "zh", "default_params": "{}"},
            files={"file": ("r2.wav", _wav_bytes(7), "audio/wav")},
        ).status_code == 201
        assert client.post(
            f"/api/tts/voices/{voice_id}/revisions/2/consent",
            json={"confirmation": CONSENT_CONFIRMATION},
        ).status_code == 200
        assert client.post(f"/api/tts/voices/{voice_id}/activate?revision=2").status_code == 200

        revoked = client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/revoke-consent",
            json={"note": "撤销旧版本"},
        )
        assert revoked.status_code == 200
        assert revoked.json()["actions"] == {
            "can_play_reference": False,
            "can_preview": False,
            "can_use_in_session": False,
            "reason": "revoked",
        }
        assert client.get(f"/api/tts/voices/{voice_id}/revisions/1/audio").status_code == 409
        assert client.get(f"/api/tts/voices/{voice_id}/revisions/2/audio").status_code == 200


def test_voice_actions_are_safe_lifecycle_contract(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = _create(client)
        voice_id = created["voice_id"]
        assert created["revision"]["actions"]["reason"] == "consent_required"
        assert created["revision"]["actions"]["can_play_reference"] is True
        assert created["revision"]["actions"]["can_use_in_session"] is False

        assert client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/consent",
            json={"confirmation": CONSENT_CONFIRMATION},
        ).status_code == 200
        active = client.post(f"/api/tts/voices/{voice_id}/activate?revision=1")
        assert active.status_code == 200
        assert active.json()["revision"]["actions"] == {
            "can_play_reference": True,
            "can_preview": True,
            "can_use_in_session": True,
            "reason": "ready",
        }


def test_voice_preview_uses_authorized_pinned_revision(tmp_path: Path) -> None:
    class PreviewTts:
        def __init__(self) -> None:
            self.calls = []

        async def synthesize_snapshot(self, text, snapshot) -> bytes:
            self.calls.append((text, snapshot))
            return b"RIFFpreview"

        def capabilities(self) -> dict:
            return {"backend": "indextts_mlx", "model_revision": "model-r1"}

    with _client(tmp_path) as client:
        created = _create(client)
        voice_id = created["voice_id"]
        assert client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/consent",
            json={"confirmation": CONSENT_CONFIRMATION},
        ).status_code == 200
        assert client.post(f"/api/tts/voices/{voice_id}/activate?revision=1").status_code == 200
        tts = PreviewTts()
        client.app.state.pipeline = type("Pipeline", (), {"tts": tts})()

        response = client.post(
            f"/api/tts/voices/{voice_id}/test",
            json={"revision": 1, "text": "你好，Astra。"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.headers["x-astra-voice-id"] == voice_id
        assert response.headers["x-astra-voice-revision"] == "1"
        assert response.headers["x-astra-backend"] == "indextts25_mlx"
        assert response.headers["x-astra-model-revision"] == "model-r1"
        assert tts.calls[0][0] == "你好，Astra。"
        assert tts.calls[0][1].voice_revision == 1


def test_voice_preview_rejects_unusable_revision_and_queue_overload(tmp_path: Path) -> None:
    class QueueFullTts:
        async def synthesize_snapshot(self, text, snapshot) -> bytes:
            raise TTSClientError("TTS backend indextts_mlx queue is full")

    with _client(tmp_path) as client:
        created = _create(client)
        voice_id = created["voice_id"]
        assert client.post(
            f"/api/tts/voices/{voice_id}/test",
            json={"revision": 1, "text": "未授权"},
        ).status_code == 409
        assert client.post(
            f"/api/tts/voices/{voice_id}/revisions/1/consent",
            json={"confirmation": CONSENT_CONFIRMATION},
        ).status_code == 200
        assert client.post(f"/api/tts/voices/{voice_id}/activate?revision=1").status_code == 200
        client.app.state.pipeline = type("Pipeline", (), {"tts": QueueFullTts()})()
        assert client.post(
            f"/api/tts/voices/{voice_id}/test",
            json={"revision": 1, "text": "排队"},
        ).status_code == 429
