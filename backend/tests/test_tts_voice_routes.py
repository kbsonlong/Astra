import io
import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas.tts_voice import CONSENT_CONFIRMATION


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
