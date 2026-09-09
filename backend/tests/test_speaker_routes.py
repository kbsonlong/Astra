import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.core.speaker_registry import EnrollmentResult, ResemblyzerEnrollmentService
from app.main import create_app


def _client(tmp_path):
    settings = Settings(speaker_store_path=str(tmp_path / "speakers.sqlite3"))
    return TestClient(
        create_app(settings=settings, enable_pipeline=False, enable_meeting=False)
    )


def test_speaker_profile_lifecycle_and_rename(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.post("/api/speakers", json={"display_name": "忠思"})
    assert response.status_code == 200
    profile = response.json()
    speaker_id = profile["speaker_id"]
    uuid.UUID(speaker_id)
    assert profile["display_name"] == "忠思"
    assert profile["sample_count"] == 0

    response = client.patch(
        f"/api/speakers/{speaker_id}", json={"display_name": "主持人"}
    )
    assert response.status_code == 200
    assert response.json()["speaker_id"] == speaker_id
    assert response.json()["display_name"] == "主持人"

    response = client.get(f"/api/speakers/{speaker_id}")
    assert response.status_code == 200
    assert response.json()["display_name"] == "主持人"

    response = client.delete(f"/api/speakers/{speaker_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "disabled"
    assert client.get(f"/api/speakers/{speaker_id}").status_code == 200
    assert client.get("/api/speakers").json()["items"] == []


def test_enroll_sample_api_uses_uuid_and_returns_metadata(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    speaker_id = client.post(
        "/api/speakers", json={"display_name": "参会人"}
    ).json()["speaker_id"]

    def fake_enroll(self, audio, filename, store, requested_id):
        assert audio == b"fake audio"
        assert filename == "sample.wav"
        assert requested_id == speaker_id
        return EnrollmentResult("sample-id", 5.0, 4.0, 0.8)

    monkeypatch.setattr(ResemblyzerEnrollmentService, "enroll", fake_enroll)
    response = client.post(
        f"/api/speakers/{speaker_id}/samples",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "speaker_id": speaker_id,
        "sample_id": "sample-id",
        "status": "ready",
        "duration_s": 5.0,
        "speech_duration_s": 4.0,
        "quality_score": 0.8,
        "embedding_model": "resemblyzer",
        "embedding_dimension": 256,
    }


def test_speaker_api_rejects_invalid_id_and_audio(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/speakers/not-a-uuid").status_code == 422
    assert client.post(
        "/api/speakers", json={"display_name": "   "}
    ).status_code == 422

    speaker_id = client.post(
        "/api/speakers", json={"display_name": "参会人"}
    ).json()["speaker_id"]
    response = client.post(
        f"/api/speakers/{speaker_id}/samples",
        files={"file": ("sample.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_pending_speaker_is_listed_notified_and_approved(tmp_path) -> None:
    from app.core.speaker_registry import SpeakerProfileStore
    import numpy as np

    settings = Settings(speaker_store_path=str(tmp_path / "speakers.sqlite3"))
    store = SpeakerProfileStore(settings.speaker_store_path)
    pending = store.create_pending_candidate(np.ones(256, dtype=np.float32), duration_s=4.0)
    client = TestClient(create_app(settings=settings, enable_pipeline=False, enable_meeting=False))

    listed = client.get("/api/speakers").json()["items"]
    assert listed[0]["status"] == "pending_review"
    with client.websocket_connect("/api/notifications/events") as websocket:
        notifications = websocket.receive_json()
        assert notifications["type"] == "notifications"
        assert notifications["unread_count"] == 1
        assert notifications["items"][0]["speaker_id"] == pending.speaker_id

        approved = client.post(f"/api/speakers/{pending.speaker_id}/approve")
        assert approved.status_code == 200
        assert approved.json()["status"] == "active"

        notifications = websocket.receive_json()
        assert notifications["unread_count"] == 0


def test_speaker_sample_audio_can_be_listed_and_played(tmp_path) -> None:
    from app.core.speaker_registry import SpeakerProfileStore
    import numpy as np

    settings = Settings(
        speaker_store_path=str(tmp_path / "speakers.sqlite3"),
        speaker_sample_dir=str(tmp_path / "samples"),
    )
    store = SpeakerProfileStore(settings.speaker_store_path, sample_dir=settings.speaker_sample_dir)
    profile = store.create("参会人")
    sample_id = store.add_sample(
        profile.speaker_id, np.ones(256, dtype=np.float32), duration_s=3.0,
        speech_duration_s=2.5, quality_score=0.8, audio=b"RIFFsample", filename="clip.wav",
    )
    client = TestClient(create_app(settings=settings, enable_pipeline=False, enable_meeting=False))

    samples = client.get(f"/api/speakers/{profile.speaker_id}/samples")
    assert samples.status_code == 200
    assert samples.json()["items"][0]["audio_url"].endswith(f"/{sample_id}/audio")
    audio = client.get(f"/api/speakers/{profile.speaker_id}/samples/{sample_id}/audio")
    assert audio.status_code == 200
    assert audio.content == b"RIFFsample"
