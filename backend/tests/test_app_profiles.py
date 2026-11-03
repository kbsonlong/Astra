"""The create_app profile split: realtime vs workbench vs all."""
import pytest

from app.config import Settings
from app.main import create_app


def _settings(tmp_path) -> Settings:
    return Settings(
        task_store_path=str(tmp_path / "tasks.sqlite3"),
        speaker_store_path=str(tmp_path / "speakers.sqlite3"),
        speaker_sample_dir=str(tmp_path / "samples"),
        tts_voice_store_path=str(tmp_path / "voices.sqlite3"),
        tts_voice_audio_dir=str(tmp_path / "voice-audio"),
        realtime_recording_dir=str(tmp_path / "recordings"),
        meeting_output_dir=str(tmp_path / "meetings"),
    )


def _paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_unknown_profile_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="profile must be one of"):
        create_app(_settings(tmp_path), enable_pipeline=False, enable_meeting=False, profile="bogus")


def test_realtime_profile_mounts_only_realtime_routes(tmp_path) -> None:
    app = create_app(
        _settings(tmp_path), enable_pipeline=False, enable_meeting=False, profile="realtime"
    )
    paths = _paths(app)
    # Realtime + shared routes present.
    assert "/ws" in paths
    assert "/api/config" in paths
    assert any(path.startswith("/api/tts/voices") for path in paths)
    # Workbench-only routes absent.
    assert not any(path.startswith("/api/meeting") for path in paths)
    assert not any(path.startswith("/api/speakers") for path in paths)
    assert not any(path.startswith("/api/training") for path in paths)
    # Workbench-only state is not built.
    assert app.state.task_store is None
    assert app.state.speaker_store is None
    assert app.state.training_manager is None
    # The shared voice store is still available for snapshot reads.
    assert app.state.tts_voice_store is not None


def test_workbench_profile_mounts_only_workbench_routes(tmp_path) -> None:
    app = create_app(
        _settings(tmp_path), enable_pipeline=False, enable_meeting=False, profile="workbench"
    )
    paths = _paths(app)
    assert "/ws" not in paths
    assert any(path.startswith("/api/meeting") for path in paths)
    assert any(path.startswith("/api/speakers") for path in paths)
    assert any(path.startswith("/api/training") for path in paths)
    # Shared routes still present.
    assert "/api/config" in paths
    assert any(path.startswith("/api/tts/voices") for path in paths)
    # Workbench-only state is built.
    assert app.state.task_store is not None
    assert app.state.speaker_store is not None
    assert app.state.training_manager is not None


def test_all_profile_mounts_everything(tmp_path) -> None:
    app = create_app(
        _settings(tmp_path), enable_pipeline=False, enable_meeting=False, profile="all"
    )
    paths = _paths(app)
    assert "/ws" in paths
    assert any(path.startswith("/api/meeting") for path in paths)
    assert app.state.task_store is not None
    assert app.state.tts_voice_store is not None
