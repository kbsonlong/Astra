from pathlib import Path

import pytest

from app.core.tts_voice_store import (
    TtsVoiceConflictError,
    TtsVoicePathError,
    TtsVoiceStore,
    TtsVoiceValidationError,
)


def _sha(char: str) -> str:
    return char * 64


def _audio_files(tmp_path: Path, revision: int = 1) -> tuple[str, str]:
    root = tmp_path / "audio" / "voice-1"
    root.mkdir(parents=True, exist_ok=True)
    original = root / f"rev-{revision:04d}-original.wav"
    reference = root / f"rev-{revision:04d}-reference.wav"
    original.write_bytes(b"original")
    reference.write_bytes(b"reference")
    return str(original.relative_to(tmp_path / "audio")), str(
        reference.relative_to(tmp_path / "audio")
    )


def _create_store(tmp_path: Path) -> tuple[TtsVoiceStore, object]:
    store = TtsVoiceStore(tmp_path / "voices.sqlite3", tmp_path / "audio")
    original, reference = _audio_files(tmp_path)
    voice = store.create_draft(
        "小雅",
        backend_family="indextts25_mlx",
        language="zh",
        reference_sha256=_sha("a"),
        original_sha256=_sha("b"),
        original_filename="recording.wav",
        original_relpath=original,
        reference_relpath=reference,
        duration_s=8.4,
        speech_duration_s=7.9,
        default_params={"greedy": True},
    )
    return store, voice


def test_create_draft_persists_candidate_revision(tmp_path: Path) -> None:
    store, voice = _create_store(tmp_path)

    assert voice.status == "draft"
    assert voice.current_revision is None
    assert voice.revision is not None
    assert voice.revision.lifecycle == "candidate"
    assert voice.revision.consent_status == "unknown"
    assert voice.revision.default_params == {"greedy": True}


def test_add_revision_is_immutable_and_does_not_switch_current(tmp_path: Path) -> None:
    store, voice = _create_store(tmp_path)
    original, reference = _audio_files(tmp_path, revision=2)

    revision = store.add_revision(
        voice.voice_id,
        backend_family="indextts25_mlx",
        language="zh",
        reference_sha256=_sha("c"),
        original_sha256=_sha("d"),
        original_filename="replacement.wav",
        original_relpath=original,
        reference_relpath=reference,
        duration_s=9.0,
    )

    assert revision.revision == 2
    assert store.get(voice.voice_id).current_revision is None
    assert store.get_revision(voice.voice_id, 1).reference_sha256 == _sha("a")


def test_consent_activation_snapshot_and_archive_restore(tmp_path: Path) -> None:
    store, voice = _create_store(tmp_path)
    store.confirm_consent(voice.voice_id, 1, note="本人授权")
    active = store.activate(voice.voice_id, 1)

    assert active.status == "active"
    snapshot = store.snapshot(voice.voice_id)
    assert snapshot.voice_revision == 1
    assert snapshot.reference_path.is_file()
    assert snapshot.default_params == {"greedy": True}

    archived = store.archive(voice.voice_id)
    assert archived.status == "archived"
    with pytest.raises(TtsVoiceConflictError):
        store.snapshot(voice.voice_id)
    assert store.restore(voice.voice_id).status == "active"


def test_unconfirmed_revision_cannot_activate_or_snapshot(tmp_path: Path) -> None:
    store, voice = _create_store(tmp_path)

    with pytest.raises(TtsVoiceConflictError):
        store.activate(voice.voice_id, 1)
    with pytest.raises(TtsVoiceConflictError):
        store.snapshot(voice.voice_id)


def test_revoke_current_consent_blocks_future_snapshot(tmp_path: Path) -> None:
    store, voice = _create_store(tmp_path)
    store.confirm_consent(voice.voice_id, 1)
    store.activate(voice.voice_id, 1)
    revision = store.revoke_consent(voice.voice_id, 1, note="撤销授权")

    assert revision.consent_status == "revoked"
    assert store.get(voice.voice_id).status == "revoked"
    with pytest.raises(TtsVoiceConflictError):
        store.snapshot(voice.voice_id)
    with pytest.raises(TtsVoiceConflictError):
        store.resolve_audio_path(voice.voice_id, 1)


def test_path_traversal_and_missing_files_are_rejected(tmp_path: Path) -> None:
    store = TtsVoiceStore(tmp_path / "voices.sqlite3", tmp_path / "audio")
    with pytest.raises(TtsVoicePathError):
        store.create_draft(
            "bad",
            backend_family="indextts25_mlx",
            language="zh",
            reference_sha256=_sha("a"),
            original_sha256=_sha("b"),
            original_filename="bad.wav",
            original_relpath="../outside.wav",
            reference_relpath="voice/reference.wav",
            duration_s=1,
        )
    with pytest.raises(TtsVoicePathError):
        store.create_draft(
            "missing",
            backend_family="indextts25_mlx",
            language="zh",
            reference_sha256=_sha("a"),
            original_sha256=_sha("b"),
            original_filename="missing.wav",
            original_relpath="voice/original.wav",
            reference_relpath="voice/reference.wav",
            duration_s=1,
        )


def test_invalid_audio_metadata_and_params_are_rejected(tmp_path: Path) -> None:
    store = TtsVoiceStore(tmp_path / "voices.sqlite3", tmp_path / "audio")
    original, reference = _audio_files(tmp_path)
    with pytest.raises(TtsVoiceValidationError):
        store.create_draft(
            "bad",
            backend_family="indextts25_mlx",
            language="zh",
            reference_sha256="not-a-sha",
            original_sha256=_sha("b"),
            original_filename="bad.wav",
            original_relpath=original,
            reference_relpath=reference,
            duration_s=1,
        )
    with pytest.raises(TtsVoiceValidationError):
        store.create_draft(
            "bad",
            backend_family="indextts25_mlx",
            language="zh",
            reference_sha256=_sha("a"),
            original_sha256=_sha("b"),
            original_filename="bad.wav",
            original_relpath=original,
            reference_relpath=reference,
            duration_s=1,
            speech_duration_s=2,
        )


def test_list_filters_and_paginates(tmp_path: Path) -> None:
    store, voice = _create_store(tmp_path)
    items, page = store.list(status="draft", language="zh", page=1, page_size=10)

    assert [item.voice_id for item in items] == [voice.voice_id]
    assert page == {"page": 1, "page_size": 10, "total": 1, "pages": 1}


def test_resolve_audio_path_rejects_symlink(tmp_path: Path) -> None:
    store, voice = _create_store(tmp_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    reference = tmp_path / "audio" / "voice-1" / "rev-0001-reference.wav"
    reference.unlink()
    reference.symlink_to(outside)

    with pytest.raises(TtsVoicePathError):
        store.resolve_audio_path(voice.voice_id, 1)
