"""Local, revisioned storage for TTS reference voices.

This store deliberately has different semantics from ``SpeakerProfileStore``:
speaker profiles identify people, while TTS voices authorize and reproduce a
reference recording for synthesis.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_VOICE_STATUSES = {"draft", "active", "archived", "revoked"}
_REVISION_LIFECYCLES = {"candidate", "ready", "superseded", "revoked"}
_CONSENT_STATUSES = {"unknown", "confirmed", "revoked"}


class TtsVoiceError(RuntimeError):
    """Base class for expected TTS voice store failures."""


class TtsVoiceNotFoundError(TtsVoiceError, LookupError):
    pass


class TtsVoiceRevisionNotFoundError(TtsVoiceError, LookupError):
    pass


class TtsVoiceValidationError(TtsVoiceError, ValueError):
    pass


class TtsVoiceConflictError(TtsVoiceError):
    pass


class TtsVoicePathError(TtsVoiceValidationError):
    pass


@dataclass(frozen=True)
class TtsVoiceRevision:
    voice_id: str
    revision: int
    lifecycle: str
    backend_family: str
    reference_sha256: str
    original_sha256: str
    original_filename: str
    original_relpath: str
    reference_relpath: str
    mime_type: str
    language: str
    duration_s: float
    speech_duration_s: float | None
    sample_rate: int
    channels: int
    sample_width_bytes: int
    quality_score: float | None
    default_params: dict[str, object]
    consent_status: str
    consent_note: str | None
    consent_confirmed_at: str | None
    created_at: str
    superseded_at: str | None


@dataclass(frozen=True)
class TtsVoice:
    voice_id: str
    display_name: str
    status: str
    current_revision: int | None
    created_at: str
    updated_at: str
    archived_at: str | None
    revoked_at: str | None
    revision: TtsVoiceRevision | None


@dataclass(frozen=True)
class TtsVoiceSnapshot:
    voice_id: str
    voice_revision: int
    backend_family: str
    language: str
    reference_path: Path
    reference_sha256: str
    default_params: dict[str, object]


class TtsVoiceStore:
    """SQLite-backed TTS voice identities and immutable reference revisions."""

    def __init__(self, path: str | Path, audio_dir: str | Path) -> None:
        if str(path) == ":memory:":
            raise ValueError("TtsVoiceStore requires a file-backed SQLite database")
        self.path = Path(path).expanduser()
        self.audio_dir = Path(audio_dir).expanduser()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tts_voices (
                voice_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft', 'active', 'archived', 'revoked')),
                current_revision INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                archived_at TEXT,
                revoked_at TEXT,
                CHECK (length(trim(display_name)) > 0)
            );
            CREATE INDEX IF NOT EXISTS idx_tts_voices_status_updated
                ON tts_voices(status, updated_at DESC);

            CREATE TABLE IF NOT EXISTS tts_voice_revisions (
                voice_id TEXT NOT NULL REFERENCES tts_voices(voice_id),
                revision INTEGER NOT NULL,
                lifecycle TEXT NOT NULL DEFAULT 'candidate'
                    CHECK (lifecycle IN ('candidate', 'ready', 'superseded', 'revoked')),
                backend_family TEXT NOT NULL,
                reference_sha256 TEXT NOT NULL,
                original_sha256 TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                original_relpath TEXT NOT NULL,
                reference_relpath TEXT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT 'audio/wav',
                language TEXT NOT NULL DEFAULT 'zh',
                duration_s REAL NOT NULL CHECK (duration_s > 0),
                speech_duration_s REAL,
                sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
                channels INTEGER NOT NULL CHECK (channels > 0),
                sample_width_bytes INTEGER NOT NULL CHECK (sample_width_bytes > 0),
                quality_score REAL,
                default_params_json TEXT NOT NULL DEFAULT '{}',
                consent_status TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (consent_status IN ('unknown', 'confirmed', 'revoked')),
                consent_note TEXT,
                consent_confirmed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                superseded_at TEXT,
                PRIMARY KEY (voice_id, revision),
                CHECK (json_valid(default_params_json)),
                CHECK (
                    consent_status != 'confirmed'
                    OR consent_confirmed_at IS NOT NULL
                )
            );
            CREATE INDEX IF NOT EXISTS idx_tts_voice_revisions_sha
                ON tts_voice_revisions(reference_sha256);
            CREATE INDEX IF NOT EXISTS idx_tts_voice_revisions_backend
                ON tts_voice_revisions(backend_family, lifecycle);

            CREATE TABLE IF NOT EXISTS tts_voice_events (
                event_id TEXT PRIMARY KEY,
                voice_id TEXT NOT NULL REFERENCES tts_voices(voice_id),
                revision INTEGER,
                event_type TEXT NOT NULL,
                detail_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (json_valid(detail_json))
            );
            CREATE INDEX IF NOT EXISTS idx_tts_voice_events_voice_time
                ON tts_voice_events(voice_id, created_at DESC);
            """
        )
        return connection

    def create_draft(
        self,
        display_name: str,
        *,
        backend_family: str,
        language: str,
        reference_sha256: str,
        original_sha256: str,
        original_filename: str,
        original_relpath: str,
        reference_relpath: str,
        duration_s: float,
        speech_duration_s: float | None = None,
        sample_rate: int = 16_000,
        channels: int = 1,
        sample_width_bytes: int = 2,
        quality_score: float | None = None,
        default_params: Mapping[str, object] | None = None,
        mime_type: str = "audio/wav",
    ) -> TtsVoice:
        """Create a draft voice and its first candidate revision.

        The caller must place both audio files under ``audio_dir`` before this
        call. The store validates that they are regular files within that root.
        Activation is a separate operation and requires confirmed consent.
        """
        name = self._display_name(display_name)
        fields = self._validate_revision_fields(
            backend_family=backend_family,
            language=language,
            reference_sha256=reference_sha256,
            original_sha256=original_sha256,
            original_filename=original_filename,
            original_relpath=original_relpath,
            reference_relpath=reference_relpath,
            duration_s=duration_s,
            speech_duration_s=speech_duration_s,
            sample_rate=sample_rate,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
            quality_score=quality_score,
            default_params=default_params,
            mime_type=mime_type,
        )
        self._require_audio_file(fields["original_relpath"])
        self._require_audio_file(fields["reference_relpath"])

        voice_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tts_voices(voice_id, display_name) VALUES (?, ?)",
                (voice_id, name),
            )
            self._insert_revision(connection, voice_id, 1, fields)
            self._event(connection, voice_id, 1, "created", {"status": "draft"})
        return self.get(voice_id)

    def add_revision(
        self,
        voice_id: str,
        *,
        backend_family: str,
        language: str,
        reference_sha256: str,
        original_sha256: str,
        original_filename: str,
        original_relpath: str,
        reference_relpath: str,
        duration_s: float,
        speech_duration_s: float | None = None,
        sample_rate: int = 16_000,
        channels: int = 1,
        sample_width_bytes: int = 2,
        quality_score: float | None = None,
        default_params: Mapping[str, object] | None = None,
        mime_type: str = "audio/wav",
    ) -> TtsVoiceRevision:
        """Add an immutable candidate revision without changing the active one."""
        fields = self._validate_revision_fields(
            backend_family=backend_family,
            language=language,
            reference_sha256=reference_sha256,
            original_sha256=original_sha256,
            original_filename=original_filename,
            original_relpath=original_relpath,
            reference_relpath=reference_relpath,
            duration_s=duration_s,
            speech_duration_s=speech_duration_s,
            sample_rate=sample_rate,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
            quality_score=quality_score,
            default_params=default_params,
            mime_type=mime_type,
        )
        self._require_audio_file(fields["original_relpath"])
        self._require_audio_file(fields["reference_relpath"])

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            voice = connection.execute(
                "SELECT status FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone()
            if voice is None:
                raise TtsVoiceNotFoundError(voice_id)
            if voice["status"] == "revoked":
                raise TtsVoiceConflictError("cannot add a revision to a revoked voice")
            next_revision = int(
                connection.execute(
                    "SELECT COALESCE(MAX(revision), 0) + 1 "
                    "FROM tts_voice_revisions WHERE voice_id = ?",
                    (voice_id,),
                ).fetchone()[0]
            )
            self._insert_revision(connection, voice_id, next_revision, fields)
            connection.execute(
                "UPDATE tts_voices SET updated_at = CURRENT_TIMESTAMP WHERE voice_id = ?",
                (voice_id,),
            )
            self._event(connection, voice_id, next_revision, "revision_created", {})
        return self.get_revision(voice_id, next_revision)

    def get(self, voice_id: str) -> TtsVoice:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone()
            if row is None:
                raise TtsVoiceNotFoundError(voice_id)
            return self._voice_from_row(connection, row)

    def list(
        self,
        *,
        status: str | None = None,
        language: str | None = None,
        backend_family: str | None = None,
        query: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[TtsVoice], dict[str, int]]:
        if page < 1 or page_size < 1 or page_size > 200:
            raise TtsVoiceValidationError("page must be >= 1 and page_size must be 1..200")
        if status is not None and status not in _VOICE_STATUSES:
            raise TtsVoiceValidationError("unsupported voice status")
        where: list[str] = []
        params: list[object] = []
        if status:
            where.append("v.status = ?")
            params.append(status)
        if language:
            where.append("r.language = ?")
            params.append(language.strip().lower())
        if backend_family:
            where.append("r.backend_family = ?")
            params.append(backend_family.strip())
        if query and query.strip():
            where.append("v.display_name LIKE ?")
            params.append(f"%{query.strip()}%")
        clause = " WHERE " + " AND ".join(where) if where else ""
        revision_join = (
            " LEFT JOIN tts_voice_revisions r ON r.voice_id = v.voice_id "
            "AND r.revision = COALESCE(v.current_revision, "
            "(SELECT MAX(r2.revision) FROM tts_voice_revisions r2 "
            "WHERE r2.voice_id = v.voice_id))"
        )
        with self._connect() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT v.voice_id) FROM tts_voices v "
                    + revision_join
                    + clause,
                    params,
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT DISTINCT v.* FROM tts_voices v "
                + revision_join
                + clause
                + " ORDER BY v.updated_at DESC, v.voice_id "
                + "LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
            result = [self._voice_from_row(connection, row) for row in rows]
        pages = (total + page_size - 1) // page_size
        return result, {"page": page, "page_size": page_size, "total": total, "pages": pages}

    def get_revision(self, voice_id: str, revision: int) -> TtsVoiceRevision:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tts_voice_revisions WHERE voice_id = ? AND revision = ?",
                (voice_id, revision),
            ).fetchone()
            if row is None:
                raise TtsVoiceRevisionNotFoundError(f"{voice_id}:{revision}")
            return self._revision(row)

    def list_revisions(self, voice_id: str) -> list[TtsVoiceRevision]:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone() is None:
                raise TtsVoiceNotFoundError(voice_id)
            rows = connection.execute(
                "SELECT * FROM tts_voice_revisions WHERE voice_id = ? "
                "ORDER BY revision DESC",
                (voice_id,),
            ).fetchall()
        return [self._revision(row) for row in rows]

    def update_metadata(self, voice_id: str, display_name: str) -> TtsVoice:
        name = self._display_name(display_name)
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE tts_voices SET display_name = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE voice_id = ?",
                (name, voice_id),
            )
            if result.rowcount == 0:
                raise TtsVoiceNotFoundError(voice_id)
        return self.get(voice_id)

    def confirm_consent(
        self, voice_id: str, revision: int, *, note: str | None = None
    ) -> TtsVoiceRevision:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._revision_row(connection, voice_id, revision)
            if row["lifecycle"] == "revoked":
                raise TtsVoiceConflictError("cannot confirm consent for a revoked revision")
            connection.execute(
                "UPDATE tts_voice_revisions SET consent_status = 'confirmed', "
                "consent_note = ?, consent_confirmed_at = CURRENT_TIMESTAMP "
                "WHERE voice_id = ? AND revision = ?",
                (note.strip() if note else None, voice_id, revision),
            )
            connection.execute(
                "UPDATE tts_voices SET updated_at = CURRENT_TIMESTAMP WHERE voice_id = ?",
                (voice_id,),
            )
            self._event(connection, voice_id, revision, "consent_confirmed", {})
        return self.get_revision(voice_id, revision)

    def revoke_consent(self, voice_id: str, revision: int, *, note: str | None = None) -> TtsVoiceRevision:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._revision_row(connection, voice_id, revision)
            connection.execute(
                "UPDATE tts_voice_revisions SET consent_status = 'revoked', "
                "consent_note = ?, lifecycle = 'revoked' "
                "WHERE voice_id = ? AND revision = ?",
                (note.strip() if note else None, voice_id, revision),
            )
            if row["current_revision"] == revision:
                connection.execute(
                    "UPDATE tts_voices SET status = 'revoked', revoked_at = CURRENT_TIMESTAMP, "
                    "updated_at = CURRENT_TIMESTAMP WHERE voice_id = ?",
                    (voice_id,),
                )
            else:
                connection.execute(
                    "UPDATE tts_voices SET updated_at = CURRENT_TIMESTAMP WHERE voice_id = ?",
                    (voice_id,),
                )
            self._event(connection, voice_id, revision, "consent_revoked", {})
        return self.get_revision(voice_id, revision)

    def activate(self, voice_id: str, revision: int | None = None) -> TtsVoice:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            voice = connection.execute(
                "SELECT * FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone()
            if voice is None:
                raise TtsVoiceNotFoundError(voice_id)
            if voice["status"] == "revoked":
                raise TtsVoiceConflictError("cannot activate a revoked voice")
            target = self._select_revision_row(connection, voice_id, revision)
            if target["lifecycle"] == "revoked" or target["consent_status"] != "confirmed":
                raise TtsVoiceConflictError("revision requires confirmed consent")
            self._require_audio_file(target["reference_relpath"])
            old_revision = voice["current_revision"]
            if old_revision is not None and old_revision != target["revision"]:
                connection.execute(
                    "UPDATE tts_voice_revisions SET lifecycle = 'superseded', "
                    "superseded_at = CURRENT_TIMESTAMP WHERE voice_id = ? "
                    "AND revision = ? AND lifecycle = 'ready'",
                    (voice_id, old_revision),
                )
            connection.execute(
                "UPDATE tts_voice_revisions SET lifecycle = 'ready', superseded_at = NULL "
                "WHERE voice_id = ? AND revision = ?",
                (voice_id, target["revision"]),
            )
            connection.execute(
                "UPDATE tts_voices SET status = 'active', current_revision = ?, "
                "archived_at = NULL, revoked_at = NULL, updated_at = CURRENT_TIMESTAMP "
                "WHERE voice_id = ?",
                (target["revision"], voice_id),
            )
            self._event(connection, voice_id, target["revision"], "activated", {})
        return self.get(voice_id)

    def archive(self, voice_id: str) -> TtsVoice:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone()
            if row is None:
                raise TtsVoiceNotFoundError(voice_id)
            if row["status"] != "active":
                raise TtsVoiceConflictError("only an active voice can be archived")
            connection.execute(
                "UPDATE tts_voices SET status = 'archived', archived_at = CURRENT_TIMESTAMP, "
                "updated_at = CURRENT_TIMESTAMP WHERE voice_id = ?",
                (voice_id,),
            )
            self._event(connection, voice_id, None, "archived", {})
        return self.get(voice_id)

    def restore(self, voice_id: str) -> TtsVoice:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            voice = connection.execute(
                "SELECT * FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone()
            if voice is None:
                raise TtsVoiceNotFoundError(voice_id)
            if voice["status"] != "archived":
                raise TtsVoiceConflictError("only an archived voice can be restored")
            target = self._select_revision_row(connection, voice_id, voice["current_revision"])
            if target["lifecycle"] != "ready" or target["consent_status"] != "confirmed":
                raise TtsVoiceConflictError("current revision is not usable")
            self._require_audio_file(target["reference_relpath"])
            connection.execute(
                "UPDATE tts_voices SET status = 'active', archived_at = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE voice_id = ?",
                (voice_id,),
            )
            self._event(connection, voice_id, target["revision"], "restored", {})
        return self.get(voice_id)

    def snapshot(self, voice_id: str, revision: int | None = None) -> TtsVoiceSnapshot:
        with self._connect() as connection:
            voice = connection.execute(
                "SELECT * FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone()
            if voice is None:
                raise TtsVoiceNotFoundError(voice_id)
            if voice["status"] != "active":
                raise TtsVoiceConflictError("voice is not active")
            target = self._select_revision_row(connection, voice_id, revision or voice["current_revision"])
            if target["lifecycle"] != "ready" or target["consent_status"] != "confirmed":
                raise TtsVoiceConflictError("revision is not ready or consent is not confirmed")
            path = self._require_audio_file(target["reference_relpath"])
            return TtsVoiceSnapshot(
                voice_id=voice_id,
                voice_revision=int(target["revision"]),
                backend_family=target["backend_family"],
                language=target["language"],
                reference_path=path,
                reference_sha256=target["reference_sha256"],
                default_params=self._decode_params(target["default_params_json"]),
            )

    def resolve_audio_path(self, voice_id: str, revision: int, kind: str = "reference") -> Path:
        if kind not in {"original", "reference"}:
            raise TtsVoiceValidationError("audio kind must be original or reference")
        row = self._revision_row_for_public_read(voice_id, revision)
        if row["voice_status"] == "revoked":
            raise TtsVoiceConflictError("revoked voice audio is not available")
        return self._require_audio_file(row[f"{kind}_relpath"])

    def _voice_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> TtsVoice:
        revision = None
        revision_number = row["current_revision"]
        if revision_number is None:
            latest = connection.execute(
                "SELECT MAX(revision) FROM tts_voice_revisions WHERE voice_id = ?",
                (row["voice_id"],),
            ).fetchone()
            revision_number = latest[0] if latest is not None else None
        if revision_number is not None:
            current = connection.execute(
                "SELECT * FROM tts_voice_revisions WHERE voice_id = ? AND revision = ?",
                (row["voice_id"], revision_number),
            ).fetchone()
            if current is not None:
                revision = self._revision(current)
        return TtsVoice(
            voice_id=row["voice_id"],
            display_name=row["display_name"],
            status=row["status"],
            current_revision=(
                int(row["current_revision"]) if row["current_revision"] is not None else None
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived_at=row["archived_at"],
            revoked_at=row["revoked_at"],
            revision=revision,
        )

    @staticmethod
    def _revision(row: sqlite3.Row) -> TtsVoiceRevision:
        return TtsVoiceRevision(
            voice_id=row["voice_id"],
            revision=int(row["revision"]),
            lifecycle=row["lifecycle"],
            backend_family=row["backend_family"],
            reference_sha256=row["reference_sha256"],
            original_sha256=row["original_sha256"],
            original_filename=row["original_filename"],
            original_relpath=row["original_relpath"],
            reference_relpath=row["reference_relpath"],
            mime_type=row["mime_type"],
            language=row["language"],
            duration_s=float(row["duration_s"]),
            speech_duration_s=(
                float(row["speech_duration_s"])
                if row["speech_duration_s"] is not None
                else None
            ),
            sample_rate=int(row["sample_rate"]),
            channels=int(row["channels"]),
            sample_width_bytes=int(row["sample_width_bytes"]),
            quality_score=(
                float(row["quality_score"]) if row["quality_score"] is not None else None
            ),
            default_params=TtsVoiceStore._decode_params(row["default_params_json"]),
            consent_status=row["consent_status"],
            consent_note=row["consent_note"],
            consent_confirmed_at=row["consent_confirmed_at"],
            created_at=row["created_at"],
            superseded_at=row["superseded_at"],
        )

    def _revision_row(self, connection: sqlite3.Connection, voice_id: str, revision: int) -> sqlite3.Row:
        row = connection.execute(
            "SELECT r.*, v.current_revision FROM tts_voice_revisions r "
            "JOIN tts_voices v ON v.voice_id = r.voice_id "
            "WHERE r.voice_id = ? AND r.revision = ?",
            (voice_id, revision),
        ).fetchone()
        if row is None:
            if connection.execute(
                "SELECT 1 FROM tts_voices WHERE voice_id = ?", (voice_id,)
            ).fetchone() is None:
                raise TtsVoiceNotFoundError(voice_id)
            raise TtsVoiceRevisionNotFoundError(f"{voice_id}:{revision}")
        return row

    def _revision_row_for_public_read(self, voice_id: str, revision: int) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT r.*, v.current_revision, v.status AS voice_status "
                "FROM tts_voice_revisions r JOIN tts_voices v ON v.voice_id = r.voice_id "
                "WHERE r.voice_id = ? AND r.revision = ?",
                (voice_id, revision),
            ).fetchone()
            if row is None:
                if connection.execute(
                    "SELECT 1 FROM tts_voices WHERE voice_id = ?", (voice_id,)
                ).fetchone() is None:
                    raise TtsVoiceNotFoundError(voice_id)
                raise TtsVoiceRevisionNotFoundError(f"{voice_id}:{revision}")
            return row

    def _select_revision_row(
        self, connection: sqlite3.Connection, voice_id: str, revision: int | None
    ) -> sqlite3.Row:
        if revision is None:
            row = connection.execute(
                "SELECT r.*, v.current_revision FROM tts_voice_revisions r "
                "JOIN tts_voices v ON v.voice_id = r.voice_id "
                "WHERE r.voice_id = ? ORDER BY r.revision DESC LIMIT 1",
                (voice_id,),
            ).fetchone()
        else:
            row = self._revision_row(connection, voice_id, int(revision))
        if row is None:
            raise TtsVoiceRevisionNotFoundError(voice_id)
        return row

    def _insert_revision(
        self, connection: sqlite3.Connection, voice_id: str, revision: int, fields: dict[str, object]
    ) -> None:
        connection.execute(
            "INSERT INTO tts_voice_revisions "
            "(voice_id, revision, backend_family, reference_sha256, original_sha256, "
            "original_filename, original_relpath, reference_relpath, mime_type, language, "
            "duration_s, speech_duration_s, sample_rate, channels, sample_width_bytes, "
            "quality_score, default_params_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                voice_id,
                revision,
                fields["backend_family"],
                fields["reference_sha256"],
                fields["original_sha256"],
                fields["original_filename"],
                fields["original_relpath"],
                fields["reference_relpath"],
                fields["mime_type"],
                fields["language"],
                fields["duration_s"],
                fields["speech_duration_s"],
                fields["sample_rate"],
                fields["channels"],
                fields["sample_width_bytes"],
                fields["quality_score"],
                fields["default_params_json"],
            ),
        )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        voice_id: str,
        revision: int | None,
        event_type: str,
        detail: Mapping[str, object],
    ) -> None:
        connection.execute(
            "INSERT INTO tts_voice_events(event_id, voice_id, revision, event_type, detail_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), voice_id, revision, event_type, json.dumps(detail, ensure_ascii=False)),
        )

    def _require_audio_file(self, relative_path: str) -> Path:
        relative = self._relative_path(relative_path)
        root = self.audio_dir.resolve()
        candidate = self.audio_dir / relative
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise TtsVoicePathError("audio path escapes configured audio directory")
        if candidate.is_symlink() or not candidate.is_file():
            raise TtsVoicePathError("audio file is missing or is not a regular file")
        return resolved

    @staticmethod
    def _display_name(value: str) -> str:
        name = value.strip()
        if not name or len(name) > 128:
            raise TtsVoiceValidationError("display_name must be 1..128 characters")
        return name

    @staticmethod
    def _relative_path(value: str) -> str:
        raw = str(value).strip()
        path = Path(raw)
        if not raw or path.is_absolute() or ".." in path.parts:
            raise TtsVoicePathError("audio path must be a relative path without '..'")
        return path.as_posix()

    @staticmethod
    def _decode_params(value: str) -> dict[str, object]:
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise TtsVoiceValidationError("default_params_json is invalid") from exc
        if not isinstance(parsed, dict):
            raise TtsVoiceValidationError("default_params must be an object")
        return dict(parsed)

    @classmethod
    def _validate_revision_fields(cls, **values: object) -> dict[str, object]:
        backend = str(values["backend_family"]).strip()
        language = str(values["language"]).strip().lower()
        if not backend or not language:
            raise TtsVoiceValidationError("backend_family and language are required")
        for field in ("reference_sha256", "original_sha256"):
            digest = str(values[field]).strip().lower()
            if not _SHA256.fullmatch(digest):
                raise TtsVoiceValidationError(f"{field} must be a SHA-256 hex digest")
            values[field] = digest
        original_filename = Path(str(values["original_filename"])).name
        if not original_filename:
            raise TtsVoiceValidationError("original_filename is required")
        values["original_filename"] = original_filename
        values["original_relpath"] = cls._relative_path(str(values["original_relpath"]))
        values["reference_relpath"] = cls._relative_path(str(values["reference_relpath"]))
        duration = float(values["duration_s"])
        if duration <= 0:
            raise TtsVoiceValidationError("duration_s must be greater than 0")
        speech = values["speech_duration_s"]
        if speech is not None and (float(speech) < 0 or float(speech) > duration):
            raise TtsVoiceValidationError("speech_duration_s must be within duration_s")
        sample_rate = int(values["sample_rate"])
        channels = int(values["channels"])
        sample_width = int(values["sample_width_bytes"])
        if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
            raise TtsVoiceValidationError("audio format values must be positive")
        quality = values["quality_score"]
        if quality is not None and not 0 <= float(quality) <= 1:
            raise TtsVoiceValidationError("quality_score must be between 0 and 1")
        params = values["default_params"] or {}
        if not isinstance(params, Mapping):
            raise TtsVoiceValidationError("default_params must be an object")
        try:
            params_json = json.dumps(dict(params), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TtsVoiceValidationError("default_params must be JSON serializable") from exc
        mime_type = str(values["mime_type"]).strip() or "audio/wav"
        return {
            "backend_family": backend,
            "language": language,
            "reference_sha256": values["reference_sha256"],
            "original_sha256": values["original_sha256"],
            "original_filename": original_filename,
            "original_relpath": values["original_relpath"],
            "reference_relpath": values["reference_relpath"],
            "duration_s": duration,
            "speech_duration_s": float(speech) if speech is not None else None,
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "quality_score": float(quality) if quality is not None else None,
            "default_params_json": params_json,
            "mime_type": mime_type,
        }
