"""可复用的 UUID 声纹档案和本地样本注册服务。"""
from __future__ import annotations

import sqlite3
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


class SpeakerNotFoundError(LookupError):
    pass


class SpeakerEnrollmentError(ValueError):
    pass


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    display_name: str
    status: str
    sample_count: int
    embedding_model: str
    updated_at: str


@dataclass(frozen=True)
class SpeakerMatch:
    speaker_id: str
    display_name: str
    similarity: float
    confidence: str


@dataclass(frozen=True)
class SpeakerSample:
    sample_id: str
    speaker_id: str
    duration_s: float
    speech_duration_s: float
    quality_score: float | None
    original_filename: str
    audio_path: str | None
    created_at: str


class SpeakerProfileStore:
    """SQLite 存储声纹和可供管理员复核的录入样本。"""

    def __init__(
        self,
        path: str | Path,
        *,
        match_threshold: float = 0.75,
        match_margin: float = 0.05,
        duplicate_threshold: float = 0.82,
        sample_dir: str | Path = "~/.astra/speaker_samples",
    ) -> None:
        self.path = Path(path).expanduser()
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.duplicate_threshold = duplicate_threshold
        self.sample_dir = Path(sample_dir).expanduser()

    def _connect(self) -> sqlite3.Connection:
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS speaker_profiles (
                speaker_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                embedding_model TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS speaker_samples (
                sample_id TEXT PRIMARY KEY,
                speaker_id TEXT NOT NULL REFERENCES speaker_profiles(speaker_id),
                embedding BLOB NOT NULL,
                duration_s REAL NOT NULL,
                speech_duration_s REAL NOT NULL,
                quality_score REAL,
                original_filename TEXT NOT NULL DEFAULT 'sample.wav',
                audio_path TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_speaker_samples_speaker
                ON speaker_samples(speaker_id);
            """
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS speaker_notifications (
                notification_id TEXT PRIMARY KEY,
                speaker_id TEXT NOT NULL REFERENCES speaker_profiles(speaker_id),
                notification_type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT
            )"""
        )
        return connection

    @staticmethod
    def _profile(row: sqlite3.Row) -> SpeakerProfile:
        return SpeakerProfile(
            speaker_id=row["speaker_id"],
            display_name=row["display_name"],
            status=row["status"],
            sample_count=int(row["sample_count"]),
            embedding_model=row["embedding_model"],
            updated_at=row["updated_at"],
        )

    def create(self, display_name: str) -> SpeakerProfile:
        name = display_name.strip()
        if not name:
            raise ValueError("display_name is required")
        speaker_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO speaker_profiles "
                "(speaker_id, display_name, embedding_model, embedding_dimension) "
                "VALUES (?, ?, ?, ?)",
                (speaker_id, name, "resemblyzer", 256),
            )
        return self.get(speaker_id)

    def create_pending_candidate(
        self,
        embedding: object,
        *,
        duration_s: float,
        quality_score: float = 0.5,
        audio: bytes | None = None,
        filename: str = "meeting-cluster.wav",
    ) -> SpeakerProfile:
        """Register an unmatched meeting cluster for administrator review."""
        import numpy as np

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size != 256 or norm == 0:
            raise SpeakerEnrollmentError("invalid resemblyzer embedding")
        vector = vector / norm
        with self._connect() as connection:
            speaker_id = str(uuid.uuid4())
            display_name = f"待审核-{speaker_id[:8]}"
            connection.execute(
                "INSERT INTO speaker_profiles "
                "(speaker_id, display_name, status, embedding_model, embedding_dimension) "
                "VALUES (?, ?, 'pending_review', ?, ?)",
                (speaker_id, display_name, "resemblyzer", 256),
            )
            sample_id = str(uuid.uuid4())
            audio_path = self._save_audio(speaker_id, sample_id, audio, filename)
            connection.execute(
                "INSERT INTO speaker_samples "
                "(sample_id, speaker_id, embedding, duration_s, speech_duration_s, quality_score, original_filename, audio_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sample_id, speaker_id, vector.tobytes(), duration_s, duration_s,
                 quality_score, Path(filename).name, audio_path),
            )
            self._notify(
                connection,
                speaker_id,
                "speaker_review",
                f"{display_name} 已从会议录音中提取，请播放样本、确认身份并改名",
            )
        return self.get(speaker_id)

    def register_or_append_candidate(
        self,
        embedding: object,
        *,
        duration_s: float,
        quality_score: float = 0.5,
        audio: bytes | None = None,
        filename: str = "meeting-cluster.wav",
    ) -> SpeakerProfile:
        """Reuse a similar active/pending profile, otherwise create a review candidate."""
        duplicate = self.find_similar(embedding, threshold=self.duplicate_threshold)
        if duplicate is not None:
            self.add_sample(
                duplicate.speaker_id,
                embedding,
                duration_s=duration_s,
                speech_duration_s=duration_s,
                quality_score=quality_score,
                audio=audio,
                filename=filename,
            )
            with self._connect() as connection:
                self._notify(
                    connection,
                    duplicate.speaker_id,
                    "speaker_sample_supplement",
                    f"新会议声纹与 {duplicate.display_name} 相似，请播放新样本并补充确认",
                )
            return self.get(duplicate.speaker_id)
        return self.create_pending_candidate(
            embedding, duration_s=duration_s, quality_score=quality_score,
            audio=audio, filename=filename,
        )

    @staticmethod
    def _notify(connection: sqlite3.Connection, speaker_id: str, notification_type: str, message: str) -> None:
        exists = connection.execute(
            "SELECT 1 FROM speaker_notifications WHERE speaker_id = ? "
            "AND notification_type = ? AND resolved_at IS NULL LIMIT 1",
            (speaker_id, notification_type),
        ).fetchone()
        if exists is None:
            connection.execute(
                "INSERT INTO speaker_notifications "
                "(notification_id, speaker_id, notification_type, message) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), speaker_id, notification_type, message),
            )

    def list(self) -> list[SpeakerProfile]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.*, COUNT(s.sample_id) AS sample_count "
                "FROM speaker_profiles p LEFT JOIN speaker_samples s "
                "ON s.speaker_id = p.speaker_id "
                "WHERE p.status IN ('active', 'pending_review') "
                "GROUP BY p.speaker_id ORDER BY p.created_at"
            ).fetchall()
        return [self._profile(row) for row in rows]

    def get(self, speaker_id: str) -> SpeakerProfile:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT p.*, COUNT(s.sample_id) AS sample_count "
                "FROM speaker_profiles p LEFT JOIN speaker_samples s "
                "ON s.speaker_id = p.speaker_id "
                "WHERE p.speaker_id = ? GROUP BY p.speaker_id",
                (speaker_id,),
            ).fetchone()
        if row is None:
            raise SpeakerNotFoundError(speaker_id)
        return self._profile(row)

    def disable(self, speaker_id: str) -> SpeakerProfile:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE speaker_profiles SET status = 'disabled', "
                "updated_at = CURRENT_TIMESTAMP WHERE speaker_id = ?",
                (speaker_id,),
            )
            if result.rowcount == 0:
                raise SpeakerNotFoundError(speaker_id)
        return self.get(speaker_id)

    def approve(self, speaker_id: str) -> SpeakerProfile:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE speaker_profiles SET status = 'active', "
                "updated_at = CURRENT_TIMESTAMP WHERE speaker_id = ?",
                (speaker_id,),
            )
            if result.rowcount == 0:
                raise SpeakerNotFoundError(speaker_id)
            connection.execute(
                "UPDATE speaker_notifications SET resolved_at = CURRENT_TIMESTAMP "
                "WHERE speaker_id = ? AND notification_type = 'speaker_review' "
                "AND resolved_at IS NULL",
                (speaker_id,),
            )
        return self.get(speaker_id)

    def rename(self, speaker_id: str, display_name: str) -> SpeakerProfile:
        name = display_name.strip()
        if not name:
            raise ValueError("display_name is required")
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE speaker_profiles SET display_name = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE speaker_id = ?",
                (name, speaker_id),
            )
            if result.rowcount == 0:
                raise SpeakerNotFoundError(speaker_id)
        return self.get(speaker_id)

    def add_sample(
        self,
        speaker_id: str,
        embedding: object,
        *,
        duration_s: float,
        speech_duration_s: float,
        quality_score: float,
        audio: bytes | None = None,
        filename: str = "sample.wav",
    ) -> str:
        import numpy as np

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size != 256 or norm == 0:
            raise SpeakerEnrollmentError("invalid resemblyzer embedding")
        vector = vector / norm
        sample_id = str(uuid.uuid4())
        audio_path: str | None = None
        if audio:
            audio_path = self._save_audio(speaker_id, sample_id, audio, filename)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM speaker_profiles WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
            if row is None:
                raise SpeakerNotFoundError(speaker_id)
            if row["status"] not in {"active", "pending_review"}:
                raise SpeakerEnrollmentError("speaker profile is disabled")
            connection.execute(
                "INSERT INTO speaker_samples "
                "(sample_id, speaker_id, embedding, duration_s, speech_duration_s, quality_score, original_filename, audio_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    speaker_id,
                    vector.tobytes(),
                    duration_s,
                    speech_duration_s,
                    quality_score,
                    Path(filename).name,
                    audio_path,
                ),
            )
            connection.execute(
                "UPDATE speaker_profiles SET updated_at = CURRENT_TIMESTAMP "
                "WHERE speaker_id = ?",
                (speaker_id,),
            )
        return sample_id

    def _save_audio(
        self, speaker_id: str, sample_id: str, audio: bytes | None, filename: str
    ) -> str | None:
        if not audio:
            return None
        suffix = Path(filename).suffix.lower()
        if suffix not in {".m4a", ".wav", ".mp3", ".flac", ".aac", ".mov", ".mp4"}:
            suffix = ".wav"
        destination = self.sample_dir / speaker_id / f"{sample_id}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(audio)
        return str(destination)

    def list_samples(self, speaker_id: str) -> list[SpeakerSample]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sample_id, speaker_id, duration_s, speech_duration_s, quality_score, "
                "original_filename, audio_path, created_at FROM speaker_samples "
                "WHERE speaker_id = ? ORDER BY created_at DESC",
                (speaker_id,),
            ).fetchall()
        return [
            SpeakerSample(
                row["sample_id"], row["speaker_id"], float(row["duration_s"]),
                float(row["speech_duration_s"]), row["quality_score"],
                row["original_filename"], row["audio_path"], row["created_at"],
            )
            for row in rows
        ]

    def sample_audio_path(self, speaker_id: str, sample_id: str) -> Path | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT audio_path FROM speaker_samples WHERE speaker_id = ? AND sample_id = ?",
                (speaker_id, sample_id),
            ).fetchone()
        if row is None:
            raise SpeakerNotFoundError(sample_id)
        path = Path(row["audio_path"]) if row["audio_path"] else None
        return path if path and path.is_file() else None

    def find_similar(self, embedding: object, *, threshold: float | None = None) -> SpeakerMatch | None:
        import numpy as np

        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query /= np.linalg.norm(query) + 1e-9
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.speaker_id, p.display_name, s.embedding FROM speaker_profiles p "
                "JOIN speaker_samples s ON s.speaker_id = p.speaker_id "
                "WHERE p.status IN ('active', 'pending_review')"
            ).fetchall()
        if not rows:
            return None
        candidates = [
            (float(np.dot(query, np.frombuffer(row["embedding"], dtype=np.float32))), row["speaker_id"], row["display_name"])
            for row in rows
        ]
        score, speaker_id, name = max(candidates)
        if score < (self.duplicate_threshold if threshold is None else threshold):
            return None
        return SpeakerMatch(speaker_id, name, score, "high" if score >= 0.92 else "medium")

    def list_notifications(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT notification_id, speaker_id, notification_type, message, created_at "
                "FROM speaker_notifications WHERE resolved_at IS NULL ORDER BY created_at DESC"
            ).fetchall()

    def resolve_notifications(self, speaker_id: str, notification_type: str | None = None) -> None:
        with self._connect() as connection:
            if notification_type:
                connection.execute(
                    "UPDATE speaker_notifications SET resolved_at = CURRENT_TIMESTAMP "
                    "WHERE speaker_id = ? AND notification_type = ? AND resolved_at IS NULL",
                    (speaker_id, notification_type),
                )
            else:
                connection.execute(
                    "UPDATE speaker_notifications SET resolved_at = CURRENT_TIMESTAMP "
                    "WHERE speaker_id = ? AND resolved_at IS NULL", (speaker_id,)
                )

    def match(self, embedding: object) -> SpeakerMatch | None:
        import numpy as np

        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query /= np.linalg.norm(query) + 1e-9
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.speaker_id, p.display_name, s.embedding "
                "FROM speaker_profiles p JOIN speaker_samples s "
                "ON s.speaker_id = p.speaker_id WHERE p.status = 'active'"
            ).fetchall()
        if not rows:
            return None
        grouped: dict[str, tuple[str, list[object]]] = {}
        for row in rows:
            name, vectors = grouped.setdefault(row["speaker_id"], (row["display_name"], []))
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        candidates: list[tuple[float, str, str]] = []
        for speaker_id, (name, vectors) in grouped.items():
            centroid = np.mean(np.stack(vectors), axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            candidates.append((float(np.dot(query, centroid)), speaker_id, name))
        candidates.sort(reverse=True)
        best_score, speaker_id, name = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else -1.0
        if best_score < self.match_threshold or best_score - second_score < self.match_margin:
            return None
        confidence = "high" if best_score >= self.match_threshold + 0.1 else "medium"
        return SpeakerMatch(speaker_id, name, best_score, confidence)


@dataclass(frozen=True)
class EnrollmentResult:
    sample_id: str
    duration_s: float
    speech_duration_s: float
    quality_score: float


class ResemblyzerEnrollmentService:
    """将短音频解码、VAD 去静音并提取单人声纹。"""

    def __init__(self, vad_model: str) -> None:
        self.vad_model = vad_model

    def enroll(
        self,
        audio: bytes,
        filename: str,
        store: SpeakerProfileStore,
        speaker_id: str,
    ) -> EnrollmentResult:
        import numpy as np
        import soundfile as sf
        from resemblyzer import VoiceEncoder
        from scipy.io import wavfile
        from .workflow import SileroVADStage

        suffix = Path(filename).suffix.lower() or ".audio"
        with tempfile.TemporaryDirectory(prefix="astra_speaker_") as temp_dir:
            source = Path(temp_dir) / f"source{suffix}"
            wav = Path(temp_dir) / "audio.wav"
            source.write_bytes(audio)
            try:
                subprocess.run(
                    [
                        "/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                        "-c", "1", str(source), str(wav),
                    ],
                    check=True, capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise SpeakerEnrollmentError("audio cannot be decoded by afconvert") from exc
            sr, data = wavfile.read(wav)
            if sr != 16000:
                raise SpeakerEnrollmentError("enrollment audio must decode to 16kHz")
            if np.issubdtype(data.dtype, np.integer):
                waveform = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1)
            else:
                waveform = data.astype(np.float32)
            duration_s = len(waveform) / sr
            vad = SileroVADStage(self.vad_model)
            chunks = vad._detect_blocking(wav)
            speech_duration_s = sum(chunk.end - chunk.start for chunk in chunks)
            if speech_duration_s < 2.0:
                raise SpeakerEnrollmentError("need at least 2 seconds of speech")
            speech_parts = []
            for chunk in chunks:
                part, part_sr = sf.read(BytesIO(chunk.audio), dtype="float32")
                if part_sr == sr:
                    speech_parts.append(np.asarray(part).reshape(-1))
            if not speech_parts:
                raise SpeakerEnrollmentError("no valid speech segment found")
            encoder = VoiceEncoder(device="cpu")
            embedding = encoder.embed_utterance(np.concatenate(speech_parts))
            quality_score = min(1.0, speech_duration_s / max(duration_s, 1e-6))
            sample_id = store.add_sample(
                speaker_id,
                embedding,
                duration_s=duration_s,
                speech_duration_s=speech_duration_s,
                quality_score=quality_score,
                audio=audio,
                filename=filename,
            )
            return EnrollmentResult(
                sample_id, duration_s, speech_duration_s, quality_score
            )
