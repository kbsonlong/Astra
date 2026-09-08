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


class SpeakerProfileStore:
    """SQLite 存储；只保存 embedding，不保存注册原始音频。"""

    def __init__(
        self,
        path: str | Path,
        *,
        match_threshold: float = 0.75,
        match_margin: float = 0.05,
    ) -> None:
        self.path = Path(path).expanduser()
        self.match_threshold = match_threshold
        self.match_margin = match_margin

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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_speaker_samples_speaker
                ON speaker_samples(speaker_id);
            """
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

    def list(self) -> list[SpeakerProfile]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.*, COUNT(s.sample_id) AS sample_count "
                "FROM speaker_profiles p LEFT JOIN speaker_samples s "
                "ON s.speaker_id = p.speaker_id "
                "WHERE p.status = 'active' "
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
    ) -> str:
        import numpy as np

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size != 256 or norm == 0:
            raise SpeakerEnrollmentError("invalid resemblyzer embedding")
        vector = vector / norm
        sample_id = str(uuid.uuid4())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM speaker_profiles WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
            if row is None:
                raise SpeakerNotFoundError(speaker_id)
            if row["status"] != "active":
                raise SpeakerEnrollmentError("speaker profile is disabled")
            connection.execute(
                "INSERT INTO speaker_samples "
                "(sample_id, speaker_id, embedding, duration_s, speech_duration_s, quality_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    speaker_id,
                    vector.tobytes(),
                    duration_s,
                    speech_duration_s,
                    quality_score,
                ),
            )
            connection.execute(
                "UPDATE speaker_profiles SET updated_at = CURRENT_TIMESTAMP "
                "WHERE speaker_id = ?",
                (speaker_id,),
            )
        return sample_id

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
            )
            return EnrollmentResult(
                sample_id, duration_s, speech_duration_s, quality_score
            )
