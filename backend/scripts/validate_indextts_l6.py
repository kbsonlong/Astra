"""Run a real IndexTTS MLX -> Astra WebSocket acceptance check.

The ASR and LLM are deterministic in-process doubles so this command isolates
the real TTS model, TTS voice snapshot resolution, and the WebSocket data path.
It intentionally does not upload or persist the supplied reference audio in
the repository.

Example:

    PYTHONPATH=backend .venv/bin/python backend/scripts/validate_indextts_l6.py \
      --model-dir ~/.astra/models/indextts-2.5-mlx \
      --reference /absolute/path/to/authorized-reference.wav \
      --output-dir /tmp/astra-indextts-l6
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import wave
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.core.pipeline import VoicePipeline
from app.core.tts_voice_store import TtsVoiceStore
from app.main import _build_tts_client, create_app


class _FixedAsr:
    async def transcribe(self, audio: bytes) -> str:
        if not audio:
            raise RuntimeError("L6 audio fixture must not be empty")
        return "本地 WebSocket TTS 验收输入"


class _FixedLlm:
    def __init__(self, text: str) -> None:
        self.text = text

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> AsyncIterator[str]:
        del messages, chat_template_kwargs
        yield self.text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.getenv("ASTRA_L6_INDEXTTS_MODEL_DIR", ""),
        help="prepared IndexTTS 2.5 MLX model directory",
    )
    parser.add_argument(
        "--reference",
        default=os.getenv("ASTRA_L6_REFERENCE_WAV", ""),
        help="authorized 5-15 second mono PCM16 reference WAV",
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("ASTRA_L6_INDEXTTS_REPO_ID", "yunfengwang/IndexTTS-2.5-mlx"),
    )
    parser.add_argument(
        "--model-revision",
        default=os.getenv("ASTRA_L6_INDEXTTS_MODEL_REVISION", ""),
    )
    parser.add_argument(
        "--text",
        default="你好，这是 Astra 通过 WebSocket 调用 IndexTTS 2.5 MLX 的本地验收。",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("ASTRA_L6_OUTPUT_DIR", ""),
        help="optional directory for round-1.wav and round-2.wav",
    )
    return parser


def _reference_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".wav" or not path.is_file() or path.is_symlink():
        raise ValueError("reference must be a regular .wav file")
    try:
        handle = wave.open(str(path), "rb")
    except (EOFError, wave.Error) as exc:
        raise ValueError("reference is not a valid WAV") from exc
    with handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        if channels != 1 or sample_width != 2 or sample_rate <= 0 or frame_count <= 0:
            raise ValueError("reference must be non-empty mono PCM16 WAV")
        duration = frame_count / sample_rate
    if not 5.0 <= duration <= 15.0:
        raise ValueError("reference duration must be between 5 and 15 seconds")
    return {
        "duration_s": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_voice(store: TtsVoiceStore, reference: Path, metadata: Mapping[str, Any]) -> str:
    token = "l6-reference"
    original = store.audio_dir / "uploads" / f"{token}-original.wav"
    canonical = store.audio_dir / "uploads" / f"{token}-reference.wav"
    original.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference, original)
    shutil.copyfile(reference, canonical)
    digest = _sha256(reference)
    voice = store.create_draft(
        "IndexTTS L6 acceptance voice",
        backend_family="indextts25_mlx",
        language="zh",
        reference_sha256=digest,
        original_sha256=digest,
        original_filename=reference.name,
        original_relpath="uploads/l6-reference-original.wav",
        reference_relpath="uploads/l6-reference-reference.wav",
        duration_s=float(metadata["duration_s"]),
        sample_rate=int(metadata["sample_rate"]),
        channels=int(metadata["channels"]),
        sample_width_bytes=int(metadata["sample_width_bytes"]),
    )
    store.confirm_consent(voice.voice_id, 1, note="explicit local L6 acceptance fixture")
    store.activate(voice.voice_id, 1)
    return voice.voice_id


def _read_round(websocket: Any, generation_id: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if (
            event.get("type") == "state_change"
            and event.get("state") == "LISTENING"
            and event.get("generation_id") == generation_id
        ):
            break
    types = [str(event.get("type")) for event in events]
    required = ["asr_final", "llm_token", "tts_start", "tts_chunk", "tts_end"]
    missing = [event_type for event_type in required if event_type not in types]
    if missing:
        raise RuntimeError(f"L6 generation {generation_id} missing events: {missing}; {events}")
    chunk = next(event for event in events if event.get("type") == "tts_chunk")
    import base64

    audio = base64.b64decode(str(chunk["audio_b64"]))
    with wave.open(io.BytesIO(audio), "rb") as handle:
        audio_metadata = {
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "sample_rate": handle.getframerate(),
            "frames": handle.getnframes(),
            "duration_s": handle.getnframes() / handle.getframerate(),
        }
    if audio_metadata["channels"] != 1 or audio_metadata["sample_width_bytes"] != 2:
        raise RuntimeError(f"L6 output is not mono PCM16 WAV: {audio_metadata}")
    if audio_metadata["sample_rate"] != 22_050:
        raise RuntimeError(f"unexpected IndexTTS sample rate: {audio_metadata}")
    if audio_metadata["frames"] <= 0:
        raise RuntimeError("L6 output WAV is empty")
    return {"events": types, "audio": audio_metadata, "audio_bytes": audio}


def main() -> int:
    args = _parser().parse_args()
    model_dir = Path(args.model_dir).expanduser()
    reference = Path(args.reference).expanduser()
    if not model_dir.is_dir():
        raise SystemExit("IndexTTS model directory does not exist; prepare L2 weights first")
    metadata = _reference_metadata(reference)

    with tempfile.TemporaryDirectory(prefix="astra-indextts-l6-") as work_dir:
        work = Path(work_dir)
        store = TtsVoiceStore(work / "voices.sqlite3", work / "voices")
        voice_id = _prepare_voice(store, reference, metadata)
        settings = Settings(
            tts_backend="indextts_mlx",
            tts_indextts_model_dir=str(model_dir),
            tts_indextts_repo_id=args.repo_id,
            tts_indextts_model_revision=args.model_revision,
            task_store_path=str(work / "tasks.sqlite3"),
            speaker_store_path=str(work / "speakers.sqlite3"),
            speaker_sample_dir=str(work / "speaker-samples"),
            tts_voice_store_path=str(work / "voices.sqlite3"),
            tts_voice_audio_dir=str(work / "voices"),
        )
        tts = _build_tts_client(settings)
        available, reason = tts.is_available()
        if not available:
            raise SystemExit(f"IndexTTS MLX is not available: {reason}")
        pipeline = VoicePipeline(_FixedAsr(), _FixedLlm(args.text), tts)
        app = create_app(
            settings,
            pipeline=pipeline,
            enable_pipeline=False,
            enable_meeting=False,
        )
        output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        rounds: list[dict[str, Any]] = []
        with TestClient(app) as client:
            started = time.perf_counter()
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json(
                    {
                        "type": "start_session",
                        "voice_id": voice_id,
                        "voice_revision": 1,
                    }
                )
                state = websocket.receive_json()
                if state.get("voice_id") != voice_id or state.get("voice_revision") != 1:
                    raise RuntimeError(f"WebSocket did not bind voice snapshot: {state}")
                for generation_id in (1, 2):
                    websocket.send_bytes(b"astra-l6-audio-fixture")
                    websocket.send_json({"type": "speech_end"})
                    reasoning = websocket.receive_json()
                    if reasoning.get("state") != "REASONING":
                        raise RuntimeError(f"unexpected reasoning state: {reasoning}")
                    round_started = time.perf_counter()
                    result = _read_round(websocket, generation_id)
                    elapsed = time.perf_counter() - round_started
                    audio = result.pop("audio_bytes")
                    if output_dir is not None:
                        (output_dir / f"round-{generation_id}.wav").write_bytes(audio)
                    result["generation_id"] = generation_id
                    result["elapsed_s"] = elapsed
                    rounds.append(result)
            total_elapsed = time.perf_counter() - started

        print(
            json.dumps(
                {
                    "status": "passed",
                    "backend": "indextts_mlx",
                    "voice_id": voice_id,
                    "voice_revision": 1,
                    "reference": metadata,
                    "rounds": rounds,
                    "total_elapsed_s": total_elapsed,
                    "output_dir": str(output_dir) if output_dir else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
