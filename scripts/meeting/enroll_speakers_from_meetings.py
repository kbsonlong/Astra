"""Create speaker profiles from existing diarized meeting outputs.

The script reads transcript lines like ``[03:12] S2 ...``, cuts representative
audio snippets from each meeting input, and enrolls them through the production
speaker registry service. It is intended for bootstrapping S1-S8 labels before
manual display-name confirmation.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.core.speaker_registry import (  # noqa: E402
    EnrollmentResult,
    ResemblyzerEnrollmentService,
    SpeakerEnrollmentError,
    SpeakerProfile,
    SpeakerProfileStore,
)

TRANSCRIPT_LINE_RE = re.compile(r"^\[(?P<time>\d{2}:\d{2}(?::\d{2})?)\]\s+(?P<label>S\d+)\s+")


@dataclass(frozen=True)
class TranscriptSegment:
    label: str
    start: float
    end: float
    line_no: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class EnrollmentPlan:
    meeting_dir: Path
    audio_path: Path
    segment: TranscriptSegment


@dataclass(frozen=True)
class AggregateEnrollmentPlan:
    label: str
    meeting_dir: Path
    audio_path: Path
    segments: tuple[TranscriptSegment, ...]

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)


def parse_timestamp(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"unsupported timestamp: {value}")


def parse_transcript(path: Path, *, default_tail_seconds: float = 8.0) -> list[TranscriptSegment]:
    starts: list[tuple[str, float, int]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TRANSCRIPT_LINE_RE.match(line)
        if match:
            starts.append((match.group("label"), parse_timestamp(match.group("time")), line_no))
    segments: list[TranscriptSegment] = []
    for index, (label, start, line_no) in enumerate(starts):
        end = (
            starts[index + 1][1]
            if index + 1 < len(starts)
            else start + default_tail_seconds
        )
        segments.append(TranscriptSegment(label, start, max(start, end), line_no))
    return segments


def select_segments(
    meeting_dirs: list[Path],
    labels: set[str],
    *,
    max_samples_per_speaker: int,
    min_duration: float,
    max_duration: float,
) -> list[EnrollmentPlan]:
    selected: dict[str, list[EnrollmentPlan]] = {label: [] for label in labels}
    candidates: list[EnrollmentPlan] = []
    for meeting_dir in meeting_dirs:
        transcript = meeting_dir / "transcript.txt"
        if not transcript.is_file():
            continue
        audio_path = find_audio(meeting_dir)
        if audio_path is None:
            continue
        for segment in parse_transcript(transcript):
            if segment.label in labels and segment.duration >= min_duration:
                clipped = TranscriptSegment(
                    segment.label,
                    segment.start,
                    min(segment.end, segment.start + max_duration),
                    segment.line_no,
                )
                candidates.append(EnrollmentPlan(meeting_dir, audio_path, clipped))

    candidates.sort(
        key=lambda item: (
            item.segment.label,
            -item.segment.duration,
            str(item.meeting_dir),
            item.segment.line_no,
        )
    )
    for candidate in candidates:
        bucket = selected[candidate.segment.label]
        if len(bucket) < max_samples_per_speaker:
            bucket.append(candidate)
    return [item for label in sorted(selected, key=label_number) for item in selected[label]]


def find_audio(meeting_dir: Path) -> Path | None:
    for path in sorted(meeting_dir.glob("input.*")):
        if path.suffix.lower() in {".m4a", ".wav", ".mp3", ".flac", ".aac", ".mov", ".mp4"}:
            return path
    return None


def select_aggregate_segments(
    meeting_dirs: list[Path],
    labels: set[str],
    *,
    min_duration: float,
    max_duration: float,
) -> list[AggregateEnrollmentPlan]:
    plans: list[AggregateEnrollmentPlan] = []
    for label in sorted(labels, key=label_number):
        best: AggregateEnrollmentPlan | None = None
        for meeting_dir in meeting_dirs:
            transcript = meeting_dir / "transcript.txt"
            if not transcript.is_file():
                continue
            audio_path = find_audio(meeting_dir)
            if audio_path is None:
                continue
            segments = [
                segment
                for segment in parse_transcript(transcript)
                if segment.label == label and segment.duration > 0
            ]
            if not segments:
                continue
            segments.sort(key=lambda item: item.duration, reverse=True)
            chosen: list[TranscriptSegment] = []
            total = 0.0
            for segment in segments:
                remaining = max_duration - total
                if remaining <= 0:
                    break
                duration = min(segment.duration, remaining)
                chosen.append(
                    TranscriptSegment(
                        segment.label,
                        segment.start,
                        segment.start + duration,
                        segment.line_no,
                    )
                )
                total += duration
            if total < min_duration:
                continue
            candidate = AggregateEnrollmentPlan(label, meeting_dir, audio_path, tuple(chosen))
            if best is None or candidate.duration > best.duration:
                best = candidate
        if best is not None:
            plans.append(best)
    return plans


def label_number(label: str) -> int:
    return int(label[1:]) if label.startswith("S") and label[1:].isdigit() else 10_000


def get_or_create_profile(store: SpeakerProfileStore, display_name: str) -> SpeakerProfile:
    for profile in store.list():
        if profile.display_name == display_name:
            return profile
    return store.create(display_name)


def cut_audio(plan: EnrollmentPlan, out_path: Path) -> None:
    duration = plan.segment.duration
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{plan.segment.start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(plan.audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ],
        check=True,
    )


def cut_aggregate_audio(plan: AggregateEnrollmentPlan, out_path: Path, temp_dir: Path) -> None:
    parts: list[Path] = []
    for index, segment in enumerate(plan.segments):
        part = temp_dir / f"{plan.label}-{index}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{segment.start:.3f}",
                "-t",
                f"{segment.duration:.3f}",
                "-i",
                str(plan.audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(part),
            ],
            check=True,
        )
        parts.append(part)
    concat_list = temp_dir / f"{plan.label}-concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{part}'" for part in parts),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(out_path),
        ],
        check=True,
    )


def enroll_without_vad(
    sample: Path,
    *,
    store: SpeakerProfileStore,
    speaker_id: str,
) -> EnrollmentResult:
    import numpy as np
    from resemblyzer import VoiceEncoder
    from scipy.io import wavfile

    sr, data = wavfile.read(sample)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        waveform = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1)
    else:
        waveform = data.astype(np.float32)
    duration_s = len(waveform) / sr
    if duration_s < 0.8:
        raise SpeakerEnrollmentError("trusted transcript segment is too short")
    encoder = VoiceEncoder(device="cpu")
    embedding = encoder.embed_utterance(waveform)
    sample_id = store.add_sample(
        speaker_id,
        embedding,
        duration_s=duration_s,
        speech_duration_s=duration_s,
        quality_score=min(1.0, duration_s / 2.0),
        audio=sample.read_bytes(),
        filename=sample.name,
    )
    return EnrollmentResult(sample_id, duration_s, duration_s, min(1.0, duration_s / 2.0))


def enroll_plan(
    plan: EnrollmentPlan,
    *,
    store: SpeakerProfileStore,
    service: ResemblyzerEnrollmentService,
    dry_run: bool,
    allow_transcript_fallback: bool,
) -> EnrollmentResult | None:
    if dry_run:
        return None
    profile = get_or_create_profile(store, plan.segment.label)
    with tempfile.TemporaryDirectory(prefix="astra_enroll_") as temp_dir:
        sample = Path(temp_dir) / f"{plan.segment.label}-{plan.segment.line_no}.wav"
        cut_audio(plan, sample)
        try:
            return service.enroll(sample.read_bytes(), sample.name, store, profile.speaker_id)
        except SpeakerEnrollmentError:
            if not allow_transcript_fallback:
                raise
            return enroll_without_vad(sample, store=store, speaker_id=profile.speaker_id)


def enroll_aggregate_plan(
    plan: AggregateEnrollmentPlan,
    *,
    store: SpeakerProfileStore,
    service: ResemblyzerEnrollmentService,
    dry_run: bool,
    allow_transcript_fallback: bool,
) -> EnrollmentResult | None:
    if dry_run:
        return None
    profile = get_or_create_profile(store, plan.label)
    with tempfile.TemporaryDirectory(prefix="astra_enroll_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        sample = temp_dir / f"{plan.label}-aggregate.wav"
        cut_aggregate_audio(plan, sample, temp_dir)
        try:
            return service.enroll(sample.read_bytes(), sample.name, store, profile.speaker_id)
        except SpeakerEnrollmentError:
            if not allow_transcript_fallback:
                raise
            return enroll_without_vad(sample, store=store, speaker_id=profile.speaker_id)


def ensure_profiles(store: SpeakerProfileStore, labels: set[str]) -> list[SpeakerProfile]:
    return [get_or_create_profile(store, label) for label in sorted(labels, key=label_number)]


def parse_labels(value: str) -> set[str]:
    labels: set[str] = set()
    for part in value.split(","):
        part = part.strip().upper()
        if "-" in part:
            start, end = part.split("-", 1)
            if start.startswith("S") and end.startswith("S"):
                labels.update(f"S{index}" for index in range(label_number(start), label_number(end) + 1))
                continue
        if part:
            labels.add(part)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll S-label speaker profiles from meeting transcripts")
    parser.add_argument(
        "--meeting-dir",
        action="append",
        type=Path,
        required=True,
        help="Existing ~/Astra/meetings/<task_id> directory. Repeat for multiple meetings.",
    )
    parser.add_argument("--labels", default="S1-S8", help="Comma list or range, for example S1-S8,S10")
    parser.add_argument("--max-samples-per-speaker", type=int, default=2)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=18.0)
    parser.add_argument("--aggregate-max-duration", type=float, default=30.0)
    parser.add_argument(
        "--no-transcript-fallback",
        action="store_true",
        help="Do not bypass VAD for diarized transcript snippets that are too short.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    store = SpeakerProfileStore(
        settings.speaker_store_path,
        match_threshold=settings.speaker_match_threshold,
        match_margin=settings.speaker_match_margin,
    )
    service = ResemblyzerEnrollmentService(settings.vad_model)
    meeting_dirs = [path.expanduser().resolve() for path in args.meeting_dir]
    labels = parse_labels(args.labels)
    if not args.dry_run:
        ensure_profiles(store, labels)
    plans = select_segments(
        meeting_dirs,
        labels,
        max_samples_per_speaker=args.max_samples_per_speaker,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    if not plans:
        print("No eligible speaker segments found.")
        return 1

    failed_labels: set[str] = set()
    successful: set[str] = set()
    existing = {
        profile.display_name
        for profile in store.list()
        if profile.display_name in labels and profile.sample_count > 0
    }
    for plan in plans:
        if plan.segment.label in existing:
            continue
        print(
            f"{plan.segment.label}: {plan.meeting_dir.name} "
            f"{plan.segment.start:.1f}-{plan.segment.end:.1f}s line={plan.segment.line_no}",
            flush=True,
        )
        try:
            result = enroll_plan(
                plan,
                store=store,
                service=service,
                dry_run=args.dry_run,
                allow_transcript_fallback=not args.no_transcript_fallback,
            )
        except Exception as exc:
            failed_labels.add(plan.segment.label)
            print(f"  failed: {exc}", file=sys.stderr, flush=True)
            continue
        if result is not None:
            successful.add(plan.segment.label)
            failed_labels.discard(plan.segment.label)
            print(
                f"  enrolled sample={result.sample_id} "
                f"speech={result.speech_duration_s:.1f}s quality={result.quality_score:.2f}",
                flush=True,
            )

    aggregate_labels = labels - successful - existing
    aggregate_plans = select_aggregate_segments(
        meeting_dirs,
        aggregate_labels,
        min_duration=args.min_duration,
        max_duration=args.aggregate_max_duration,
    )
    for plan in aggregate_plans:
        if plan.label in successful or plan.label in existing:
            continue
        print(
            f"{plan.label}: aggregate {plan.meeting_dir.name} "
            f"segments={len(plan.segments)} duration={plan.duration:.1f}s",
            flush=True,
        )
        try:
            result = enroll_aggregate_plan(
                plan,
                store=store,
                service=service,
                dry_run=args.dry_run,
                allow_transcript_fallback=not args.no_transcript_fallback,
            )
        except Exception as exc:
            failed_labels.add(plan.label)
            print(f"  failed: {exc}", file=sys.stderr, flush=True)
            continue
        if result is not None:
            successful.add(plan.label)
            failed_labels.discard(plan.label)
            print(
                f"  enrolled sample={result.sample_id} "
                f"speech={result.speech_duration_s:.1f}s quality={result.quality_score:.2f}",
                flush=True,
            )

    planned_labels = {plan.segment.label for plan in plans} | {plan.label for plan in aggregate_plans}
    missing = sorted(labels - planned_labels - existing, key=label_number)
    if missing:
        print(f"No eligible samples for: {', '.join(missing)}", file=sys.stderr)
    unresolved_failures = failed_labels - successful - existing
    return 1 if unresolved_failures else 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
