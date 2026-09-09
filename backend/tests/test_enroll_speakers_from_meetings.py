import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/meeting/enroll_speakers_from_meetings.py"
SPEC = importlib.util.spec_from_file_location("enroll_speakers_from_meetings", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_parse_transcript_and_selects_longest_segments(tmp_path) -> None:
    meeting = tmp_path / "meeting"
    meeting.mkdir()
    (meeting / "input.m4a").write_bytes(b"audio")
    (meeting / "transcript.txt").write_text(
        "\n".join(
            [
                "[00:00] S1 开始",
                "[00:03] S2 回应",
                "[00:08] S1 较长片段",
                "[00:30] S3 太短",
                "[00:31] S1 结尾",
            ]
        ),
        encoding="utf-8",
    )

    segments = module.parse_transcript(meeting / "transcript.txt")
    assert [(item.label, item.start, item.end) for item in segments[:3]] == [
        ("S1", 0, 3),
        ("S2", 3, 8),
        ("S1", 8, 30),
    ]

    plans = module.select_segments(
        [meeting],
        {"S1", "S2", "S3"},
        max_samples_per_speaker=1,
        min_duration=2.0,
        max_duration=10.0,
    )

    assert [(plan.segment.label, plan.segment.start, plan.segment.end) for plan in plans] == [
        ("S1", 8, 18),
        ("S2", 3, 8),
    ]


def test_parse_label_range() -> None:
    assert module.parse_labels("S1-S3,S8") == {"S1", "S2", "S3", "S8"}


def test_select_aggregate_segments_combines_short_turns(tmp_path) -> None:
    meeting = tmp_path / "meeting"
    meeting.mkdir()
    (meeting / "input.m4a").write_bytes(b"audio")
    (meeting / "transcript.txt").write_text(
        "\n".join(
            [
                "[00:00] S2 短句一",
                "[00:01] S1 插话",
                "[00:03] S2 短句二",
                "[00:05] S1 插话",
                "[00:08] S2 短句三",
                "[00:10] S1 结束",
            ]
        ),
        encoding="utf-8",
    )

    plans = module.select_aggregate_segments(
        [meeting],
        {"S2"},
        min_duration=2.0,
        max_duration=4.0,
    )

    assert len(plans) == 1
    assert plans[0].label == "S2"
    assert len(plans[0].segments) == 2
    assert plans[0].duration == 4.0
