import numpy as np

from app.core.speaker_registry import SpeakerProfileStore


def test_registered_voiceprint_match_survives_display_name_change(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "speakers.sqlite3", match_threshold=0.75)
    profile = store.create("忠思")
    embedding = np.zeros(256, dtype=np.float32)
    embedding[0] = 1.0
    store.add_sample(
        profile.speaker_id,
        embedding,
        duration_s=5.0,
        speech_duration_s=4.5,
        quality_score=0.9,
    )

    store.rename(profile.speaker_id, "主持人")
    match = store.match(embedding)

    assert match is not None
    assert match.speaker_id == profile.speaker_id
    assert match.display_name == "主持人"
    assert match.similarity > 0.99
    assert store.get(profile.speaker_id).sample_count == 1
