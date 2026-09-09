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


def test_duplicate_candidate_reuses_existing_profile_and_records_audio(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "speakers.sqlite3", sample_dir=tmp_path / "samples")
    embedding = np.zeros(256, dtype=np.float32)
    embedding[0] = 1.0

    first = store.register_or_append_candidate(embedding, duration_s=4.0)
    second = store.register_or_append_candidate(embedding, duration_s=5.0)

    assert second.speaker_id == first.speaker_id
    assert store.get(first.speaker_id).sample_count == 2
    sample_id = store.add_sample(
        first.speaker_id, embedding, duration_s=3.0, speech_duration_s=2.5,
        quality_score=0.8, audio=b"RIFFsample", filename="sample.wav",
    )
    samples = store.list_samples(first.speaker_id)
    stored = next(sample for sample in samples if sample.sample_id == sample_id)
    assert stored.audio_path is not None
    assert store.sample_audio_path(first.speaker_id, sample_id).read_bytes() == b"RIFFsample"
