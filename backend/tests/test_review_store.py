import pytest

from app.core.review_store import ReviewConflictError, ReviewStore


def test_review_store_imports_updates_and_detects_stale_versions(tmp_path) -> None:
    store = ReviewStore(tmp_path / "tasks.sqlite3")
    store.import_if_empty("task-1", [{
        "segment_id": "seg-000001", "text": "原文", "corrected_text": "原文", "review_status": "pending",
    }])
    rows, counts = store.page("task-1", page=1, page_size=20)
    assert counts == {"total": 1, "pending": 1, "approved": 0, "rejected": 0}
    assert rows[0]["version"] == 1

    store.update_many("task-1", [{
        "segment_id": "seg-000001", "corrected_text": "修正", "version": 1,
    }], "approved")
    exported = store.export_rows("task-1")
    assert exported[0]["text"] == "修正"
    assert exported[0]["review_status"] == "approved"

    with pytest.raises(ReviewConflictError, match="seg-000001"):
        store.update_many("task-1", [{
            "segment_id": "seg-000001", "corrected_text": "过期更新", "version": 1,
        }], "approved")
