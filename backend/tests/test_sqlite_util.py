import sqlite3

from app.core.sqlite_util import enable_wal


def test_enable_wal_switches_on_disk_database(tmp_path) -> None:
    path = tmp_path / "db.sqlite3"
    connection = sqlite3.connect(path)
    try:
        enable_wal(connection)
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_enable_wal_ignores_readonly_database(tmp_path) -> None:
    path = tmp_path / "ro.sqlite3"
    sqlite3.connect(path).close()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        # Must not raise even though a read-only database cannot write WAL.
        enable_wal(connection)
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    finally:
        connection.close()


def test_enable_wal_is_noop_for_in_memory_database() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        enable_wal(connection)
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "memory"
    finally:
        connection.close()
