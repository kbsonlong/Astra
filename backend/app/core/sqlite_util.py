"""Shared SQLite connection helpers for the on-disk stores.

The realtime and workbench processes open the same databases concurrently, so
each store enables WAL journaling plus a busy timeout. WAL is a persistent
database property that only needs to be set once and requires a writable file;
an in-memory or read-only database cannot switch to it, so the attempt is
best-effort and silently leaves the existing journal mode in place.
"""
from __future__ import annotations

import sqlite3


def enable_wal(connection: sqlite3.Connection) -> None:
    """Switch the database to WAL when possible; ignore read-only databases."""
    try:
        connection.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        # A read-only or in-memory database keeps its current journal mode.
        pass
