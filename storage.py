"""
storage.py
==========
SQLite-backed per-analyst watchlist and case notes.
No authentication — analyst_name is a lookup key only.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parent / "watchlist.db"
_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    # New connection per call so Streamlit fragment/script threads never share
    # a cursor. check_same_thread=False + timeout avoids "database is locked"
    # on Windows when a fragment write races a read.
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    analyst_name TEXT NOT NULL,
                    vessel_id TEXT NOT NULL,
                    added_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (analyst_name, vessel_id)
                );
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analyst_name TEXT NOT NULL,
                    vessel_id TEXT NOT NULL,
                    note_text TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_notes_lookup
                    ON notes (analyst_name, vessel_id, created_at DESC);
                """
            )
            conn.commit()
        finally:
            conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def add_to_watchlist(analyst_name: str, vessel_id: str) -> None:
    init_db()
    name, vid = str(analyst_name).strip(), str(vessel_id).strip()
    if not name or not vid:
        return
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (analyst_name, vessel_id, added_at) VALUES (?, ?, ?)",
                (name, vid, _now()),
            )
            conn.commit()
        finally:
            conn.close()


def remove_from_watchlist(analyst_name: str, vessel_id: str) -> None:
    init_db()
    name, vid = str(analyst_name).strip(), str(vessel_id).strip()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "DELETE FROM watchlist WHERE analyst_name = ? AND vessel_id = ?",
                (name, vid),
            )
            conn.commit()
        finally:
            conn.close()


def get_watchlist(analyst_name: str) -> list[dict[str, Any]]:
    init_db()
    name = str(analyst_name).strip()
    if not name:
        return []
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT analyst_name, vessel_id, added_at FROM watchlist "
                "WHERE analyst_name = ? ORDER BY added_at DESC",
                (name,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def add_note(analyst_name: str, vessel_id: str, note_text: str) -> None:
    init_db()
    name, vid, text = str(analyst_name).strip(), str(vessel_id).strip(), str(note_text).strip()
    if not name or not vid or not text:
        return
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO notes (analyst_name, vessel_id, note_text, created_at) VALUES (?, ?, ?, ?)",
                (name, vid, text, _now()),
            )
            conn.commit()
        finally:
            conn.close()


def get_notes(analyst_name: str, vessel_id: str) -> list[dict[str, Any]]:
    init_db()
    name, vid = str(analyst_name).strip(), str(vessel_id).strip()
    if not name or not vid:
        return []
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT id, analyst_name, vessel_id, note_text, created_at FROM notes "
                "WHERE analyst_name = ? AND vessel_id = ? ORDER BY created_at DESC, id DESC",
                (name, vid),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


init_db()
