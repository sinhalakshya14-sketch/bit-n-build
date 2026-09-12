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
                CREATE TABLE IF NOT EXISTS zone_flags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    zone_id TEXT NOT NULL,
                    zone_name TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    flag_count INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_zone_flags_tick
                    ON zone_flags (zone_id, tick DESC);
                CREATE TABLE IF NOT EXISTS route_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    threshold_used REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
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


def log_zone_flag(zone_id: str, zone_name: str, tick: int, flag_count: int = 1) -> None:
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO zone_flags (zone_id, zone_name, tick, flag_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (zone_id, zone_name, int(tick), int(flag_count), _now()),
            )
            conn.commit()
        finally:
            conn.close()


def get_zone_rolling_stats(window_ticks: int = 30, current_tick: int = 0) -> dict[str, dict[str, Any]]:
    """Returns {zone_id: {'count': int, 'risk_label': 'elevated' | 'moderate' | 'low'}} for flags in [current_tick - window_ticks, current_tick]."""
    init_db()
    min_tick = max(0, current_tick - window_ticks)
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT zone_id, zone_name, COUNT(*) as flag_events, SUM(flag_count) as total_flags "
                "FROM zone_flags WHERE tick >= ? GROUP BY zone_id",
                (min_tick,),
            ).fetchall()
            stats = {}
            for r in rows:
                c = r["total_flags"] or r["flag_events"] or 0
                stats[r["zone_id"]] = {
                    "zone_name": r["zone_name"],
                    "count": c,
                    "risk_label": "elevated" if c >= 3 else ("moderate" if c >= 1 else "low"),
                }
            return stats
        finally:
            conn.close()


def log_route_decision(rec_id: str, action: str, reason: str, threshold: float) -> None:
    """Log an approval or override by dispatcher."""
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO route_overrides (recommendation_id, action, reason, threshold_used, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(rec_id), str(action), str(reason), float(threshold), _now()),
            )
            conn.commit()
        finally:
            conn.close()


def get_recent_override_count(limit: int = 10) -> int:
    """Count how many recent decisions were 'overridden'."""
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT action FROM route_overrides ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return sum(1 for r in rows if r["action"].lower() == "overridden")
        finally:
            conn.close()


# --------------------------------------------------------------------------- #
#  Admin / Developer Read-Only Functions                                       #
# --------------------------------------------------------------------------- #

def get_all_watchlist_entries() -> list[dict[str, Any]]:
    """Return every row in the watchlist table across all codenames."""
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT analyst_name, vessel_id, added_at FROM watchlist ORDER BY added_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_all_notes() -> list[dict[str, Any]]:
    """Return every row in the notes table across all codenames."""
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT id, analyst_name, vessel_id, note_text, created_at FROM notes ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_summary_stats() -> dict[str, int]:
    """Aggregate counts for admin overview: total watchlist, total notes, distinct codenames."""
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            wl_count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
            notes_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            distinct = conn.execute(
                "SELECT COUNT(DISTINCT analyst_name) FROM ("
                "  SELECT analyst_name FROM watchlist"
                "  UNION"
                "  SELECT analyst_name FROM notes"
                ")"
            ).fetchone()[0]
            return {
                "total_watchlist": wl_count,
                "total_notes": notes_count,
                "distinct_codenames": distinct,
            }
        finally:
            conn.close()


init_db()
