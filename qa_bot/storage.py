"""QA Bot — SQLite-based persistent conversation history.

Each user gets their own history, stored by telegram user_id.
History is trimmed to last MAX_TURNS turns automatically.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("QA_BOT_DB_PATH", "/opt/ouroboros/qa_bot_history.db")
MAX_TURNS = 20  # max turns per user (each turn = user + assistant message)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if not exist."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON history(user_id)")
    logger.info("QA Bot DB initialized: %s", DB_PATH)


def get_history(user_id: int) -> list[dict]:
    """Return last MAX_TURNS*2 messages for a user (alternating user/assistant)."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, MAX_TURNS * 2),
        ).fetchall()
    # rows are in reverse order — restore chronological
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def add_message(user_id: int, role: str, content: str) -> None:
    """Append a message to user's history."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        # Trim old entries beyond MAX_TURNS*2
        conn.execute(
            """
            DELETE FROM history WHERE user_id = ? AND id NOT IN (
                SELECT id FROM history WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
            )
            """,
            (user_id, user_id, MAX_TURNS * 2),
        )


def clear_history(user_id: int) -> None:
    """Clear all history for a user."""
    with _db() as conn:
        conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    logger.info("Cleared history for user %d", user_id)
