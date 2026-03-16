"""QA Bot — SQLite storage for persistent conversation history."""
import sqlite3
import json
import os
import time
from pathlib import Path

DB_PATH = os.environ.get("QA_BOT_DB_PATH", "/opt/ouroboros/qa_bot_data/history.db")


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            user_id INTEGER PRIMARY KEY,
            messages TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def load_history(user_id: int) -> list[dict]:
    """Load conversation history for a user (last 12 messages)."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT messages FROM history WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                return json.loads(row[0])[-12:]
    except Exception:
        pass
    return []


def save_history(user_id: int, messages: list[dict]) -> None:
    """Save conversation history for a user (keep last 20)."""
    try:
        data = json.dumps(messages[-20:])
        with _get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO history (user_id, messages, updated_at)
                VALUES (?, ?, ?)
            """, (user_id, data, time.time()))
    except Exception:
        pass


def clear_history(user_id: int) -> None:
    """Clear conversation history for a user."""
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    except Exception:
        pass
