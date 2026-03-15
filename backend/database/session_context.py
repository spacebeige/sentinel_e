"""
SQLite session-level context cache for vision summaries and asset references.
Lightweight per-session store — avoids re-calling vision models for previously
analysed images/PDFs.
"""

import sqlite3
import os
import json
import logging
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

_SESSION_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "session_cache")


def _ensure_dir():
    os.makedirs(_SESSION_DB_DIR, exist_ok=True)


def _db_path(session_id: str) -> str:
    _ensure_dir()
    safe_id = session_id.replace("/", "_").replace("..", "_")
    return os.path.join(_SESSION_DB_DIR, f"session_{safe_id}.db")


def _get_conn(session_id: str) -> sqlite3.Connection:
    path = _db_path(session_id)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_assets (
            asset_id TEXT PRIMARY KEY,
            message_index INTEGER,
            file_type TEXT NOT NULL,
            summary TEXT,
            base64_hash TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vision_cache (
            cache_key TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            description TEXT NOT NULL,
            token_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def store_vision_summary(session_id: str, cache_key: str, model_id: str,
                         description: str, token_count: int = 0) -> None:
    """Cache a vision model's description for an image/PDF."""
    try:
        conn = _get_conn(session_id)
        conn.execute(
            "INSERT OR REPLACE INTO vision_cache (cache_key, model_id, description, token_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cache_key, model_id, description, token_count, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Cached vision summary for session={session_id[:8]} key={cache_key[:16]}")
    except Exception as e:
        logger.warning(f"Failed to cache vision summary: {e}")


def get_vision_summary(session_id: str, cache_key: str) -> Optional[str]:
    """Retrieve a cached vision description. Returns None if not cached."""
    try:
        conn = _get_conn(session_id)
        row = conn.execute(
            "SELECT description FROM vision_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        conn.close()
        return row["description"] if row else None
    except Exception as e:
        logger.warning(f"Failed to read vision cache: {e}")
        return None


def store_context_asset(session_id: str, asset_id: str, message_index: int,
                        file_type: str, summary: str = None,
                        base64_hash: str = None) -> None:
    """Track an uploaded asset in the session context."""
    try:
        conn = _get_conn(session_id)
        conn.execute(
            "INSERT OR REPLACE INTO context_assets "
            "(asset_id, message_index, file_type, summary, base64_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (asset_id, message_index, file_type, summary, base64_hash,
             datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to store context asset: {e}")


def get_session_context(session_id: str) -> List[Dict]:
    """Get all context assets for the session (for injecting into prompts)."""
    try:
        conn = _get_conn(session_id)
        rows = conn.execute(
            "SELECT * FROM context_assets ORDER BY message_index"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Failed to read session context: {e}")
        return []


def cleanup_session(session_id: str) -> None:
    """Delete the session's SQLite database file."""
    try:
        path = _db_path(session_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Cleaned up session cache: {session_id[:8]}")
    except Exception as e:
        logger.warning(f"Failed to cleanup session cache: {e}")
