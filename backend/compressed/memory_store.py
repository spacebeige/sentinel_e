"""
SQLite session memory for the compressed pipeline.
Tables: sessions, messages, web_cache, token_usage.
"""

import aiosqlite
import json
import os
import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("compressed.memory")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "compressed_memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    timestamp  REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS web_cache (
    url        TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    ttl        INTEGER DEFAULT 3600
);

CREATE TABLE IF NOT EXISTS token_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    node_name  TEXT NOT NULL,
    tokens_in  INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    timestamp  REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_token_session ON token_usage(session_id);
"""

MAX_CONTEXT_MESSAGES = 12


@dataclass
class SessionContext:
    session_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0


class CompressedMemory:
    """Async SQLite-backed session memory."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._initialized = False

    async def _ensure_init(self):
        if self._initialized:
            return
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()
        self._initialized = True

    async def _get_db(self) -> aiosqlite.Connection:
        await self._ensure_init()
        return await aiosqlite.connect(self._db_path)

    # ── Session Management ──

    async def get_or_create_session(self, session_id: str) -> SessionContext:
        db = await self._get_db()
        try:
            now = time.time()
            cursor = await db.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if not row:
                await db.execute(
                    "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now),
                )
                await db.commit()

            # Load recent messages
            cursor = await db.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, MAX_CONTEXT_MESSAGES),
            )
            rows = await cursor.fetchall()
            history = [{"role": r[0], "content": r[1]} for r in reversed(rows)]

            # Load token totals
            cursor = await db.execute(
                "SELECT COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0) FROM token_usage WHERE session_id = ?",
                (session_id,),
            )
            tok_row = await cursor.fetchone()

            return SessionContext(
                session_id=session_id,
                history=history,
                total_tokens_in=tok_row[0] if tok_row else 0,
                total_tokens_out=tok_row[1] if tok_row else 0,
            )
        finally:
            await db.close()

    async def add_message(self, session_id: str, role: str, content: str):
        db = await self._get_db()
        try:
            now = time.time()
            await db.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            await db.commit()
        finally:
            await db.close()

    async def record_token_usage(
        self, session_id: str, node_name: str, tokens_in: int, tokens_out: int
    ):
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO token_usage (session_id, node_name, tokens_in, tokens_out, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, node_name, tokens_in, tokens_out, time.time()),
            )
            await db.commit()
        finally:
            await db.close()

    async def get_session_token_total(self, session_id: str) -> Dict[str, int]:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0) FROM token_usage WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            return {"tokens_in": row[0], "tokens_out": row[1]} if row else {"tokens_in": 0, "tokens_out": 0}
        finally:
            await db.close()

    # ── Web Cache ──

    async def get_cached_summary(self, url: str) -> Optional[str]:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT summary, fetched_at, ttl FROM web_cache WHERE url = ?",
                (url,),
            )
            row = await cursor.fetchone()
            if row and (time.time() - row[1]) < row[2]:
                return row[0]
            return None
        finally:
            await db.close()

    async def cache_summary(self, url: str, summary: str, ttl: int = 3600):
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO web_cache (url, summary, fetched_at, ttl) VALUES (?, ?, ?, ?)",
                (url, summary, time.time(), ttl),
            )
            await db.commit()
        finally:
            await db.close()

    def build_context_prompt(self, session_ctx: SessionContext) -> str:
        """Build a compressed context string from session history."""
        if not session_ctx.history:
            return ""
        lines = []
        for msg in session_ctx.history[-MAX_CONTEXT_MESSAGES:]:
            role = msg["role"].upper()
            content = msg["content"][:500]  # Truncate long messages
            lines.append(f"[{role}]: {content}")
        return "Previous conversation:\n" + "\n".join(lines)
