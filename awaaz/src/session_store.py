"""Session storage for AWAAZ calls - typed, concurrent-safe state management."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class AWAAZSession:
    """Atomic call session state for AWAAZ."""

    # ── AVA core fields ─────────────────────────────────────────────────────
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel_id: str = ""
    call_start_ts: float = field(default_factory=time.time)
    state: str = "GREETING"  # GREETING/GATHERING/CONFIRMING/FILING/EMERGENCY/CLOSING
    turn_number: int = 0
    history: list = field(default_factory=list)
    is_active: bool = True
    barge_in_pending: bool = False

    # ── AWAAZ caller profile fields ──────────────────────────────────────────
    caller_ani: str = ""
    lang: str = "hi"
    lang_name: str = "Hindi"
    lang_mode: str = "pure"  # "pure" | "mixed"
    lang_distribution: dict = field(default_factory=dict)
    voice_choice: str = ""  # User's preferred voice for TTS (e.g., "indian_english" for Marathi)
    accent_region: str = "default"  # Regional accent: hindi-north, hindi-south, marathi-konkan, etc.
    phonetic_style: str = "native"  # "native" | "romanized" (for transliteration handling)
    formality_score: float = 0.5  # 0.0-1.0 (0=casual, 1=formal)
    formality_label: str = "STANDARD"  # CASUAL | STANDARD | FORMAL | OFFICIAL
    script: str = "Devanagari"
    gtts_lang: str = "hi"
    confidence: float = 0.0
    is_emergency: bool = False

    # ── Pre-call lookup result ───────────────────────────────────────────────
    citizen_record: dict = field(default_factory=dict)

    # ── Ticket fields ────────────────────────────────────────────────────────
    ticket_id: str = ""
    grievance_category: str = ""
    dept_assigned: str = ""
    priority: str = "NORMAL"
    complaint_summary: str = ""


class SessionStore:
    """Thread-safe store for AWAAZ call sessions."""

    def __init__(self):
        self._sessions: Dict[str, AWAAZSession] = {}
        self._by_channel: Dict[str, str] = {}  # channel_id -> session_id
        self._lock = asyncio.Lock()

    async def create(self, channel_id: str) -> AWAAZSession:
        """Create and store a new session."""
        session = AWAAZSession(channel_id=channel_id)
        async with self._lock:
            self._sessions[session.session_id] = session
            self._by_channel[channel_id] = session.session_id
        return session

    async def get(self, session_id: str) -> Optional[AWAAZSession]:
        """Get session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_by_channel(self, channel_id: str) -> Optional[AWAAZSession]:
        """Get session by channel ID."""
        async with self._lock:
            session_id = self._by_channel.get(channel_id)
            return self._sessions.get(session_id) if session_id else None

    async def update(self, session_id: str, **kwargs) -> None:
        """Update session fields."""
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)

    async def close(self, session_id: str) -> None:
        """Mark session as inactive."""
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.is_active = False

    async def remove(self, session_id: str) -> Optional[AWAAZSession]:
        """Remove session from store."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session and session.channel_id in self._by_channel:
                del self._by_channel[session.channel_id]
            return session

    async def get_all_active(self) -> List[AWAAZSession]:
        """Get all active sessions."""
        async with self._lock:
            return [s for s in self._sessions.values() if s.is_active]
