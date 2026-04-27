import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, text
from datetime import datetime, timedelta
from database.models import ContextWindow, Message, UserMemory, UserPreference

logger = logging.getLogger("ContextBuilder")

# Token estimation
TOKENS_PER_CHAR = 0.25
DEFAULT_TOKEN_LIMIT = 4096
MESSAGE_CONTEXT_LIMIT = 10  # last N messages for context

class ContextWindowBuilder:
    def __init__(self, token_limit: int = DEFAULT_TOKEN_LIMIT):
        self.token_limit = token_limit
        self._vector_service = None  # lazy init

    def _get_vector_service(self):
        """Lazy-load vector service so missing deps don't crash at startup."""
        if self._vector_service is None:
            try:
                from utils.vector_service import get_vector_service
                self._vector_service = get_vector_service()
            except Exception:
                self._vector_service = None  # stays None — semantic retrieval skipped
        return self._vector_service

    async def build_context(
        self, 
        db: AsyncSession, 
        user_id: str, 
        chat_id: Any = None,
        query: str = "",
        force_rebuild: bool = False,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        semantic_search_results: Optional[List[Dict[str, Any]]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """
        build_context(user_id, chat_id, query) → ContextBundle
        Safe: falls back to recent messages only if anything fails.
        """
        try:
            # Compatibility path: some callers provide recent_messages/query but no chat_id
            if chat_id is None and recent_messages is not None:
                return await self._build_context_from_recent(
                    db=db,
                    user_id=user_id,
                    query=query,
                    recent_messages=recent_messages,
                    semantic_search_results=semantic_search_results,
                )
            return await self._build_context_inner(db, user_id, chat_id, query, force_rebuild)
        except Exception as e:
            logger.warning(f"Context builder failed, using recent messages only: {e}")
            if recent_messages is not None:
                safe_history = self._normalize_history(recent_messages)
                return {
                    "recent_history": safe_history,
                    "system_instructions": "",
                    "context_str": "",
                    "context": "",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            return await self._fallback_recent_messages(db, chat_id)

    async def _build_context_from_recent(
        self,
        db: AsyncSession,
        user_id: str,
        query: str,
        recent_messages: List[Dict[str, Any]],
        semantic_search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build context when caller already provides recent message payload."""
        prefs = {}
        memories = []
        try:
            pref_result = await db.execute(
                select(UserPreference).where(UserPreference.user_id == user_id)
            )
            prefs = {p.key: p.value for p in pref_result.scalars().all()}
        except Exception:
            pass

        try:
            mem_result = await db.execute(
                select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.confidence > 70
                )
            )
            memories = mem_result.scalars().all()
        except Exception:
            pass

        semantic_context = ""
        try:
            if semantic_search_results:
                semantic_parts = []
                for item in semantic_search_results[:3]:
                    if isinstance(item, dict):
                        content = item.get("content") or item.get("text") or ""
                        if content:
                            semantic_parts.append(f"[Past Interaction]: {content}")
                semantic_context = "\n".join(semantic_parts)
        except Exception:
            semantic_context = ""

        return self._assemble_and_trim(recent_messages, prefs, memories, semantic_context)

    async def _fallback_recent_messages(self, db: AsyncSession, chat_id: Any) -> Dict[str, Any]:
        """Minimal safe fallback: return last N messages, no external deps."""
        try:
            msg_result = await db.execute(
                select(Message)
                .where(Message.chat_id == chat_id)
                .order_by(Message.created_at.desc())
                .limit(MESSAGE_CONTEXT_LIMIT)
            )
            messages = sorted(msg_result.scalars().all(), key=lambda x: x.created_at)
            history = [{"role": m.role, "content": m.content} for m in messages]
            history = self._normalize_history(history)
            return {
                "recent_history": history,
                "system_instructions": "",
                "context_str": "",
                "context": "",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception:
            return {
                "recent_history": [],
                "system_instructions": "",
                "context_str": "",
                "context": "",
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _build_context_inner(
        self,
        db: AsyncSession,
        user_id: str,
        chat_id: Any,
        query: str,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Full context pipeline:
        1. Reuse check (< 5 min)
        2. Load last N messages
        3. Inject user_preferences (optional)
        4. Inject high-confidence user_memory > 70 (optional)
        5. Semantic retrieval (optional, skip silently if unavailable)
        6. Assemble & enforce token limit
        """
        # 1. Reuse Check
        if not force_rebuild:
            try:
                existing = await db.execute(
                    select(ContextWindow).where(
                        ContextWindow.user_id == user_id,
                        ContextWindow.chat_id == chat_id,
                        ContextWindow.updated_at > datetime.utcnow() - timedelta(minutes=5)
                    )
                )
                context_win = existing.scalars().first()
                if context_win:
                    return context_win.context_json
            except Exception:
                pass  # reuse check failed — rebuild

        # 2. Load last N messages
        msg_result = await db.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.desc())
            .limit(MESSAGE_CONTEXT_LIMIT)
        )
        messages = sorted(msg_result.scalars().all(), key=lambda x: x.created_at)

        # 3. Inject user_preferences (optional)
        prefs = {}
        try:
            pref_result = await db.execute(
                select(UserPreference).where(UserPreference.user_id == user_id)
            )
            prefs = {p.key: p.value for p in pref_result.scalars().all()}
        except Exception:
            pass

        # 4. Inject high-confidence user_memory (optional)
        memories = []
        try:
            mem_result = await db.execute(
                select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.confidence > 70
                )
            )
            memories = mem_result.scalars().all()
        except Exception:
            pass

        # 5. Semantic retrieval (optional — skip silently)
        semantic_context = ""
        try:
            vs = self._get_vector_service()
            if vs:
                query_vector = await vs.get_embedding(query)
                if query_vector:
                    search_results = await vs.query(
                        namespace="chat_messages",
                        vector=query_vector,
                        top_k=3,
                        filter={"user_id": {"$eq": user_id}}
                    )
                    semantic_parts = [
                        f"[Past Interaction]: {res['metadata'].get('content', '')}"
                        for res in search_results
                    ]
                    semantic_context = "\n".join(semantic_parts)
        except Exception:
            pass  # semantic retrieval is always optional

        # 6. Assembly
        context_bundle = self._assemble_and_trim(messages, prefs, memories, semantic_context)

        # Store for reuse (best-effort, non-blocking)
        try:
            from database.crud import upsert_context_window
            await upsert_context_window(db, user_id, chat_id, context_bundle)
        except Exception:
            pass

        return context_bundle

    def _normalize_history(self, messages) -> List[Dict[str, str]]:
        """Normalize, dedupe, and cap message history."""
        normalized = []
        seen = set()

        for m in messages or []:
            if isinstance(m, dict):
                role = str(m.get("role") or "user")
                content = str(m.get("content") or "").strip()
            else:
                role = str(getattr(m, "role", "user"))
                content = str(getattr(m, "content", "") or "").strip()

            if not content:
                continue

            key = (role, content)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"role": role, "content": content})

        return normalized[-MESSAGE_CONTEXT_LIMIT:]

    def _assemble_and_trim(self, messages, prefs, memories, semantic):
        """Assemble pieces while keeping within token limit."""
        history = self._normalize_history(messages)

        parts = []
        if prefs:
            pref_str = "\n".join([f"{k}: {v}" for k, v in prefs.items()])
            parts.append(f"User Preferences:\n{pref_str}")
        if memories:
            mem_str = "\n".join([f"Fact: {m.value}" for m in memories])
            parts.append(f"Relevant User Facts:\n{mem_str}")
        if semantic:
            parts.append(f"Past Context:\n{semantic}")

        system_instructions = "\n\n".join(parts)

        # Keep context deterministic and within limit
        history_chars = sum(len(m.get("content", "")) for m in history)
        max_chars = int(self.token_limit / max(TOKENS_PER_CHAR, 0.001))
        available_for_system = max(max_chars - history_chars, 0)
        if available_for_system > 0:
            system_instructions = system_instructions[:available_for_system]
        else:
            system_instructions = ""

        return {
            "recent_history": history,
            "system_instructions": system_instructions,
            "context_str": system_instructions,
            "context": system_instructions,
            "timestamp": datetime.utcnow().isoformat()
        }

_builder = ContextWindowBuilder()
def get_context_builder():
    return _builder

