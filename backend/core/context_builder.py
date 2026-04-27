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
RECENT_WEIGHT = 1.0
CROSS_SESSION_WEIGHT = 0.8
SEMANTIC_WEIGHT = 0.7
MEMORY_WEIGHT = 0.6
MAX_CONTEXT_ITEMS = 12
SEMANTIC_SIMILARITY_THRESHOLD = 0.2

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

        semantic_results = []
        try:
            if semantic_search_results:
                for item in semantic_search_results[:3]:
                    if isinstance(item, dict):
                        content = item.get("content") or item.get("text") or ""
                        if content:
                            semantic_results.append({
                                "content": str(content),
                                "score": float(item.get("score", 1.0)),
                            })
        except Exception:
            semantic_results = []

        return self._assemble_and_trim(recent_messages, prefs, memories, semantic_results, query=query)

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
        logger.info("context status db=success op=load_recent_messages user_id=%s", user_id)

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
        semantic_results = []
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
                    semantic_results = [
                        {
                            "content": str((res.get("metadata") or {}).get("content", "") or "").strip(),
                            "score": float(res.get("score", 0.0) or 0.0),
                        }
                        for res in (search_results or [])
                        if str((res.get("metadata") or {}).get("content", "") or "").strip()
                    ]
                    logger.info("context status pinecone=success op=semantic_query user_id=%s", user_id)
                else:
                    logger.info("context status pinecone=skip op=semantic_query reason=no_embedding user_id=%s", user_id)
            else:
                logger.info("context status pinecone=skip op=semantic_query reason=service_unavailable user_id=%s", user_id)
        except Exception:
            logger.warning("context status pinecone=fail op=semantic_query user_id=%s", user_id)
            semantic_results = []  # semantic retrieval is always optional

        # 5.5 Cross-session retrieval (same user, different chats)
        cross_session_results = []
        try:
            past_result = await db.execute(
                select(Message)
                .where(
                    Message.user_id == user_id,
                    Message.chat_id != chat_id,
                )
                .order_by(Message.created_at.desc())
                .limit(200)
            )
            past_messages = past_result.scalars().all()

            topic_counts: Dict[str, int] = {}
            topic_sessions: Dict[str, set] = {}
            query_topics = self._extract_topics(query)

            for pm in past_messages:
                content = str(getattr(pm, "content", "") or "").strip()
                if not content:
                    continue
                for topic in self._extract_topics(content):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    topic_sessions.setdefault(topic, set()).add(str(getattr(pm, "chat_id", "")))

            for pm in past_messages:
                content = str(getattr(pm, "content", "") or "").strip()
                if not content:
                    continue

                sim = self._similarity_hint(query, content)
                content_topics = self._extract_topics(content)
                repeated_topic_hit = any(topic_counts.get(t, 0) >= 2 for t in content_topics.intersection(query_topics))
                if sim < SEMANTIC_SIMILARITY_THRESHOLD and not repeated_topic_hit:
                    continue

                multi_session_bonus = 0.0
                for t in content_topics.intersection(query_topics):
                    if len(topic_sessions.get(t, set())) > 1:
                        multi_session_bonus += 0.1
                multi_session_bonus = min(multi_session_bonus, 0.3)

                meta = getattr(pm, "metadata_json", None) or {}
                visual_context = self._visual_metadata_context(meta)
                stitched = content
                if visual_context:
                    stitched = f"{stitched}\n{visual_context}"

                cross_session_results.append({
                    "content": stitched,
                    "score": min(sim + multi_session_bonus, 1.0),
                })
        except Exception as cross_err:
            logger.warning("context status db=fail op=cross_session user_id=%s error=%s", user_id, cross_err)
            cross_session_results = []

        # 6. Assembly
        context_bundle = self._assemble_and_trim(
            messages,
            prefs,
            memories,
            semantic_results,
            query=query,
            cross_session=cross_session_results,
        )

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

    def _similarity_hint(self, query: str, text: str) -> float:
        """Cheap lexical similarity hint in [0, 1]."""
        q = set(str(query or "").lower().split())
        t = set(str(text or "").lower().split())
        if not q or not t:
            return 0.0
        return len(q.intersection(t)) / max(len(q), 1)

    def _extract_topics(self, text: str) -> set:
        words = [w.strip(".,:;!?()[]{}\"'").lower() for w in str(text or "").split()]
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
            "are", "was", "were", "be", "this", "that", "it", "as", "at", "by", "from",
            "i", "you", "we", "they", "he", "she", "them", "our", "your",
        }
        return {w for w in words if len(w) > 2 and w not in stop}

    def _visual_metadata_context(self, metadata: Dict[str, Any]) -> str:
        if not isinstance(metadata, dict):
            return ""
        if metadata.get("type") != "image":
            return ""
        desc = str(metadata.get("description") or "").strip()
        tags = metadata.get("tags") or []
        tags_str = ", ".join([str(t).strip() for t in tags if str(t).strip()])
        if not desc and not tags_str:
            return ""
        return f"[User shared: {desc or 'image'} | tags: {tags_str or 'none'}]"

    def _assemble_and_trim(self, messages, prefs, memories, semantic, query: str = "", cross_session: Optional[List[Dict[str, Any]]] = None):
        """Assemble context with weighted ranking while keeping within token limit."""
        history = self._normalize_history(messages)

        # Recency is REQUIRED and always represented by recent_history (weight 1.0)
        _recent_score = RECENT_WEIGHT  # explicit constant for clarity/logical contract

        ranked_items: List[Dict[str, Any]] = []

        semantic_items = semantic if isinstance(semantic, list) else []
        for item in semantic_items:
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            similarity = float(item.get("score", 0.0) or 0.0)
            if similarity <= 0:
                similarity = self._similarity_hint(query, content)
            ranked_items.append({
                "section": "Semantic",
                "content": content,
                "score": similarity * SEMANTIC_WEIGHT,
            })

        cross_items = cross_session if isinstance(cross_session, list) else []
        for item in cross_items:
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            similarity = float(item.get("score", 0.0) or 0.0)
            if similarity <= 0:
                similarity = self._similarity_hint(query, content)
            ranked_items.append({
                "section": "CrossSession",
                "content": content,
                "score": similarity * CROSS_SESSION_WEIGHT,
            })

        for mem in memories or []:
            value = str(getattr(mem, "value", "") or "").strip()
            if not value:
                continue
            confidence = float(getattr(mem, "confidence", 70) or 70) / 100.0
            similarity = max(self._similarity_hint(query, value), 0.25)
            ranked_items.append({
                "section": "Memory",
                "content": f"Fact: {value}",
                "score": min(confidence, 1.0) * similarity * MEMORY_WEIGHT,
            })

        for k, v in (prefs or {}).items():
            pref_line = f"{k}: {v}"
            similarity = max(self._similarity_hint(query, pref_line), 0.2)
            ranked_items.append({
                "section": "Memory",
                "content": f"Preference: {pref_line}",
                "score": similarity * MEMORY_WEIGHT,
            })

        ranked_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        ranked_items = ranked_items[:MAX_CONTEXT_ITEMS]

        semantic_lines = [it["content"] for it in ranked_items if it["section"] == "Semantic"]
        cross_session_lines = [it["content"] for it in ranked_items if it["section"] == "CrossSession"]
        memory_lines = [it["content"] for it in ranked_items if it["section"] == "Memory"]

        parts = []
        if cross_session_lines:
            parts.append("Cross-Session Context:\n" + "\n".join(cross_session_lines))
        if semantic_lines:
            parts.append("Semantic Context:\n" + "\n".join(semantic_lines))
        if memory_lines:
            parts.append("User Memory:\n" + "\n".join(memory_lines))

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

