import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timedelta
from database.models import ContextWindow, Message, UserMemory, UserPreference
from utils.vector_service import get_vector_service

logger = logging.getLogger("ContextBuilder")

# Token estimation
TOKENS_PER_CHAR = 0.25
DEFAULT_TOKEN_LIMIT = 4096

class ContextWindowBuilder:
    def __init__(self, token_limit: int = DEFAULT_TOKEN_LIMIT):
        self.token_limit = token_limit
        self.vector_service = get_vector_service()

    async def build_context(
        self, 
        db: AsyncSession, 
        user_id: str, 
        chat_id: Any, 
        query: str,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        build_context(user_id, chat_id, query) → ContextBundle
        Pipeline:
        1. Reuse check (< 5 min)
        2. Load last N messages
        3. Inject user_preferences
        4. Inject high-confidence user_memory (> 0.7)
        5. Semantic retrieval (Pinecone)
        6. Deduplicate & Enforce Token Limit
        7. Store in context_windows table
        """
        
        # 1. Reuse Check
        if not force_rebuild:
            existing = await db.execute(
                select(ContextWindow).where(
                    ContextWindow.user_id == user_id,
                    ContextWindow.chat_id == chat_id,
                    ContextWindow.updated_at > datetime.utcnow() - timedelta(minutes=5)
                )
            )
            context_win = existing.scalars().first()
            if context_win:
                logger.info(f"Reusing existing context window for chat {chat_id}")
                return context_win.context_json

        logger.info(f"Building new context window for chat {chat_id}")
        
        # 2. Load last N messages (Start with a large chunk, we will trim)
        msg_result = await db.execute(
            select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.desc()).limit(50)
        )
        messages = msg_result.scalars().all()
        messages = sorted(messages, key=lambda x: x.created_at)

        # 3. Inject user_preferences (all)
        pref_result = await db.execute(
            select(UserPreference).where(UserPreference.user_id == user_id)
        )
        prefs = pref_result.scalars().first()
        
        # 4. Inject high-confidence user_memory (confidence > 70)
        mem_result = await db.execute(
            select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.confidence > 70
            )
        )
        memories = mem_result.scalars().all()

        # 5. Semantic retrieval (Pinecone)
        semantic_context = ""
        query_vector = await self.vector_service.get_embedding(query)
        if query_vector:
            search_results = await self.vector_service.query(
                namespace="chat_messages",
                vector=query_vector,
                top_k=5,
                filter={"user_id": {"$eq": user_id}}
            )
            semantic_parts = []
            for res in search_results:
                content = res["metadata"].get("content", "")
                semantic_parts.append(f"[Past Interaction]: {content}")
            semantic_context = "\n".join(semantic_parts)

        # 6. Assembly & Token Trimming (Priority: recent > preferences > memory > semantic)
        context_bundle = self._assemble_and_trim(
            messages=messages,
            prefs=prefs,
            memories=memories,
            semantic=semantic_context
        )

        # 7. Store in context_windows table
        await self._store_context(db, user_id, chat_id, context_bundle)

        return context_bundle

    def _assemble_and_trim(self, messages, prefs, memories, semantic) -> Dict[str, Any]:
        """Priority order: recent msgs > preferences > memory > semantic"""
        
        # 1. Preferences
        pref_str = ""
        if prefs:
            pref_str = f"[USER PREFERENCES]\nStyle: {prefs.response_style}\nTone: {prefs.tone}\nMode: {prefs.default_chat_mode}\n"
        
        # 2. Memory
        mem_str = ""
        if memories:
            mem_str = "[USER KNOWLEDGE]\n" + "\n".join([f"- {m.key}: {m.value}" for m in memories]) + "\n"

        # 3. Semantic
        sem_str = ""
        if semantic:
            sem_str = "[RELEVANT PAST CONTEXT]\n" + semantic + "\n"

        # 4. Recent Messages (Dynamic)
        # We start with the fixed components
        base_context = pref_str + mem_str + sem_str
        base_tokens = self._estimate_tokens(base_context)
        
        available_for_messages = self.token_limit - base_tokens
        
        final_messages = []
        msg_tokens = 0
        for m in reversed(messages):
            m_text = f"{m.role.upper()}: {m.content}\n"
            m_tokens = self._estimate_tokens(m_text)
            if msg_tokens + m_tokens < available_for_messages:
                final_messages.insert(0, m_text)
                msg_tokens += m_tokens
            else:
                break
        
        history_str = "[CONVERSATION HISTORY]\n" + "".join(final_messages)
        
        full_context = pref_str + mem_str + sem_str + history_str
        
        return {
            "context": full_context,
            "token_count": self._estimate_tokens(full_context),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _store_context(self, db, user_id, chat_id, bundle):
        try:
            # Upsert logic
            existing = await db.execute(
                select(ContextWindow).where(
                    ContextWindow.user_id == user_id,
                    ContextWindow.chat_id == chat_id
                )
            )
            context_win = existing.scalars().first()
            
            if context_win:
                context_win.context_json = bundle
                context_win.token_count = bundle["token_count"]
                context_win.updated_at = datetime.utcnow()
            else:
                new_win = ContextWindow(
                    user_id=user_id,
                    chat_id=chat_id,
                    context_json=bundle,
                    token_count=bundle["token_count"]
                )
                db.add(new_win)
            
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to store context window: {e}")
            await db.rollback()

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) * TOKENS_PER_CHAR))

_builder = None
def get_context_builder(token_limit: int = DEFAULT_TOKEN_LIMIT) -> ContextWindowBuilder:
    global _builder
    if _builder is None:
        _builder = ContextWindowBuilder(token_limit)
    return _builder
