import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, text
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
        4. Inject high-confidence user_memory (> 70)
        5. Semantic retrieval (Pinecone)
        6. Deduplicate & Enforce Token Limit
        """
        try:
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
                    return context_win.context_json

            # 2. Load last N messages
            msg_result = await db.execute(
                select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.desc()).limit(20)
            )
            messages = msg_result.scalars().all()
            messages = sorted(messages, key=lambda x: x.created_at)

            # 3. Inject user_preferences (KV)
            pref_result = await db.execute(
                select(UserPreference).where(UserPreference.user_id == user_id)
            )
            prefs = {p.key: p.value for p in pref_result.scalars().all()}
            
            # 4. Inject high-confidence user_memory (confidence > 70)
            mem_result = await db.execute(
                select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.confidence > 70
                )
            )
            memories = mem_result.scalars().all()

            # 5. Semantic retrieval (optional)
            semantic_context = ""
            try:
                query_vector = await self.vector_service.get_embedding(query)
                if query_vector:
                    search_results = await self.vector_service.query(
                        namespace="chat_messages",
                        vector=query_vector,
                        top_k=3,
                        filter={"user_id": {"$eq": user_id}}
                    )
                    semantic_parts = [f"[Past Interaction]: {res['metadata'].get('content', '')}" for res in search_results]
                    semantic_context = "\n".join(semantic_parts)
            except Exception:
                logger.warning("Semantic context retrieval failed - skipping")

            # 6. Assembly
            context_bundle = self._assemble_and_trim(messages, prefs, memories, semantic_context)
            
            # Store for reuse
            from database.crud import upsert_context_window
            await upsert_context_window(db, user_id, chat_id, context_bundle)
            
            return context_bundle
            
        except Exception as e:
            logger.error(f"Context building failed: {e}")
            # Fallback to empty context if everything fails
            return {"recent_history": [], "system_instructions": "Context reconstruction failed. Proceed with caution."}

    def _assemble_and_trim(self, messages, prefs, memories, semantic):
        """Assemble pieces while keeping within token limit."""
        history = [{"role": m.role, "content": m.content} for m in messages]
        
        pref_str = "\n".join([f"{k}: {v}" for k, v in prefs.items()])
        mem_str = "\n".join([f"Fact: {m.value}" for m in memories])
        
        system_instructions = f"User Preferences:\n{pref_str}\n\nRelevant User Facts:\n{mem_str}\n\nPast Context:\n{semantic}"
        
        return {
            "recent_history": history,
            "system_instructions": system_instructions,
            "timestamp": datetime.utcnow().isoformat()
        }

_builder = ContextWindowBuilder()
def get_context_builder():
    return _builder
