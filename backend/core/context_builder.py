"""
Context Window Builder — Sentinel-E v5.0

Constructs optimized context windows for MCO queries using:
  1. Recent messages (recency bias)
  2. User preferences (personalization)
  3. High-confidence user memory (learned knowledge)
  4. Semantic search results (relevance)

Token-aware trimming ensures context respects model token limits.
"""

import logging
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta

logger = logging.getLogger("ContextBuilder")

# Token estimation (rough, varies by tokenizer)
TOKENS_PER_CHAR = 0.25
TOKENS_PER_MESSAGE = 100  # Overhead per message


class ContextWindowBuilder:
    """Build optimized context windows for MCO queries."""
    
    def __init__(self, max_tokens: int = 2048, model_name: str = "llama33-70b"):
        """
        Initialize context builder.
        
        Args:
            max_tokens: Maximum tokens available for context
            model_name: Target model name (for token calculation)
        """
        self.max_tokens = max_tokens
        self.model_name = model_name
        
        # Reserve tokens for response
        self.response_reserve = 500
        self.available_tokens = max_tokens - self.response_reserve
    
    async def build_context(
        self,
        db: AsyncSession,
        user_id: str,
        query: str,
        recent_messages: List[Dict[str, Any]] = None,
        semantic_search_results: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete context window for the query.
        
        Priority order:
          1. Recent messages (establish conversation flow)
          2. User preferences (personalize response style)
          3. High-confidence memory (use learned knowledge)
          4. Semantic search results (find relevant context)
        
        Args:
            db: Database session
            user_id: User identifier
            query: Current query
            recent_messages: Last N messages from conversation
            semantic_search_results: Relevant past messages/knowledge
        
        Returns:
            Contextualized prompt with usage metrics
        """
        from backend.database.crud import get_user_memory, get_user_preference
        
        context_parts = []
        token_usage = {
            "recent_messages": 0,
            "user_preferences": 0,
            "user_memory": 0,
            "semantic_search": 0,
            "total": 0,
        }
        
        # ── 1. USER PREFERENCES ──────────────────────────────
        # Always include (typically small)
        try:
            prefs = await get_user_preference(db, user_id)
            if prefs:
                pref_prompt = self._format_preferences(prefs)
                context_parts.append(pref_prompt)
                token_usage["user_preferences"] = self._estimate_tokens(pref_prompt)
        except Exception as e:
            logger.warning(f"Failed to load user preferences: {e}")
        
        remaining_tokens = self.available_tokens - token_usage["user_preferences"]
        
        # ── 2. RECENT MESSAGES ───────────────────────────────
        # Most important for conversation continuity
        if recent_messages:
            msgs_prompt, msg_tokens = self._format_messages(
                recent_messages,
                max_tokens=int(remaining_tokens * 0.4),  # Allocate 40% to recent
            )
            context_parts.append(msgs_prompt)
            token_usage["recent_messages"] = msg_tokens
            remaining_tokens -= msg_tokens
        
        # ── 3. HIGH-CONFIDENCE MEMORY ────────────────────────
        # Learned facts about user/domain
        try:
            memories = await get_user_memory(
                db, user_id, min_confidence=75
            )
            if memories:
                mem_prompt, mem_tokens = self._format_memory(
                    memories,
                    max_tokens=int(remaining_tokens * 0.2),  # Allocate 20% to memory
                )
                context_parts.append(mem_prompt)
                token_usage["user_memory"] = mem_tokens
                remaining_tokens -= mem_tokens
        except Exception as e:
            logger.warning(f"Failed to load user memory: {e}")
        
        # ── 4. SEMANTIC SEARCH RESULTS ───────────────────────
        # Relevant past context (lowest priority due to volume)
        if semantic_search_results:
            search_prompt, search_tokens = self._format_search_results(
                semantic_search_results,
                max_tokens=int(remaining_tokens * 0.3),  # Allocate 30% to search
            )
            context_parts.append(search_prompt)
            token_usage["semantic_search"] = search_tokens
        
        # Build final context
        full_context = "\n\n".join(context_parts)
        token_usage["total"] = self._estimate_tokens(full_context)
        
        return {
            "context": full_context,
            "token_usage": token_usage,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
        }
    
    def _format_preferences(self, prefs) -> str:
        """Format user preferences as system context."""
        lines = [
            "[USER PREFERENCES]",
            f"Response Style: {prefs.response_style}",
            f"Tone: {prefs.tone}",
            f"Default Mode: {prefs.default_chat_mode}",
        ]
        
        if prefs.preferred_model:
            lines.append(f"Preferred Model: {prefs.preferred_model}")
        
        if prefs.show_reasoning:
            lines.append("Show Internal Reasoning: Yes")
        
        return "\n".join(lines)
    
    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
    ) -> tuple:
        """Format recent messages, respecting token limit."""
        lines = ["[CONVERSATION HISTORY]"]
        tokens_used = TOKENS_PER_MESSAGE  # Header overhead
        
        for msg in reversed(messages):  # Start from most recent
            if tokens_used >= max_tokens:
                break
            
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:500]  # Truncate long messages
            
            formatted = f"{role}: {content}"
            msg_tokens = self._estimate_tokens(formatted)
            
            if tokens_used + msg_tokens < max_tokens:
                lines.append(formatted)
                tokens_used += msg_tokens
            else:
                # Truncate to fit
                available = max_tokens - tokens_used
                if available > 50:  # Only add if meaningful
                    truncated = content[:int(available / TOKENS_PER_CHAR)]
                    lines.append(f"{role}: {truncated}...")
                break
        
        return "\n".join(reversed(lines)), tokens_used
    
    def _format_memory(
        self,
        memories: List[Any],
        max_tokens: int,
    ) -> tuple:
        """Format user memory facts, respecting token limit."""
        lines = ["[USER KNOWLEDGE]"]
        tokens_used = TOKENS_PER_MESSAGE  # Header overhead
        
        for mem in memories:
            if tokens_used >= max_tokens:
                break
            
            # Format as key=value with confidence
            key = mem.key
            value = mem.value
            conf = mem.confidence
            
            formatted = f"- {key}: {value} (confidence: {conf}%)"
            mem_tokens = self._estimate_tokens(formatted)
            
            if tokens_used + mem_tokens < max_tokens:
                lines.append(formatted)
                tokens_used += mem_tokens
        
        return "\n".join(lines), tokens_used
    
    def _format_search_results(
        self,
        results: List[Dict[str, Any]],
        max_tokens: int,
    ) -> tuple:
        """Format semantic search results, respecting token limit."""
        lines = ["[RELEVANT PAST CONTEXT]"]
        tokens_used = TOKENS_PER_MESSAGE  # Header overhead
        
        for result in results[:10]:  # Limit to top 10
            if tokens_used >= max_tokens:
                break
            
            content = result.get("content", "")[:300]
            similarity = result.get("similarity", 0)
            
            formatted = f"- {content} (relevance: {similarity:.2f})"
            res_tokens = self._estimate_tokens(formatted)
            
            if tokens_used + res_tokens < max_tokens:
                lines.append(formatted)
                tokens_used += res_tokens
        
        return "\n".join(lines), tokens_used
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        return max(1, int(len(text) * TOKENS_PER_CHAR))


# Singleton instance
_builder = None


def get_context_builder(max_tokens: int = 2048, model: str = "llama33-70b") -> ContextWindowBuilder:
    """Get or create context builder instance."""
    global _builder
    if _builder is None:
        _builder = ContextWindowBuilder(max_tokens, model)
    return _builder
