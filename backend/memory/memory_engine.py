"""
============================================================
3-Tier Memory Engine — Enterprise Session Memory
============================================================

Tier 1: Short-Term Memory
    - Last N messages (configurable, default 12)
    - Injected into prompt dynamically
    - Provides immediate conversational context

Tier 2: Rolling Summary Memory
    - Every N exchanges, summarize conversation
    - Store compact summary (< 500 tokens)
    - Replace long context to prevent token explosion
    - Uses LLM for summarization

Tier 3: Persistent User Memory (Database)
    - Preferred tone, model, feedback history
    - Mode preference, repeated corrections, topic interests
    - Learns from thumbs up/down
    - Adjusts routing weights
    - Respects prior instructions

Memory is contextual, not keyword-triggered.
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger("MemoryEngine")


# ============================================================
# Tier 1: Short-Term Memory
# ============================================================

@dataclass
class ShortTermMemory:
    """
    Maintains the last N messages for immediate context.
    Lightweight, in-memory, per-session.
    """
    max_messages: int = 12
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Trim to max
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Return messages formatted for prompt injection."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def get_last_n(self, n: int = 6) -> List[Dict[str, str]]:
        return self.messages[-n:]

    def clear(self):
        self.messages = []

    def to_dict(self) -> Dict:
        return {"max_messages": self.max_messages, "messages": self.messages}

    @classmethod
    def from_dict(cls, data: Dict) -> "ShortTermMemory":
        mem = cls(max_messages=data.get("max_messages", 12))
        mem.messages = data.get("messages", [])
        return mem


# ============================================================
# Tier 2: Rolling Summary Memory
# ============================================================

@dataclass
class RollingSummary:
    """
    Periodically compresses conversation history into summaries.
    Prevents token explosion while maintaining context.
    """
    summaries: List[Dict[str, Any]] = field(default_factory=list)
    exchange_count: int = 0
    summary_interval: int = 8  # Summarize every N exchanges
    max_summary_count: int = 10  # Keep at most N summaries

    def should_summarize(self) -> bool:
        return self.exchange_count >= self.summary_interval

    def record_exchange(self):
        self.exchange_count += 1

    def add_summary(self, summary_text: str, messages_covered: int):
        self.summaries.append({
            "text": summary_text,
            "messages_covered": messages_covered,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        # Trim oldest
        if len(self.summaries) > self.max_summary_count:
            self.summaries = self.summaries[-self.max_summary_count:]
        self.exchange_count = 0  # Reset counter

    def get_context_text(self) -> str:
        """Return all summaries as context."""
        if not self.summaries:
            return ""
        parts = []
        for i, s in enumerate(self.summaries[-3:], 1):  # Last 3 summaries
            parts.append(f"[Previous Context {i}]: {s['text']}")
        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "summaries": self.summaries,
            "exchange_count": self.exchange_count,
            "summary_interval": self.summary_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RollingSummary":
        rs = cls(
            summary_interval=data.get("summary_interval", 8),
        )
        rs.summaries = data.get("summaries", [])
        rs.exchange_count = data.get("exchange_count", 0)
        return rs


# ============================================================
# Tier 3: Persistent User Memory
# ============================================================

@dataclass
class UserPreferences:
    """
    Persistent user preferences learned from interactions.
    Stored in database, survives sessions.
    """
    user_id: str = ""
    preferred_tone: str = "neutral"  # formal | casual | neutral | technical
    preferred_verbosity: str = "balanced"  # concise | balanced | detailed
    preferred_mode: str = "standard"
    topic_interests: Dict[str, float] = field(default_factory=dict)  # topic -> affinity score
    corrections: List[Dict[str, str]] = field(default_factory=list)  # user corrections
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    model_preferences: Dict[str, float] = field(default_factory=dict)  # model -> preference weight
    last_updated: str = ""

    def record_feedback(self, vote: str, rating: Optional[int] = None, reason: Optional[str] = None,
                       mode: Optional[str] = None, topic: Optional[str] = None):
        """Learn from user feedback."""
        entry = {
            "vote": vote,
            "rating": rating,
            "reason": reason,
            "mode": mode,
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.feedback_history.append(entry)
        # Keep last 100 feedback entries
        if len(self.feedback_history) > 100:
            self.feedback_history = self.feedback_history[-100:]

        if vote == "up" or (rating and rating >= 4):
            self.positive_feedback_count += 1
            if mode:
                self.model_preferences[mode] = self.model_preferences.get(mode, 0.5) + 0.05
        elif vote == "down" or (rating and rating <= 2):
            self.negative_feedback_count += 1
            if mode:
                self.model_preferences[mode] = max(0.1, self.model_preferences.get(mode, 0.5) - 0.05)

        # Learn topic interest
        if topic:
            self.topic_interests[topic] = self.topic_interests.get(topic, 0.5) + 0.1
            # Normalize
            max_val = max(self.topic_interests.values()) if self.topic_interests else 1
            if max_val > 1:
                for k in self.topic_interests:
                    self.topic_interests[k] /= max_val

        self.last_updated = datetime.now(timezone.utc).isoformat()

    def record_correction(self, original: str, correction: str):
        """Track user corrections for future reference."""
        self.corrections.append({
            "original": original[:200],
            "correction": correction[:200],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.corrections) > 50:
            self.corrections = self.corrections[-50:]

    def infer_tone(self) -> str:
        """Infer preferred tone from feedback patterns."""
        # Simple heuristic: if user frequently gives positive feedback in a mode,
        # that mode's tone is preferred
        if self.positive_feedback_count > self.negative_feedback_count * 2:
            return self.preferred_tone  # User is generally happy
        return "neutral"  # Default when uncertain

    def get_routing_weights(self) -> Dict[str, float]:
        """Get model-preference-adjusted routing weights."""
        defaults = {
            "standard": 0.5,
            "experimental": 0.5,
            "debate": 0.5,
            "glass": 0.5,
            "evidence": 0.5,
        }
        for key, val in self.model_preferences.items():
            if key in defaults:
                defaults[key] = max(0.1, min(1.0, val))
        return defaults

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "preferred_tone": self.preferred_tone,
            "preferred_verbosity": self.preferred_verbosity,
            "preferred_mode": self.preferred_mode,
            "topic_interests": self.topic_interests,
            "corrections": self.corrections[-10:],  # Only last 10 for serialization
            "positive_feedback_count": self.positive_feedback_count,
            "negative_feedback_count": self.negative_feedback_count,
            "model_preferences": self.model_preferences,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserPreferences":
        prefs = cls()
        for key in ["user_id", "preferred_tone", "preferred_verbosity", "preferred_mode",
                     "last_updated"]:
            if key in data:
                setattr(prefs, key, data[key])
        prefs.topic_interests = data.get("topic_interests", {})
        prefs.corrections = data.get("corrections", [])
        prefs.feedback_history = data.get("feedback_history", [])
        prefs.positive_feedback_count = data.get("positive_feedback_count", 0)
        prefs.negative_feedback_count = data.get("negative_feedback_count", 0)
        prefs.model_preferences = data.get("model_preferences", {})
        return prefs


# ============================================================
# Unified Memory Manager
# ============================================================

class MemoryEngine:
    """
    Orchestrates all 3 memory tiers for a session.
    Singleton per chat session, persisted via Redis + Postgres.
    """

    def __init__(self, user_id: str = "", settings=None):
        from backend.gateway.config import get_settings
        self.settings = settings or get_settings()
        self.user_id = user_id
        self.short_term = ShortTermMemory(max_messages=self.settings.SHORT_TERM_MEMORY_SIZE)
        self.rolling_summary = RollingSummary(summary_interval=self.settings.ROLLING_SUMMARY_INTERVAL)
        self.user_prefs = UserPreferences(user_id=user_id)

    def add_message(self, role: str, content: str):
        """Record a message across memory tiers."""
        self.short_term.add(role, content)
        if role == "assistant":
            self.rolling_summary.record_exchange()

    def needs_summarization(self) -> bool:
        return self.rolling_summary.should_summarize()

    def build_prompt_context(self) -> str:
        """
        Build the memory context to inject into prompts.
        Combines rolling summaries + recent short-term messages.
        """
        parts = []

        # Rolling summary context (broader history)
        summary_ctx = self.rolling_summary.get_context_text()
        if summary_ctx:
            parts.append(summary_ctx)

        # User preference hints
        if self.user_prefs.preferred_tone != "neutral":
            parts.append(f"[User prefers {self.user_prefs.preferred_tone} tone]")
        if self.user_prefs.preferred_verbosity != "balanced":
            parts.append(f"[User prefers {self.user_prefs.preferred_verbosity} responses]")

        # Recent corrections
        recent_corrections = self.user_prefs.corrections[-3:]
        for corr in recent_corrections:
            parts.append(f"[Previous correction: '{corr['original']}' → '{corr['correction']}']")

        return "\n".join(parts) if parts else ""

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Get the full message list for prompt construction.
        Memory context as system message + recent conversation.
        """
        messages = []

        # Memory context as system prefix
        memory_ctx = self.build_prompt_context()
        if memory_ctx:
            messages.append({"role": "system", "content": memory_ctx})

        # Recent messages
        messages.extend(self.short_term.get_context_messages())

        return messages

    def record_feedback(self, vote: str, **kwargs):
        self.user_prefs.record_feedback(vote, **kwargs)

    def serialize(self) -> Dict:
        return {
            "user_id": self.user_id,
            "short_term": self.short_term.to_dict(),
            "rolling_summary": self.rolling_summary.to_dict(),
            "user_prefs": self.user_prefs.to_dict(),
        }

    @classmethod
    def deserialize(cls, data: Dict, settings=None) -> "MemoryEngine":
        engine = cls(user_id=data.get("user_id", ""), settings=settings)
        if "short_term" in data:
            engine.short_term = ShortTermMemory.from_dict(data["short_term"])
        if "rolling_summary" in data:
            engine.rolling_summary = RollingSummary.from_dict(data["rolling_summary"])
        if "user_prefs" in data:
            engine.user_prefs = UserPreferences.from_dict(data["user_prefs"])
        return engine

    def generate_summary_prompt(self) -> str:
        """Generate a prompt to summarize recent conversation for rolling memory."""
        recent = self.short_term.get_last_n(self.settings.ROLLING_SUMMARY_INTERVAL)
        if not recent:
            return ""
        conversation = "\n".join([f"{m['role']}: {m['content'][:300]}" for m in recent])
        return (
            "Summarize the following conversation in 2-3 sentences. "
            "Focus on key topics discussed, decisions made, and any preferences expressed. "
            "Be extremely concise.\n\n"
            f"{conversation}\n\n"
            "Summary:"
        )
