"""
============================================================
API 3 — Session & Persistence Engine
============================================================
Maintains structured session state.

Responsibilities:
  ✓ Topic centroid tracking
  ✓ Drift detection
  ✓ Goal persistence
  ✓ Memory ranking
  ✓ Background daemon scheduling
  ✓ Behavioral log storage
  ✗ No content generation

Session state is stored in Redis (hot) and PostgreSQL (cold).
============================================================
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from metacognitive.schemas import (
    SessionState,
    StructuredGoal,
    MemoryBlock,
    BehavioralRecord,
    UnresolvedQuestion,
    OperatingMode,
)
from metacognitive.embedding import (
    embed_text,
    cosine_similarity,
    compute_centroid,
    centroid_variance,
    volatility_score as compute_volatility,
    drift_score as compute_drift,
)

logger = logging.getLogger("MCO-SessionEngine")


# ============================================================
# Configuration
# ============================================================

MAX_MEMORY_BLOCKS = 100
MAX_BEHAVIORAL_RECORDS = 200
DRIFT_THRESHOLD = 0.5
MEMORY_RELEVANCE_DECAY = 0.95  # Per-turn decay factor


class SessionPersistenceEngine:
    """
    API 3 — Session & Persistence Engine.

    Enforces strict separation:
      ✓ State management only
      ✗ No content generation
      ✗ No model invocation
      ✗ No retrieval logic
    """

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._redis = None  # Injected externally

    def set_redis(self, redis_client):
        """Inject Redis client for hot storage."""
        self._redis = redis_client

    # ── Session Lifecycle ────────────────────────────────────

    def create_session(
        self,
        session_id: Optional[str] = None,
        mode: OperatingMode = OperatingMode.STANDARD,
    ) -> SessionState:
        """Create a new session."""
        state = SessionState(mode=mode)
        if session_id:
            state.session_id = session_id
        self._sessions[state.session_id] = state
        logger.info(f"Session created: {state.session_id} (mode={mode.value})")
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session state."""
        return self._sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        mode: OperatingMode = OperatingMode.STANDARD,
    ) -> SessionState:
        """Get existing or create new session."""
        existing = self.get_session(session_id)
        if existing:
            return existing
        return self.create_session(session_id=session_id, mode=mode)

    # ── Topic Centroid ───────────────────────────────────────

    def update_topic_centroid(
        self,
        session_id: str,
        new_embedding: List[float],
    ) -> float:
        """
        Update topic centroid with new query/output embedding.
        Returns updated drift_score.
        """
        session = self.get_session(session_id)
        if not session:
            return 0.0

        # Collect all memory block embeddings + new embedding
        embeddings = [
            b.embedding for b in session.memory_blocks
            if b.embedding
        ]
        embeddings.append(new_embedding)

        # Recompute centroid
        session.topic_centroid_embedding = compute_centroid(embeddings)

        # Compute drift
        session.drift_score = compute_drift(
            new_embedding, session.topic_centroid_embedding
        )

        session.updated_at = datetime.now(timezone.utc).isoformat()
        return session.drift_score

    def compute_volatility(
        self,
        session_id: str,
        query_embedding: List[float],
    ) -> float:
        """Compute volatility of a query relative to session centroid."""
        session = self.get_session(session_id)
        if not session:
            return 1.0  # Maximum volatility for unknown session
        vol = compute_volatility(query_embedding, session.topic_centroid_embedding)
        session.volatility_score = vol
        return vol

    # ── Memory Management ────────────────────────────────────

    def add_memory_block(
        self,
        session_id: str,
        content: str,
        source: str = "user",
        embedding: Optional[List[float]] = None,
    ) -> Optional[MemoryBlock]:
        """Add a memory block to the session."""
        session = self.get_session(session_id)
        if not session:
            return None

        if not embedding:
            embedding = embed_text(content)

        block = MemoryBlock(
            content=content,
            embedding=embedding,
            source=source,
            turn_created=session.turn_count,
        )

        # Compute relevance to centroid
        if session.topic_centroid_embedding:
            block.relevance_score = cosine_similarity(
                embedding, session.topic_centroid_embedding
            )

        session.memory_blocks.append(block)

        # Trim if over limit
        if len(session.memory_blocks) > MAX_MEMORY_BLOCKS:
            self._compress_memory(session)

        session.updated_at = datetime.now(timezone.utc).isoformat()
        return block

    def rank_memory_blocks(
        self,
        session_id: str,
        query_embedding: Optional[List[float]] = None,
    ) -> List[MemoryBlock]:
        """
        Rank memory blocks by relevance to query and centroid.
        Apply temporal decay.
        """
        session = self.get_session(session_id)
        if not session:
            return []

        reference = query_embedding or session.topic_centroid_embedding
        if not reference:
            return session.memory_blocks

        for block in session.memory_blocks:
            if block.embedding:
                sim = cosine_similarity(block.embedding, reference)
                # Apply temporal decay
                age = session.turn_count - block.turn_created
                decay = MEMORY_RELEVANCE_DECAY ** age
                block.relevance_score = sim * decay

        # Sort by relevance (descending)
        session.memory_blocks.sort(
            key=lambda b: b.relevance_score, reverse=True
        )
        return session.memory_blocks

    def suppress_low_signal_blocks(
        self,
        session_id: str,
        threshold: float = 0.15,
    ) -> int:
        """
        Suppress (remove) memory blocks below relevance threshold.
        Returns count of suppressed blocks.
        """
        session = self.get_session(session_id)
        if not session:
            return 0

        before = len(session.memory_blocks)
        session.memory_blocks = [
            b for b in session.memory_blocks
            if b.relevance_score >= threshold
        ]
        suppressed = before - len(session.memory_blocks)
        if suppressed > 0:
            logger.info(f"Suppressed {suppressed} low-signal memory blocks")
        return suppressed

    def _compress_memory(self, session: SessionState):
        """Compress memory by removing lowest-relevance blocks."""
        session.memory_blocks.sort(
            key=lambda b: b.relevance_score, reverse=True
        )
        session.memory_blocks = session.memory_blocks[:MAX_MEMORY_BLOCKS]

    # ── Goal Management ──────────────────────────────────────

    def add_goal(
        self,
        session_id: str,
        description: str,
    ) -> Optional[StructuredGoal]:
        """Add a structured goal to the session."""
        session = self.get_session(session_id)
        if not session:
            return None

        goal = StructuredGoal(description=description)
        session.structured_goals.append(goal)
        session.updated_at = datetime.now(timezone.utc).isoformat()
        return goal

    def resolve_goal(
        self,
        session_id: str,
        goal_id: str,
    ) -> bool:
        """Mark a goal as resolved."""
        session = self.get_session(session_id)
        if not session:
            return False

        for goal in session.structured_goals:
            if goal.id == goal_id:
                goal.status = "resolved"
                goal.resolved_at = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def get_active_goals(self, session_id: str) -> List[StructuredGoal]:
        """Get all active (unresolved) goals."""
        session = self.get_session(session_id)
        if not session:
            return []
        return [g for g in session.structured_goals if g.status == "active"]

    # ── Behavioral Logging ───────────────────────────────────

    def log_behavior(
        self,
        session_id: str,
        record: BehavioralRecord,
    ):
        """Log a behavioral observation."""
        session = self.get_session(session_id)
        if not session:
            return

        session.behavioral_history.append(record)

        # Trim history
        if len(session.behavioral_history) > MAX_BEHAVIORAL_RECORDS:
            session.behavioral_history = session.behavioral_history[-MAX_BEHAVIORAL_RECORDS:]

    # ── Unresolved Questions ─────────────────────────────────

    def add_unresolved_question(
        self,
        session_id: str,
        question: str,
        priority: float = 0.5,
    ) -> Optional[UnresolvedQuestion]:
        """Add an unresolved question for daemon refinement."""
        session = self.get_session(session_id)
        if not session:
            return None

        uq = UnresolvedQuestion(question=question, priority=priority)
        session.unresolved_questions.append(uq)
        return uq

    def get_unresolved_questions(self, session_id: str) -> List[UnresolvedQuestion]:
        """Get all unresolved questions."""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.unresolved_questions

    def resolve_question(self, session_id: str, question_id: str) -> bool:
        """Remove a resolved question."""
        session = self.get_session(session_id)
        if not session:
            return False
        session.unresolved_questions = [
            q for q in session.unresolved_questions
            if q.id != question_id
        ]
        return True

    # ── Turn Management ──────────────────────────────────────

    def increment_turn(self, session_id: str) -> int:
        """Increment turn counter and refinement cycles."""
        session = self.get_session(session_id)
        if not session:
            return 0
        session.turn_count += 1
        session.refinement_cycles += 1
        session.updated_at = datetime.now(timezone.utc).isoformat()
        return session.turn_count

    # ── Session Summary ──────────────────────────────────────

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Build a compact session summary for model context injection."""
        session = self.get_session(session_id)
        if not session:
            return {}

        active_goals = [g.description for g in session.structured_goals if g.status == "active"]
        top_memories = sorted(
            session.memory_blocks,
            key=lambda b: b.relevance_score,
            reverse=True,
        )[:5]

        return {
            "session_id": session.session_id,
            "turn_count": session.turn_count,
            "drift_score": round(session.drift_score, 4),
            "volatility_score": round(session.volatility_score, 4),
            "active_goals": active_goals,
            "recent_context": [m.content[:200] for m in top_memories],
            "refinement_cycles": session.refinement_cycles,
            "mode": session.mode.value,
        }

    # ── Serialization (Redis) ────────────────────────────────

    def serialize_session(self, session_id: str) -> str:
        """Serialize session for Redis storage."""
        session = self.get_session(session_id)
        if not session:
            return "{}"
        return session.model_dump_json()

    def restore_session(self, session_data: str) -> Optional[SessionState]:
        """Restore session from serialized data."""
        try:
            session = SessionState.model_validate_json(session_data)
            self._sessions[session.session_id] = session
            return session
        except Exception as e:
            logger.error(f"Session restore failed: {e}")
            return None

    async def persist_to_redis(self, session_id: str, ttl: int = 7200):
        """Persist session to Redis with TTL."""
        if not self._redis:
            return
        try:
            data = self.serialize_session(session_id)
            await self._redis.setex(
                f"mco:session:{session_id}",
                ttl,
                data,
            )
        except Exception as e:
            logger.warning(f"Redis persist failed: {e}")

    async def restore_from_redis(self, session_id: str) -> Optional[SessionState]:
        """Restore session from Redis."""
        if not self._redis:
            return None
        try:
            data = await self._redis.get(f"mco:session:{session_id}")
            if data:
                return self.restore_session(data)
        except Exception as e:
            logger.warning(f"Redis restore failed: {e}")
        return None

    # ── Centroid Stability Check ─────────────────────────────

    def is_centroid_stable(
        self,
        session_id: str,
        epsilon: float = 0.05,
    ) -> bool:
        """
        Check if topic centroid has stabilized.
        Used in recursive context stabilization loop.
        """
        session = self.get_session(session_id)
        if not session or not session.topic_centroid_embedding:
            return False

        embeddings = [
            b.embedding for b in session.memory_blocks
            if b.embedding
        ]
        if not embeddings:
            return True  # Nothing to be unstable about

        variance = centroid_variance(
            embeddings, session.topic_centroid_embedding
        )
        return variance < epsilon
