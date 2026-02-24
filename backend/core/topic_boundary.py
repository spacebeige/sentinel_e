"""
Topic Boundary Controller — Sentinel-E Autonomous Reasoning Engine

Detects topic shifts, anchors follow-ups, and applies context decay.
Stabilizes conversational context and prevents drift.

Integrates with:
  - backend/core/knowledge_memory.py (SessionMemoryTier)
  - backend/core/intent_hasher.py (IntentHash)
  - backend/core/session_intelligence.py (domain inference)
"""

import math
import logging
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("TopicBoundary")


# ============================================================
# CONFIGURATION
# ============================================================

TOPIC_SHIFT_THRESHOLD = 0.40      # cosine < this → hard topic shift
TOPIC_EXPAND_THRESHOLD = 0.58     # cosine between shift and expand → soft drift
FOLLOWUP_STRONG_THRESHOLD = 0.85  # High similarity → definite follow-up
FOLLOWUP_PROBABLE_THRESHOLD = 0.70  # Moderate similarity → probable follow-up
DECAY_LAMBDA = 0.15               # Exponential decay rate (per hour)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# RESULT STRUCTURES
# ============================================================

@dataclass
class TopicBoundaryResult:
    """Result of topic boundary detection."""
    is_shift: bool = False
    drift_score: float = 0.0
    action: str = "continue"  # continue | expand_cluster | archive_and_reset
    previous_cluster_id: str = ""
    similarity_global: float = 0.0
    similarity_local: float = 0.0


@dataclass
class FollowUpAnchor:
    """Result of follow-up anchoring analysis."""
    anchor_type: str = "initial"  # initial | linked | ambiguous | new_topic
    linked_intent_hash: Optional[str] = None
    similarity: float = 0.0
    reuse_evidence: bool = False
    clarification_needed: bool = False
    clarification_prompt: str = ""


# ============================================================
# TOPIC BOUNDARY DETECTOR
# ============================================================

class TopicBoundaryDetector:
    """
    Detects when the user shifts topic, requiring memory cluster
    archival and reset.
    
    Uses dual-similarity scoring:
      - Global: cosine(query, topic_centroid) — broad topic match
      - Local: cosine(query, recent_user_centroid) — recent drift
    
    Combined: drift_score = 1 - (0.6 * global + 0.4 * local)
    """

    def detect(
        self,
        query_embedding: Optional[np.ndarray],
        topic_embedding: Optional[np.ndarray],
        recent_user_embeddings: Optional[List[np.ndarray]] = None,
        current_cluster_id: str = "",
    ) -> TopicBoundaryResult:
        """
        Check if the current query represents a topic shift.
        """
        if query_embedding is None or topic_embedding is None:
            return TopicBoundaryResult(action="continue")

        # Global similarity
        global_sim = _cosine_similarity(query_embedding, topic_embedding)

        # Local similarity (last 5 user messages)
        local_sim = global_sim  # Default to global if no recent messages
        if recent_user_embeddings and len(recent_user_embeddings) > 0:
            recent_centroid = np.mean(recent_user_embeddings, axis=0).astype(np.float32)
            local_sim = _cosine_similarity(query_embedding, recent_centroid)

        # Combined drift score
        combined_sim = 0.6 * global_sim + 0.4 * local_sim
        drift_score = 1.0 - combined_sim

        # Decision
        if combined_sim < TOPIC_SHIFT_THRESHOLD:
            action = "archive_and_reset"
            is_shift = True
        elif combined_sim < TOPIC_EXPAND_THRESHOLD:
            action = "expand_cluster"
            is_shift = False
        else:
            action = "continue"
            is_shift = False

        result = TopicBoundaryResult(
            is_shift=is_shift,
            drift_score=drift_score,
            action=action,
            previous_cluster_id=current_cluster_id,
            similarity_global=round(global_sim, 4),
            similarity_local=round(local_sim, 4),
        )

        if is_shift:
            logger.info(
                f"Topic shift detected: drift={drift_score:.3f}, "
                f"global_sim={global_sim:.3f}, local_sim={local_sim:.3f}"
            )

        return result


# ============================================================
# FOLLOW-UP ANCHORING
# ============================================================

class FollowUpAnchorEngine:
    """
    Determines whether a new query is a follow-up to a previous intent,
    and whether to reuse cached evidence.
    """

    def anchor(
        self,
        current_embedding: Optional[np.ndarray],
        current_hash: str,
        intent_history: List[Dict[str, Any]],
        current_cluster_id: str = "",
    ) -> FollowUpAnchor:
        """
        Args:
            current_embedding: Embedding of current query.
            current_hash: Exact hash of current intent.
            intent_history: List of previous intents with fields:
                {"exact_hash", "embedding", "session_id", "canonical"}
            current_cluster_id: Active topic cluster ID.
        """
        if not intent_history or current_embedding is None:
            return FollowUpAnchor(anchor_type="initial")

        last = intent_history[-1]
        last_embedding = last.get("embedding")

        if last_embedding is None:
            return FollowUpAnchor(anchor_type="initial")

        if isinstance(last_embedding, list):
            last_embedding = np.array(last_embedding, dtype=np.float32)

        sim = _cosine_similarity(current_embedding, last_embedding)

        if sim > FOLLOWUP_STRONG_THRESHOLD:
            return FollowUpAnchor(
                anchor_type="linked",
                linked_intent_hash=last.get("exact_hash"),
                similarity=round(sim, 4),
                reuse_evidence=True,
            )
        elif sim > FOLLOWUP_PROBABLE_THRESHOLD:
            # Probable follow-up — check scope
            last_session = last.get("session_id", "")
            scope_match = (current_cluster_id == last_session) if last_session else True

            if scope_match or sim > 0.80:
                return FollowUpAnchor(
                    anchor_type="linked",
                    linked_intent_hash=last.get("exact_hash"),
                    similarity=round(sim, 4),
                    reuse_evidence=True,
                )
            else:
                last_canonical = last.get("canonical", "previous topic")
                return FollowUpAnchor(
                    anchor_type="ambiguous",
                    linked_intent_hash=last.get("exact_hash"),
                    similarity=round(sim, 4),
                    reuse_evidence=False,
                    clarification_needed=True,
                    clarification_prompt=(
                        f"Are you continuing the discussion about '{last_canonical}' "
                        f"or starting a new topic?"
                    ),
                )
        else:
            return FollowUpAnchor(anchor_type="new_topic")


# ============================================================
# CONTEXT DECAY ENGINE
# ============================================================

class ContextDecayEngine:
    """
    Applies exponential decay with topic relevance modulation.
    Called periodically to update message weights in SessionMemoryTier.
    """

    def __init__(self, decay_lambda: float = DECAY_LAMBDA, prune_threshold: float = 0.05):
        self.decay_lambda = decay_lambda
        self.prune_threshold = prune_threshold

    def apply(
        self,
        messages: List[Dict[str, Any]],
        topic_embedding: Optional[np.ndarray],
        current_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply decay to a list of message dicts (must have 'embedding', 'timestamp', 'role').
        Returns updated list with 'relevance_weight' set. Messages below prune threshold are removed.
        
        This is a standalone version for use outside SessionMemoryTier.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        surviving = []
        for msg in messages:
            ts = msg.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = current_time

            age_hours = (current_time - ts).total_seconds() / 3600 if ts else 0
            time_decay = math.exp(-self.decay_lambda * age_hours)

            embedding = msg.get("embedding")
            if embedding is not None and topic_embedding is not None:
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                topic_rel = _cosine_similarity(embedding, topic_embedding)
            else:
                topic_rel = 0.5

            role = msg.get("role", "user")
            role_boost = 1.5 if role == "system" else 1.0
            compressed = msg.get("compressed", False)
            if compressed:
                role_boost *= 0.8

            weight = max(0.0, min(1.0, time_decay * topic_rel * role_boost))
            msg["relevance_weight"] = weight

            if weight >= self.prune_threshold:
                surviving.append(msg)

        return surviving
