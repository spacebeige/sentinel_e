"""
Intent Hashing Layer — Sentinel-E Autonomous Reasoning Engine

Converts user queries into deterministic, comparable structural
representations for exact-match and fuzzy-match cache lookups.

Integrates with: backend/retrieval/cognitive_rag.py (QueryClassifier)
"""

import hashlib
import re
import logging
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("IntentHasher")

# ============================================================
# STOPWORDS (minimal, covers English)
# ============================================================

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "very", "just", "also", "than", "then", "too", "only",
})


@dataclass
class IntentHash:
    """Structural representation of a user query intent."""
    exact_hash: str                 # SHA3-256 of canonical form
    embedding: Optional[np.ndarray] = None  # 384-dim float32
    canonical: str = ""             # Normalized token string
    intent_type: str = "conversational"
    retrieval_p: float = 0.0
    temporal_flag: bool = False
    session_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "exact_hash": self.exact_hash,
            "canonical": self.canonical,
            "intent_type": self.intent_type,
            "retrieval_p": self.retrieval_p,
            "temporal_flag": self.temporal_flag,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }


def _simple_lemmatize(token: str) -> str:
    """
    Lightweight rule-based lemmatization.
    Not as accurate as spaCy but zero-dependency and fast.
    """
    if len(token) <= 3:
        return token
    # Common suffixes
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        if base and base[-1] == base[-2] if len(base) > 1 else False:
            return base[:-1]  # "running" → "run"
        return base
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("ly") and len(token) > 4:
        return token[:-2]
    return token


class IntentHasher:
    """
    Converts user queries into IntentHash objects.
    Uses QueryClassifier from cognitive_rag for intent classification.
    """

    def __init__(self, embed_fn=None):
        """
        Args:
            embed_fn: Optional function(text) -> np.ndarray (384-dim).
                      If None, embeddings are skipped (exact-hash only).
        """
        self._embed_fn = embed_fn
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            try:
                from backend.retrieval.cognitive_rag import QueryClassifier
                self._classifier = QueryClassifier()
            except ImportError:
                logger.warning("QueryClassifier not available; using defaults")
                self._classifier = None
        return self._classifier

    def hash_intent(self, query: str, session_id: str = "") -> IntentHash:
        """
        Convert query into a deterministic IntentHash.

        Steps:
          1. Normalize (lowercase, strip)
          2. Tokenize
          3. Remove stopwords
          4. Lemmatize
          5. Sort canonically
          6. SHA3-256 hash
          7. Optionally embed
          8. Classify intent
        """
        if not query or not query.strip():
            return IntentHash(
                exact_hash=hashlib.sha3_256(b"").hexdigest(),
                session_id=session_id,
            )

        # --- Normalize ---
        normalized = query.lower().strip()
        # Remove punctuation except hyphens
        normalized = re.sub(r"[^\w\s\-]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # --- Tokenize ---
        tokens = normalized.split()

        # --- Remove stopwords ---
        filtered = [t for t in tokens if t not in STOPWORDS]
        if not filtered:
            filtered = tokens  # Keep original if all stopwords

        # --- Lemmatize ---
        lemmatized = [_simple_lemmatize(t) for t in filtered]

        # --- Canonical form (sorted) ---
        canonical = " ".join(sorted(lemmatized))

        # --- Hash ---
        exact_hash = hashlib.sha3_256(canonical.encode("utf-8")).hexdigest()

        # --- Embedding (optional) ---
        embedding = None
        if self._embed_fn is not None:
            try:
                embedding = self._embed_fn(query)
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")

        # --- Classify ---
        intent_type = "conversational"
        retrieval_p = 0.0
        temporal_flag = False
        classifier = self._get_classifier()
        if classifier:
            try:
                classification = classifier.classify(query)
                intent_type = classification.primary_intent
                retrieval_p = classification.retrieval_probability
                temporal_flag = classification.temporal_sensitivity > 0.5
            except Exception as e:
                logger.warning(f"Classification failed: {e}")

        return IntentHash(
            exact_hash=exact_hash,
            embedding=embedding,
            canonical=canonical,
            intent_type=intent_type,
            retrieval_p=retrieval_p,
            temporal_flag=temporal_flag,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
        )
