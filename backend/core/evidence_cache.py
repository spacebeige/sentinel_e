"""
Evidence Cache System — Sentinel-E Autonomous Reasoning Engine

Multi-level cache (L1: exact hash → Redis, L2: semantic ANN → FAISS,
L3: knowledge graph coverage) that gates all retrieval calls.

Integrates with:
  - backend/retrieval/cognitive_rag.py (CognitiveRAG)
  - backend/core/intent_hasher.py (IntentHash)
  - backend/storage/redis.py (RedisClient)
"""

import json
import time
import math
import logging
import hashlib
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("EvidenceCache")

# ============================================================
# CONFIGURATION
# ============================================================

SEMANTIC_CACHE_THRESHOLD = 0.87   # Cosine similarity for L2 hit
DEFAULT_TTL = 1800                # 30 min for factual queries
TEMPORAL_TTL = 300                # 5 min for time-sensitive queries
ANALYTICAL_TTL = 3600             # 1 hour for analytical queries
MAX_CACHE_ENTRIES = 1000          # LRU eviction limit
STALE_RATIO = 0.7                 # Entry downgraded to "aged" at 70% TTL


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class CachedChunk:
    """A cached evidence chunk."""
    content: str
    source_url: str
    source_domain: str
    reliability_score: float
    content_hash: str
    embedding: Optional[List[float]] = None  # Serializable form

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source_url": self.source_url,
            "source_domain": self.source_domain,
            "reliability_score": self.reliability_score,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CachedChunk":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EvidenceCacheEntry:
    """A complete cached evidence result."""
    cache_key: str                        # SHA3-256; also used as Redis key suffix
    query_canonical: str
    query_embedding: Optional[List[float]] = None  # Serializable
    topic_centroid: Optional[List[float]] = None
    chunks: List[CachedChunk] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    timestamp: float = 0.0               # Unix epoch
    ttl_seconds: int = DEFAULT_TTL
    confidence_score: float = 0.5
    freshness_class: str = "live"         # live | recent | aged | stale
    retrieval_latency_ms: int = 0
    hit_count: int = 0
    intent_type: str = "factual"
    citations_text: str = ""
    contradictions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_ttl(self) -> float:
        elapsed = time.time() - self.timestamp
        return max(0.0, self.ttl_seconds - elapsed)

    @property
    def is_expired(self) -> bool:
        return self.remaining_ttl <= 0

    def update_freshness(self):
        if self.ttl_seconds <= 0:
            self.freshness_class = "stale"
            return
        elapsed = time.time() - self.timestamp
        ratio = elapsed / self.ttl_seconds
        if ratio < 0.3:
            self.freshness_class = "live"
        elif ratio < 0.7:
            self.freshness_class = "recent"
        elif ratio < 1.0:
            self.freshness_class = "aged"
        else:
            self.freshness_class = "stale"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "query_canonical": self.query_canonical,
            "query_embedding": self.query_embedding,
            "topic_centroid": self.topic_centroid,
            "chunks": [c.to_dict() for c in self.chunks],
            "source_urls": self.source_urls,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "confidence_score": self.confidence_score,
            "freshness_class": self.freshness_class,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "hit_count": self.hit_count,
            "intent_type": self.intent_type,
            "citations_text": self.citations_text,
            "contradictions": self.contradictions,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EvidenceCacheEntry":
        entry = cls(
            cache_key=d.get("cache_key", ""),
            query_canonical=d.get("query_canonical", ""),
            query_embedding=d.get("query_embedding"),
            topic_centroid=d.get("topic_centroid"),
            source_urls=d.get("source_urls", []),
            timestamp=d.get("timestamp", 0.0),
            ttl_seconds=d.get("ttl_seconds", DEFAULT_TTL),
            confidence_score=d.get("confidence_score", 0.5),
            freshness_class=d.get("freshness_class", "stale"),
            retrieval_latency_ms=d.get("retrieval_latency_ms", 0),
            hit_count=d.get("hit_count", 0),
            intent_type=d.get("intent_type", "factual"),
            citations_text=d.get("citations_text", ""),
            contradictions=d.get("contradictions", []),
        )
        entry.chunks = [CachedChunk.from_dict(c) for c in d.get("chunks", [])]
        return entry


# ============================================================
# COSINE SIMILARITY
# ============================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# EVIDENCE CACHE
# ============================================================

class EvidenceCache:
    """
    Multi-level evidence cache.

    L1: Exact hash lookup (Redis or in-memory dict).
    L2: Semantic similarity scan (in-memory FAISS-like ANN via brute force
        on small cache; upgradeable to real FAISS index).
    L3: Knowledge graph coverage check (delegated to KG engine).
    """

    def __init__(self, redis_client=None, max_entries: int = MAX_CACHE_ENTRIES):
        self._redis = redis_client
        self._max_entries = max_entries
        # In-memory L1 + L2 store (fallback if Redis unavailable)
        self._local_cache: Dict[str, EvidenceCacheEntry] = {}
        self._embedding_index: List[Tuple[str, np.ndarray]] = []  # (cache_key, embedding)
        # Metrics
        self._metrics = {"l1_hit": 0, "l2_hit": 0, "l3_hit": 0, "miss": 0}

    @property
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    # ----------------------------------------------------------
    # CACHE LOOKUP
    # ----------------------------------------------------------

    async def check(
        self,
        cache_key: str,
        query_embedding: Optional[np.ndarray] = None,
        temporal_flag: bool = False,
        intent_type: str = "factual",
    ) -> Optional[EvidenceCacheEntry]:
        """
        Multi-level cache check.

        Returns cached EvidenceCacheEntry if hit, else None.
        """
        # --- L1: Exact hash ---
        entry = await self._get_l1(cache_key)
        if entry is not None:
            entry.update_freshness()
            if not entry.is_expired:
                if temporal_flag and entry.freshness_class in ("aged", "stale"):
                    pass  # Temporal query needs fresh data; fall through
                else:
                    entry.hit_count += 1
                    await self._put_l1(cache_key, entry)
                    self._metrics["l1_hit"] += 1
                    logger.debug(f"L1 cache hit: {cache_key[:16]}...")
                    return entry

        # --- L2: Semantic similarity ---
        if query_embedding is not None and len(self._embedding_index) > 0:
            best_entry, best_sim = self._scan_l2(query_embedding)
            if best_entry is not None and best_sim >= SEMANTIC_CACHE_THRESHOLD:
                best_entry.update_freshness()
                if not best_entry.is_expired:
                    if temporal_flag and best_entry.freshness_class in ("aged", "stale"):
                        pass  # Need fresh
                    else:
                        best_entry.hit_count += 1
                        # Promote to L1 under current key
                        await self._put_l1(cache_key, best_entry)
                        self._metrics["l2_hit"] += 1
                        logger.debug(f"L2 cache hit (sim={best_sim:.3f}): {best_entry.cache_key[:16]}...")
                        return best_entry

        # --- Miss ---
        self._metrics["miss"] += 1
        return None

    # ----------------------------------------------------------
    # CACHE STORE
    # ----------------------------------------------------------

    async def store(self, entry: EvidenceCacheEntry):
        """Store a new evidence cache entry after live retrieval."""
        entry.timestamp = time.time()
        entry.update_freshness()

        # TTL by intent type
        if entry.intent_type in ("temporal",):
            entry.ttl_seconds = TEMPORAL_TTL
        elif entry.intent_type in ("analytical",):
            entry.ttl_seconds = ANALYTICAL_TTL
        elif entry.intent_type in ("creative", "conversational"):
            return  # Don't cache creative/conversational
        else:
            entry.ttl_seconds = DEFAULT_TTL

        # L1 store
        await self._put_l1(entry.cache_key, entry)

        # L2 index
        if entry.query_embedding is not None:
            emb = np.array(entry.query_embedding, dtype=np.float32)
            self._embedding_index.append((entry.cache_key, emb))
            # Evict oldest if over limit
            if len(self._embedding_index) > self._max_entries:
                self._embedding_index = self._embedding_index[-self._max_entries:]

        logger.debug(f"Cached evidence: {entry.cache_key[:16]}... TTL={entry.ttl_seconds}s")

    # ----------------------------------------------------------
    # INVALIDATION
    # ----------------------------------------------------------

    async def invalidate(self, cache_key: str):
        """Invalidate a specific cache entry."""
        self._local_cache.pop(cache_key, None)
        self._embedding_index = [(k, e) for k, e in self._embedding_index if k != cache_key]
        if self._redis:
            try:
                await self._redis.client.delete(f"evidence:{cache_key}")
            except Exception:
                pass

    async def invalidate_stale(self):
        """Purge all stale entries (hard eviction at 2×TTL)."""
        now = time.time()
        to_remove = []
        for key, entry in self._local_cache.items():
            elapsed = now - entry.timestamp
            if elapsed > entry.ttl_seconds * 2:
                to_remove.append(key)
        for key in to_remove:
            await self.invalidate(key)
        if to_remove:
            logger.info(f"Evicted {len(to_remove)} stale cache entries")

    # ----------------------------------------------------------
    # INTERNAL L1 get/put
    # ----------------------------------------------------------

    async def _get_l1(self, cache_key: str) -> Optional[EvidenceCacheEntry]:
        # Try local first
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]
        # Try Redis
        if self._redis and self._redis.connected:
            try:
                raw = await self._redis.get(f"evidence:{cache_key}")
                if raw:
                    entry = EvidenceCacheEntry.from_dict(json.loads(raw))
                    self._local_cache[cache_key] = entry
                    return entry
            except Exception as e:
                logger.debug(f"Redis L1 get failed: {e}")
        return None

    async def _put_l1(self, cache_key: str, entry: EvidenceCacheEntry):
        self._local_cache[cache_key] = entry
        # Enforce max size (LRU approx: just trim oldest)
        if len(self._local_cache) > self._max_entries:
            oldest_key = next(iter(self._local_cache))
            del self._local_cache[oldest_key]
        # Persist to Redis
        if self._redis and self._redis.connected:
            try:
                ttl = max(1, int(entry.remaining_ttl))
                await self._redis.set(
                    f"evidence:{cache_key}",
                    json.dumps(entry.to_dict()),
                    ttl=ttl,
                )
            except Exception as e:
                logger.debug(f"Redis L1 put failed: {e}")

    # ----------------------------------------------------------
    # L2 brute-force scan (small cache; replace with FAISS for scale)
    # ----------------------------------------------------------

    def _scan_l2(self, query_embedding: np.ndarray) -> Tuple[Optional[EvidenceCacheEntry], float]:
        best_sim = 0.0
        best_key = None
        for key, emb in self._embedding_index:
            sim = _cosine_similarity(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_key = key
        if best_key and best_key in self._local_cache:
            return self._local_cache[best_key], best_sim
        return None, 0.0
