"""
============================================================
Multi-Layer Caching Engine — Sentinel-E v3
============================================================

Four-layer caching system for reasoning pipeline results:

  Layer 1 — Query Cache:    Full MCO/debate responses (TTL: 10 min)
  Layer 2 — Search Cache:   Evidence search results (TTL: 30 min)
  Layer 3 — Model Cache:    Individual model responses (TTL: 15 min)
  Layer 4 — Graph Cache:    Knowledge graph fragments (TTL: 60 min)

Keys:  SHA-256(query + mode + sub_mode)
Store: Redis (real or InMemoryRedisStub fallback)

Usage:
    from core.cache_engine import reasoning_cache
    hit = await reasoning_cache.get_query("some query", "debate")
    if hit: return hit
    ...
    await reasoning_cache.set_query("some query", "debate", result)
============================================================
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("CacheEngine")

# TTLs in seconds
TTL_QUERY = 600       # 10 min — full pipeline results
TTL_SEARCH = 1800     # 30 min — evidence / web search
TTL_MODEL = 900       # 15 min — individual model calls
TTL_GRAPH = 3600      # 60 min — graph fragments


def _cache_key(prefix: str, query: str, discriminator: str = "") -> str:
    raw = f"{query.strip().lower()}:{discriminator}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return f"sentinel:cache:{prefix}:{digest}"


class ReasoningCache:
    """Multi-layer caching backed by Redis (or InMemoryRedisStub)."""

    def __init__(self):
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            from database.connection import redis_client
            self._redis = redis_client
        return self._redis

    # ── helpers ──────────────────────────────────────────────

    async def _get(self, key: str) -> Optional[Dict]:
        try:
            raw = await self._get_redis().get(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"Cache miss (error): {e}")
        return None

    async def _set(self, key: str, value: Any, ttl: int):
        try:
            await self._get_redis().setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    async def _invalidate(self, key: str):
        try:
            await self._get_redis().delete(key)
        except Exception:
            pass

    # ── Layer 1: Query Cache ─────────────────────────────────

    async def get_query(self, query: str, mode: str, sub_mode: str = "") -> Optional[Dict]:
        key = _cache_key("q", query, f"{mode}:{sub_mode}")
        hit = await self._get(key)
        if hit:
            logger.info(f"Query cache HIT [{mode}/{sub_mode}]")
        return hit

    async def set_query(self, query: str, mode: str, result: Dict, sub_mode: str = ""):
        key = _cache_key("q", query, f"{mode}:{sub_mode}")
        await self._set(key, result, TTL_QUERY)

    # ── Layer 2: Search Cache ────────────────────────────────

    async def get_search(self, query: str) -> Optional[Dict]:
        key = _cache_key("s", query)
        hit = await self._get(key)
        if hit:
            logger.info("Search cache HIT")
        return hit

    async def set_search(self, query: str, result: Dict):
        key = _cache_key("s", query)
        await self._set(key, result, TTL_SEARCH)

    # ── Layer 3: Model Response Cache ────────────────────────

    async def get_model(self, query: str, model_name: str) -> Optional[Dict]:
        key = _cache_key("m", query, model_name)
        return await self._get(key)

    async def set_model(self, query: str, model_name: str, result: Dict):
        key = _cache_key("m", query, model_name)
        await self._set(key, result, TTL_MODEL)

    # ── Layer 4: Graph Cache ─────────────────────────────────

    async def get_graph(self, query: str, graph_type: str = "reasoning") -> Optional[Dict]:
        key = _cache_key("g", query, graph_type)
        hit = await self._get(key)
        if hit:
            logger.info("Graph cache HIT")
        return hit

    async def set_graph(self, query: str, graph_type: str, result: Dict):
        key = _cache_key("g", query, graph_type)
        await self._set(key, result, TTL_GRAPH)

    # ── Bulk invalidation ────────────────────────────────────

    async def invalidate_query(self, query: str, mode: str, sub_mode: str = ""):
        key = _cache_key("q", query, f"{mode}:{sub_mode}")
        await self._invalidate(key)

    async def flush_all(self):
        """Flush all sentinel cache keys. Use sparingly."""
        try:
            keys = await self._get_redis().keys("sentinel:cache:*")
            for k in keys:
                await self._get_redis().delete(k)
            logger.info(f"Flushed {len(keys)} cache keys")
        except Exception as e:
            logger.warning(f"Cache flush failed: {e}")


# ── Singleton ────────────────────────────────────────────────
reasoning_cache = ReasoningCache()
