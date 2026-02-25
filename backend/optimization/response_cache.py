"""
============================================================
Sentinel-E — 3-Tier Response Cache
============================================================
Cost-reduction caching without heavy embedding dependencies.

Tier 1: Exact Match       — O(1) hash lookup, full response reuse
Tier 2: Lexical Similarity — Jaccard on word n-grams, threshold-gated
Tier 3: Semantic Approx    — Intent + entity + question-type hashing

Storage: Redis preferred, bounded in-memory dict fallback.
Memory safety: No FAISS, no SentenceTransformer, no torch.
============================================================
"""

import re
import time
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger("ResponseCache")


# ============================================================
# BOUNDED LRU CACHE (in-memory, Redis fallback)
# ============================================================

class BoundedLRUCache:
    """
    Thread-safe bounded LRU cache using OrderedDict.
    Evicts least-recently-used entries when capacity exceeded.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value by key. Returns None if missing or expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        value, expires_at = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value with TTL. Evicts LRU if over capacity."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl

        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, expires_at)

        # Evict oldest entries
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def delete(self, key: str):
        """Remove a key."""
        self._cache.pop(key, None)

    def clear(self):
        """Clear all entries."""
        self._cache.clear()

    @property
    def stats(self) -> Dict[str, int]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_query(text: str) -> str:
    """Normalize query text for cache matching."""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)      # collapse whitespace
    return text


def compute_hash(text: str) -> str:
    """Deterministic hash of normalized text."""
    return hashlib.sha256(normalize_query(text).encode()).hexdigest()[:32]


def extract_word_ngrams(text: str, n: int = 2) -> Set[str]:
    """Extract word n-grams for similarity comparison."""
    words = normalize_query(text).split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ============================================================
# TIER 1: EXACT MATCH CACHE
# ============================================================

class ExactMatchCache:
    """
    O(1) hash-based exact match cache.
    Stores full structured responses keyed by normalized query hash.
    """

    def __init__(self, max_size: int = 500, ttl: int = 1800):
        self._store = BoundedLRUCache(max_size=max_size, default_ttl=ttl)

    def lookup(self, query: str, mode: str = "standard") -> Optional[Dict[str, Any]]:
        """Check for exact cache match."""
        key = f"exact:{mode}:{compute_hash(query)}"
        result = self._store.get(key)
        if result:
            logger.debug(f"Tier-1 cache HIT: {key[:24]}...")
        return result

    def store(self, query: str, mode: str, response: Dict[str, Any], ttl: Optional[int] = None):
        """Store response in exact cache."""
        key = f"exact:{mode}:{compute_hash(query)}"
        entry = {
            **response,
            "_cache_meta": {
                "tier": 1,
                "cached_at": time.time(),
                "query_hash": key,
            },
        }
        self._store.set(key, entry, ttl)

    @property
    def stats(self) -> Dict:
        return {"tier": 1, "type": "exact_match", **self._store.stats}


# ============================================================
# TIER 2: LEXICAL SIMILARITY CACHE
# ============================================================

class LexicalSimilarityCache:
    """
    Jaccard similarity on word bigrams.
    If similarity > threshold, reuse cached response.
    No heavy embeddings — pure set intersection.
    """

    def __init__(
        self,
        max_size: int = 300,
        ttl: int = 1200,
        similarity_threshold: float = 0.75,
        ngram_size: int = 2,
    ):
        self._store = BoundedLRUCache(max_size=max_size, default_ttl=ttl)
        self._index: OrderedDict[str, Tuple[Set[str], float]] = OrderedDict()
        self.threshold = similarity_threshold
        self.ngram_size = ngram_size
        self.max_index_size = max_size
        self._hits = 0
        self._misses = 0

    def lookup(self, query: str, mode: str = "standard") -> Optional[Dict[str, Any]]:
        """Find best lexical match above threshold."""
        query_grams = extract_word_ngrams(query, self.ngram_size)

        if not query_grams:
            self._misses += 1
            return None

        best_key = None
        best_sim = 0.0

        # Linear scan — bounded by max_index_size
        expired_keys = []
        now = time.time()

        for key, (grams, expires_at) in self._index.items():
            if now > expires_at:
                expired_keys.append(key)
                continue
            if not key.startswith(f"lex:{mode}:"):
                continue
            sim = jaccard_similarity(query_grams, grams)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        # Cleanup expired
        for k in expired_keys:
            self._index.pop(k, None)
            self._store.delete(k)

        if best_sim >= self.threshold and best_key:
            result = self._store.get(best_key)
            if result:
                self._hits += 1
                logger.debug(
                    f"Tier-2 cache HIT: sim={best_sim:.3f} key={best_key[:24]}..."
                )
                result["_cache_meta"]["similarity"] = round(best_sim, 3)
                return result

        self._misses += 1
        return None

    def store(self, query: str, mode: str, response: Dict[str, Any], ttl: Optional[int] = None):
        """Store response with ngram index."""
        effective_ttl = ttl or self._store.default_ttl
        key = f"lex:{mode}:{compute_hash(query)}"
        grams = extract_word_ngrams(query, self.ngram_size)

        entry = {
            **response,
            "_cache_meta": {
                "tier": 2,
                "cached_at": time.time(),
                "query_hash": key,
            },
        }
        self._store.set(key, entry, effective_ttl)
        self._index[key] = (grams, time.time() + effective_ttl)

        # Evict oldest index entries
        while len(self._index) > self.max_index_size:
            old_key, _ = self._index.popitem(last=False)
            self._store.delete(old_key)

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "tier": 2,
            "type": "lexical_similarity",
            "threshold": self.threshold,
            "index_size": len(self._index),
            **self._store.stats,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


# ============================================================
# TIER 3: SEMANTIC APPROXIMATION CACHE (Lightweight)
# ============================================================

class SemanticApproxCache:
    """
    Lightweight semantic cache using deterministic hashing of:
      - Intent class (from query classification)
      - Question type (what/how/why/when/compare/list)
      - Extracted entities (capitalized words, quoted terms)

    No FAISS, no embeddings library, no torch.
    """

    # Question type patterns
    QUESTION_TYPES = {
        "definition": re.compile(r'^(what is|define|explain)\b', re.I),
        "causal": re.compile(r'^(why|how come|what causes)\b', re.I),
        "procedural": re.compile(r'^(how to|how do|how can|steps to)\b', re.I),
        "comparative": re.compile(r'\b(compare|vs|versus|difference|better|worse)\b', re.I),
        "temporal": re.compile(r'^(when|what time|what date|how long)\b', re.I),
        "list": re.compile(r'^(list|name|enumerate|what are)\b', re.I),
        "opinion": re.compile(r'\b(should|recommend|best|opinion|think)\b', re.I),
        "factual": re.compile(r'^(who|where|which|is it true)\b', re.I),
    }

    def __init__(self, max_size: int = 200, ttl: int = 900, match_threshold: float = 0.8):
        self._store = BoundedLRUCache(max_size=max_size, default_ttl=ttl)
        self._keys: OrderedDict[str, Tuple[Dict[str, Any], float]] = OrderedDict()
        self.max_keys = max_size
        self.match_threshold = match_threshold
        self._hits = 0
        self._misses = 0

    def lookup(
        self,
        query: str,
        mode: str = "standard",
        intent: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find semantic approximate match.
        """
        features = self._extract_features(query, intent)
        feature_hash = self._hash_features(features, mode)

        # Direct feature-hash match
        result = self._store.get(feature_hash)
        if result:
            self._hits += 1
            logger.debug(f"Tier-3 cache HIT: {feature_hash[:24]}...")
            return result

        # Fuzzy match across stored feature sets
        now = time.time()
        expired = []
        best_key = None
        best_score = 0.0

        for key, (stored_features, expires_at) in self._keys.items():
            if now > expires_at:
                expired.append(key)
                continue
            if not key.startswith(f"sem:{mode}:"):
                continue
            score = self._feature_similarity(features, stored_features)
            if score > best_score:
                best_score = score
                best_key = key

        for k in expired:
            self._keys.pop(k, None)
            self._store.delete(k)

        if best_score >= self.match_threshold and best_key:
            result = self._store.get(best_key)
            if result:
                self._hits += 1
                result["_cache_meta"]["feature_similarity"] = round(best_score, 3)
                return result

        self._misses += 1
        return None

    def store(
        self,
        query: str,
        mode: str,
        response: Dict[str, Any],
        intent: Optional[str] = None,
        ttl: Optional[int] = None,
    ):
        """Store response with feature index."""
        effective_ttl = ttl or self._store.default_ttl
        features = self._extract_features(query, intent)
        feature_hash = self._hash_features(features, mode)

        entry = {
            **response,
            "_cache_meta": {
                "tier": 3,
                "cached_at": time.time(),
                "features": features,
            },
        }
        self._store.set(feature_hash, entry, effective_ttl)
        self._keys[feature_hash] = (features, time.time() + effective_ttl)

        while len(self._keys) > self.max_keys:
            old_key, _ = self._keys.popitem(last=False)
            self._store.delete(old_key)

    def _extract_features(self, query: str, intent: Optional[str] = None) -> Dict[str, Any]:
        """Extract deterministic features from query."""
        normalized = normalize_query(query)
        words = normalized.split()

        # Question type
        q_type = "general"
        for qt, pattern in self.QUESTION_TYPES.items():
            if pattern.search(query):
                q_type = qt
                break

        # Entity extraction (capitalized words, quoted terms, numbers)
        entities = set()
        for word in query.split():
            if word[0:1].isupper() and len(word) > 1:
                entities.add(word.lower())
        quoted = re.findall(r'"([^"]+)"', query)
        entities.update(q.lower() for q in quoted)

        # Topic words (content words, filtered)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
            'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'and', 'or', 'but', 'not', 'about', 'if', 'then', 'so',
        }
        topic_words = sorted(set(
            w for w in words if w not in stop_words and len(w) > 2
        ))[:10]  # Cap at 10 topic words

        return {
            "question_type": q_type,
            "intent": intent or "unknown",
            "entities": sorted(entities)[:5],
            "topic_words": topic_words,
            "word_count_bucket": "short" if len(words) < 10 else "medium" if len(words) < 30 else "long",
        }

    def _hash_features(self, features: Dict[str, Any], mode: str) -> str:
        """Deterministic hash of feature dict."""
        canonical = json.dumps(features, sort_keys=True)
        h = hashlib.sha256(f"{mode}:{canonical}".encode()).hexdigest()[:32]
        return f"sem:{mode}:{h}"

    def _feature_similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Compute similarity between two feature sets (0-1)."""
        score = 0.0
        weights_total = 0.0

        # Question type match (weight: 0.3)
        if a.get("question_type") == b.get("question_type"):
            score += 0.3
        weights_total += 0.3

        # Intent match (weight: 0.2)
        if a.get("intent") == b.get("intent"):
            score += 0.2
        weights_total += 0.2

        # Entity overlap (weight: 0.25)
        ents_a = set(a.get("entities", []))
        ents_b = set(b.get("entities", []))
        if ents_a or ents_b:
            score += 0.25 * jaccard_similarity(ents_a, ents_b)
        else:
            score += 0.25  # Both empty → match
        weights_total += 0.25

        # Topic word overlap (weight: 0.25)
        topics_a = set(a.get("topic_words", []))
        topics_b = set(b.get("topic_words", []))
        if topics_a or topics_b:
            score += 0.25 * jaccard_similarity(topics_a, topics_b)
        else:
            score += 0.25
        weights_total += 0.25

        return score / weights_total if weights_total > 0 else 0.0

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "tier": 3,
            "type": "semantic_approx",
            "index_size": len(self._keys),
            **self._store.stats,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


# ============================================================
# UNIFIED RESPONSE CACHE
# ============================================================

@dataclass
class CacheResult:
    """Result from cache lookup."""
    hit: bool
    tier: int                     # 0 = miss, 1/2/3 = cache tier
    response: Optional[Dict[str, Any]] = None
    similarity: float = 0.0
    stale: bool = False           # True if TTL expired but data available


class ResponseCache:
    """
    Unified 3-tier response cache.

    Usage:
        cache = ResponseCache()

        # Check cache before LLM call
        result = cache.lookup(query, mode)
        if result.hit:
            return result.response

        # After LLM call, store
        cache.store(query, mode, response, intent="factual")
    """

    def __init__(
        self,
        exact_max: int = 500,
        exact_ttl: int = 1800,
        lexical_max: int = 300,
        lexical_ttl: int = 1200,
        lexical_threshold: float = 0.75,
        semantic_max: int = 200,
        semantic_ttl: int = 900,
        semantic_threshold: float = 0.8,
    ):
        self.exact = ExactMatchCache(max_size=exact_max, ttl=exact_ttl)
        self.lexical = LexicalSimilarityCache(
            max_size=lexical_max,
            ttl=lexical_ttl,
            similarity_threshold=lexical_threshold,
        )
        self.semantic = SemanticApproxCache(
            max_size=semantic_max,
            ttl=semantic_ttl,
            match_threshold=semantic_threshold,
        )

    def lookup(
        self,
        query: str,
        mode: str = "standard",
        intent: Optional[str] = None,
    ) -> CacheResult:
        """
        Cascading cache lookup: Tier 1 → Tier 2 → Tier 3.
        """
        # Tier 1: Exact match
        result = self.exact.lookup(query, mode)
        if result:
            return CacheResult(hit=True, tier=1, response=result, similarity=1.0)

        # Tier 2: Lexical similarity
        result = self.lexical.lookup(query, mode)
        if result:
            sim = result.get("_cache_meta", {}).get("similarity", 0.0)
            return CacheResult(hit=True, tier=2, response=result, similarity=sim)

        # Tier 3: Semantic approximation
        result = self.semantic.lookup(query, mode, intent)
        if result:
            sim = result.get("_cache_meta", {}).get("feature_similarity", 0.0)
            return CacheResult(hit=True, tier=3, response=result, similarity=sim)

        return CacheResult(hit=False, tier=0)

    def store(
        self,
        query: str,
        mode: str,
        response: Dict[str, Any],
        intent: Optional[str] = None,
        ttl: Optional[int] = None,
    ):
        """Store response in all applicable cache tiers."""
        # Always store in exact cache
        self.exact.store(query, mode, response, ttl)

        # Store in lexical cache for similarity matching
        self.lexical.store(query, mode, response, ttl)

        # Store in semantic cache if intent available
        self.semantic.store(query, mode, response, intent, ttl)

    def invalidate(self, query: str, mode: str = "standard"):
        """Invalidate a specific query from all tiers."""
        exact_key = f"exact:{mode}:{compute_hash(query)}"
        lex_key = f"lex:{mode}:{compute_hash(query)}"
        self.exact._store.delete(exact_key)
        self.lexical._store.delete(lex_key)
        self.lexical._index.pop(lex_key, None)

    def clear_all(self):
        """Clear all cache tiers."""
        self.exact._store.clear()
        self.lexical._store.clear()
        self.lexical._index.clear()
        self.semantic._store.clear()
        self.semantic._keys.clear()

    @property
    def stats(self) -> Dict:
        return {
            "tier_1": self.exact.stats,
            "tier_2": self.lexical.stats,
            "tier_3": self.semantic.stats,
        }


# ── Module-level singleton ──
_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Get or create the singleton ResponseCache."""
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache
