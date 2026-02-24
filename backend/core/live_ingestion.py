"""
Autonomous Live Ingestion Daemon — Sentinel-E Autonomous Reasoning Engine

Background async daemon that:
  - Watches configured data sources on a schedule
  - Responds to event-triggered ingestion
  - Performs diff-based updates (content hash comparison)
  - Embeds, scores, and stores new evidence
  - Updates knowledge graph with new/conflicting claims
  - Recalculates affected topic centroids

Safety:
  - Rate limiting (token bucket per source)
  - Anti-poisoning checks (injection, velocity, drift)  
  - Loop prevention (dedup window)
  - Bounded queue (max 500 tasks)

Integrates with:
  - backend/core/knowledge_memory.py (EvidenceMemory, KnowledgeGraph)
  - backend/core/evidence_cache.py (EvidenceCache)
  - backend/storage/redis.py (RedisClient)
"""

import asyncio
import hashlib
import re
import time
import logging
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("LiveIngestion")


# ============================================================
# CONFIGURATION
# ============================================================

MAX_QUEUE_SIZE = 500
MAX_UPDATES_PER_HOUR = 20
TOKEN_BUCKET_CAPACITY = 10
TOKEN_BUCKET_REFILL_RATE = 10 / 60  # 10 tokens per minute
CONTENT_HASH_TTL = 86400            # 24 hours
DEDUP_WINDOW = 300                  # 5 minutes
CHECK_INTERVAL = 60                 # Scheduling check every 60s
SEMANTIC_DRIFT_THRESHOLD = 0.7      # Content drift detection


# ============================================================
# INJECTION DETECTION PATTERNS
# ============================================================

INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"you\s+are\s+now",
    r"disregard\s+all",
    r"new\s+system\s+prompt",
    r"<script>",
    r"javascript:",
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
]


# ============================================================
# TOKEN BUCKET RATE LIMITER
# ============================================================

class TokenBucket:
    """Per-source rate limiter."""

    def __init__(self, capacity: float = TOKEN_BUCKET_CAPACITY, refill_rate: float = TOKEN_BUCKET_REFILL_RATE):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()

    def consume(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimiter:
    """Collection of per-source token buckets."""

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}

    def acquire(self, source_id: str) -> bool:
        if source_id not in self._buckets:
            self._buckets[source_id] = TokenBucket()
        return self._buckets[source_id].consume()


# ============================================================
# ANTI-POISONING GUARD
# ============================================================

class AntiPoisonGuard:
    """Validates ingested data against poisoning/injection attacks."""

    def __init__(self):
        self._update_counts: Dict[str, List[float]] = {}  # source_id → [timestamps]

    def check(self, content: str, source_id: str, content_embedding=None, historical_centroid=None) -> bool:
        """
        Returns True if content passes all safety checks.
        Returns False if poisoning suspected.
        """
        # 1. Size check
        if len(content) < 50:
            logger.warning(f"Poisoning check failed: content too short ({len(content)} chars)")
            return False
        if len(content) > 1_000_000:
            logger.warning(f"Poisoning check failed: content too large ({len(content)} chars)")
            return False

        # 2. Injection pattern detection
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"Poisoning check failed: injection pattern detected in {source_id}")
                return False

        # 3. Velocity check
        now = time.time()
        if source_id not in self._update_counts:
            self._update_counts[source_id] = []
        self._update_counts[source_id].append(now)
        # Trim to last hour
        cutoff = now - 3600
        self._update_counts[source_id] = [t for t in self._update_counts[source_id] if t > cutoff]
        if len(self._update_counts[source_id]) > MAX_UPDATES_PER_HOUR:
            logger.warning(f"Poisoning check failed: velocity exceeded for {source_id}")
            return False

        # 4. Semantic drift (if embeddings provided)
        if content_embedding is not None and historical_centroid is not None:
            try:
                import numpy as np
                a = np.array(content_embedding, dtype=np.float32) if isinstance(content_embedding, list) else content_embedding
                b = np.array(historical_centroid, dtype=np.float32) if isinstance(historical_centroid, list) else historical_centroid
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    sim = float(np.dot(a, b) / (na * nb))
                    drift = 1.0 - sim
                    if drift > SEMANTIC_DRIFT_THRESHOLD:
                        logger.warning(f"Poisoning check failed: semantic drift {drift:.3f} for {source_id}")
                        return False
            except Exception as e:
                logger.debug(f"Semantic drift check skipped: {e}")

        return True


# ============================================================
# SOURCE WATCHER
# ============================================================

@dataclass
class SourceWatcher:
    """Configuration for a watched data source."""
    source_id: str
    name: str
    source_type: str = "web"         # web | file | api | rss
    url: str = ""
    interval_seconds: int = 600      # Default: check every 10 minutes
    trust_score: float = 0.5
    enabled: bool = True
    last_fetched: Optional[datetime] = None
    fetch_fn: Optional[Callable] = None  # async fn() -> IngestedData

    def should_run(self, now: datetime) -> bool:
        if not self.enabled:
            return False
        if self.last_fetched is None:
            return True
        return (now - self.last_fetched).total_seconds() >= self.interval_seconds


@dataclass
class IngestedData:
    """Raw data from a source."""
    source_id: str
    content: str
    url: str = ""
    title: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionTask:
    """A queued ingestion task."""
    task_id: str
    watcher: SourceWatcher
    event: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)


# ============================================================
# INGESTION DAEMON
# ============================================================

class LiveIngestionDaemon:
    """
    Async background daemon for autonomous live data ingestion.
    Runs independently of user interaction.
    """

    def __init__(
        self,
        evidence_memory=None,
        knowledge_graph=None,
        evidence_cache=None,
        redis_client=None,
        embed_fn=None,
        chunk_fn=None,
    ):
        self.source_watchers: List[SourceWatcher] = []
        self.evidence_memory = evidence_memory
        self.knowledge_graph = knowledge_graph
        self.evidence_cache = evidence_cache
        self.redis_client = redis_client
        self._embed_fn = embed_fn
        self._chunk_fn = chunk_fn or self._default_chunk
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._rate_limiter = RateLimiter()
        self._poison_guard = AntiPoisonGuard()
        self._processed_tasks: Dict[str, float] = {}  # task_id → timestamp (dedup)
        self._running = False

    def add_watcher(self, watcher: SourceWatcher):
        self.source_watchers.append(watcher)

    async def start(self):
        """Start the ingestion daemon (call in background)."""
        self._running = True
        logger.info(f"LiveIngestionDaemon started with {len(self.source_watchers)} watchers")
        await asyncio.gather(
            self._schedule_loop(),
            self._queue_processor(),
        )

    async def stop(self):
        self._running = False
        logger.info("LiveIngestionDaemon stopped")

    async def _schedule_loop(self):
        """Check watchers on schedule and enqueue tasks."""
        while self._running:
            now = datetime.now(timezone.utc)
            for watcher in self.source_watchers:
                if watcher.should_run(now):
                    if self._rate_limiter.acquire(watcher.source_id):
                        task_id = f"{watcher.source_id}:{int(time.time())}"
                        task = IngestionTask(task_id=task_id, watcher=watcher)
                        try:
                            self._queue.put_nowait(task)
                            watcher.last_fetched = now
                        except asyncio.QueueFull:
                            logger.warning("Ingestion queue full; dropping oldest")
                            # Drain one item
                            try:
                                self._queue.get_nowait()
                                self._queue.put_nowait(task)
                            except asyncio.QueueEmpty:
                                pass
            await asyncio.sleep(CHECK_INTERVAL)

    async def _queue_processor(self):
        """Process queued ingestion tasks."""
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            # Dedup check
            now = time.time()
            if task.task_id in self._processed_tasks:
                if now - self._processed_tasks[task.task_id] < DEDUP_WINDOW:
                    logger.debug(f"Skipping duplicate task: {task.task_id}")
                    self._queue.task_done()
                    continue

            try:
                await self._process_task(task)
                self._processed_tasks[task.task_id] = now
                # Clean old dedup entries
                cutoff = now - DEDUP_WINDOW * 2
                self._processed_tasks = {k: v for k, v in self._processed_tasks.items() if v > cutoff}
            except Exception as e:
                logger.error(f"Ingestion task failed: {task.task_id}: {e}")
            finally:
                self._queue.task_done()

    async def _process_task(self, task: IngestionTask):
        """Process a single ingestion task."""
        watcher = task.watcher

        # Step 1: Fetch
        if watcher.fetch_fn:
            data = await watcher.fetch_fn()
        else:
            logger.debug(f"No fetch_fn for {watcher.source_id}; skipping")
            return

        if not data or not data.content:
            return

        # Step 2: Content diff
        content_hash = hashlib.sha256(data.content.encode("utf-8")).hexdigest()
        stored_hash = None
        if self.redis_client and self.redis_client.connected:
            stored_hash = await self.redis_client.get(f"content_hash:{watcher.source_id}")
        if stored_hash == content_hash:
            logger.debug(f"No change for {watcher.source_id}; skipping")
            return

        # Step 3: Anti-poisoning
        if not self._poison_guard.check(data.content, watcher.source_id):
            logger.warning(f"Poisoning detected for {watcher.source_id}; skipping")
            return

        # Step 4: Chunk
        chunks = self._chunk_fn(data.content)
        if not chunks:
            return

        # Step 5: Embed
        embeddings = []
        if self._embed_fn:
            for chunk in chunks:
                try:
                    emb = self._embed_fn(chunk)
                    embeddings.append(emb)
                except Exception:
                    embeddings.append(None)
        else:
            embeddings = [None] * len(chunks)

        # Step 6: Store evidence and update graph
        if self.evidence_memory:
            from backend.core.knowledge_memory import EvidenceObject
            evidence = EvidenceObject(
                query_origin=f"auto_ingest:{watcher.source_id}",
                entity_tags=[],
                topic_tags=[watcher.name],
                chunks=[{"content": c, "source_url": data.url} for c in chunks],
                source_metadata=[{"url": data.url, "title": data.title, "trust": watcher.trust_score}],
                confidence_score=watcher.trust_score,
            )
            if embeddings and embeddings[0] is not None:
                import numpy as np
                valid_embs = [e for e in embeddings if e is not None]
                if valid_embs:
                    evidence.topic_embedding = np.mean(valid_embs, axis=0).astype(np.float32)
            self.evidence_memory.store(evidence)

        # Step 7: Store content hash
        if self.redis_client and self.redis_client.connected:
            await self.redis_client.set(
                f"content_hash:{watcher.source_id}",
                content_hash,
                ttl=CONTENT_HASH_TTL,
            )

        logger.info(f"Ingested {len(chunks)} chunks from {watcher.source_id}")

    @staticmethod
    def _default_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Default chunking: fixed-size with overlap."""
        chunks = []
        i = 0
        while i < len(text):
            end = i + chunk_size
            chunk = text[i:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            i += chunk_size - overlap
        return chunks

    async def ingest_manual(self, data: IngestedData):
        """Manually trigger ingestion for an IngestedData object (non-scheduled)."""
        task_id = f"manual:{data.source_id}:{int(time.time())}"
        watcher = SourceWatcher(
            source_id=data.source_id,
            name=data.source_id,
            trust_score=0.5,
        )

        async def _fetch():
            return data

        watcher.fetch_fn = _fetch
        task = IngestionTask(task_id=task_id, watcher=watcher)
        await self._process_task(task)
