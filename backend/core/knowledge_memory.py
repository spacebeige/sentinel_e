"""
Three-Tier Knowledge Memory — Sentinel-E Autonomous Reasoning Engine

Tier 1: Session Memory (semantic-decay sliding window)
Tier 2: Evidence Memory (structured evidence objects, indexed)
Tier 3: Knowledge Graph (entities, claims, relationships)

Replaces the flat memory in backend/memory/memory_engine.py with
topic-clustered, decay-weighted, semantically compressed memory.
"""

import uuid
import math
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("KnowledgeMemory")

# ============================================================
# CONFIGURATION
# ============================================================

DECAY_LAMBDA = 0.15           # 15% weight loss per hour of inactivity
COMPRESSION_THRESHOLD = 0.65  # Compress messages below this relevance
PRUNE_THRESHOLD = 0.05        # Remove messages below this weight
DEFAULT_MAX_TOKENS = 6144     # Context token budget
DEFAULT_MAX_MESSAGES = 16     # Hard message limit


# ============================================================
# COSINE SIMILARITY (standalone, no scipy dependency)
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token for English)."""
    return max(1, len(text) // 4)


# ============================================================
# TIER 1: SESSION MEMORY (Short-Term)
# ============================================================

@dataclass
class WeightedMessage:
    """A message with semantic weight and optional compression."""
    role: str                              # user | assistant | system
    content: str
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    relevance_weight: float = 1.0
    token_count: int = 0
    compressed: bool = False
    original_content: Optional[str] = None

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = _count_tokens_approx(self.content)

    def to_prompt_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TopicArchive:
    """Archived topic cluster."""
    cluster_id: str
    topic_embedding: Optional[np.ndarray]
    message_count: int
    summary: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    archived_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionMemoryTier:
    """
    Tier 1: Semantic-decay sliding window scoped to the active topic cluster.
    Replaces backend/memory/memory_engine.py::ShortTermMemory.
    """

    def __init__(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        embed_fn=None,
        compress_fn=None,
    ):
        self.topic_cluster_id: str = str(uuid.uuid4())
        self.topic_embedding: Optional[np.ndarray] = None
        self.messages: List[WeightedMessage] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.archives: List[TopicArchive] = []
        self.last_query_time: datetime = datetime.now(timezone.utc)
        self._embed_fn = embed_fn
        self._compress_fn = compress_fn

    def add_message(self, role: str, content: str, embedding: Optional[np.ndarray] = None):
        """Add a new message, compute embedding, update centroid, apply decay."""
        if embedding is None and self._embed_fn is not None:
            try:
                embedding = self._embed_fn(content)
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
            except Exception:
                pass

        msg = WeightedMessage(
            role=role,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(timezone.utc),
            relevance_weight=1.0,
        )
        self.messages.append(msg)

        if role == "user":
            self.last_query_time = msg.timestamp

        # Update topic centroid from user messages
        user_embeddings = [m.embedding for m in self.messages if m.role == "user" and m.embedding is not None]
        if user_embeddings:
            self.topic_embedding = np.mean(user_embeddings, axis=0).astype(np.float32)

        # Apply decay and manage capacity
        self._apply_decay()
        self._manage_capacity()

    def _apply_decay(self):
        """Apply time + topic relevance decay to all messages."""
        now = datetime.now(timezone.utc)
        for msg in self.messages:
            age_hours = (now - msg.timestamp).total_seconds() / 3600
            time_decay = math.exp(-DECAY_LAMBDA * age_hours)

            topic_rel = 0.5
            if self.topic_embedding is not None and msg.embedding is not None:
                topic_rel = cosine_similarity(msg.embedding, self.topic_embedding)

            role_boost = 1.5 if msg.role == "system" else 1.0
            if msg.compressed:
                role_boost *= 0.8

            msg.relevance_weight = max(0.0, min(1.0, time_decay * topic_rel * role_boost))

    def _manage_capacity(self):
        """Compress low-relevance messages and prune if over budget."""
        # Prune dead-weight messages
        self.messages = [m for m in self.messages if m.relevance_weight >= PRUNE_THRESHOLD]

        total_tokens = sum(m.token_count for m in self.messages)

        while total_tokens > self.max_tokens or len(self.messages) > self.max_messages:
            # Find lowest-weight non-compressed message to compress
            candidates = [
                m for m in self.messages
                if not m.compressed and m.relevance_weight < COMPRESSION_THRESHOLD
            ]
            if candidates and self._compress_fn:
                target = min(candidates, key=lambda m: m.relevance_weight)
                try:
                    compressed_text = self._compress_fn(target.content)
                    target.original_content = target.content
                    target.content = compressed_text
                    target.token_count = _count_tokens_approx(compressed_text)
                    target.compressed = True
                except Exception:
                    # Compression failed — just remove
                    self.messages.remove(target)
            else:
                # Remove lowest-weight message entirely
                if self.messages:
                    weakest = min(self.messages, key=lambda m: m.relevance_weight)
                    self.messages.remove(weakest)
                else:
                    break
            total_tokens = sum(m.token_count for m in self.messages)

    def get_context_messages(self, min_weight: float = 0.3) -> List[Dict[str, str]]:
        """Return messages above minimum weight, formatted for prompt injection."""
        return [
            m.to_prompt_dict()
            for m in self.messages
            if m.relevance_weight >= min_weight
        ]

    def archive_and_reset(self, summary: str = ""):
        """Archive current topic cluster and start fresh."""
        archive = TopicArchive(
            cluster_id=self.topic_cluster_id,
            topic_embedding=self.topic_embedding,
            message_count=len(self.messages),
            summary=summary,
        )
        self.archives.append(archive)
        # Keep last 5 archives
        if len(self.archives) > 5:
            self.archives = self.archives[-5:]

        # Reset
        self.topic_cluster_id = str(uuid.uuid4())
        self.topic_embedding = None
        self.messages = []

    def restore_archive(self, cluster_id: str) -> bool:
        """Restore an archived topic cluster."""
        for archive in self.archives:
            if archive.cluster_id == cluster_id:
                self.topic_cluster_id = archive.cluster_id
                self.topic_embedding = archive.topic_embedding
                # Note: messages are not preserved in archive (by design — only centroid)
                self.archives.remove(archive)
                return True
        return False


# ============================================================
# TIER 2: EVIDENCE MEMORY (Mid-Term)
# ============================================================

@dataclass
class AtomicClaim:
    """A single atomic claim extracted from evidence."""
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str = ""
    source_url: str = ""
    confidence: float = 0.5
    claim_type: str = "factual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "claim_type": self.claim_type,
        }


@dataclass
class EvidenceObject:
    """Structured evidence stored in mid-term memory."""
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_origin: str = ""
    entity_tags: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    topic_embedding: Optional[np.ndarray] = None
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    source_metadata: List[Dict[str, Any]] = field(default_factory=list)
    claims_extracted: List[AtomicClaim] = field(default_factory=list)
    confidence_score: float = 0.5
    contradiction_flags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "query_origin": self.query_origin,
            "entity_tags": self.entity_tags,
            "topic_tags": self.topic_tags,
            "chunks": self.chunks,
            "source_metadata": self.source_metadata,
            "claims_extracted": [c.to_dict() for c in self.claims_extracted],
            "confidence_score": self.confidence_score,
            "contradiction_flags": self.contradiction_flags,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
        }


class EvidenceMemory:
    """
    Tier 2: Persistent evidence memory with multi-index retrieval.
    In-memory implementation; pluggable Postgres backend.
    """

    def __init__(self):
        self._store: Dict[str, EvidenceObject] = {}  # evidence_id → object
        self._entity_index: Dict[str, List[str]] = {}  # entity → [evidence_id]
        self._topic_embeddings: List[Tuple[str, np.ndarray]] = []  # (ev_id, embedding)

    def store(self, evidence: EvidenceObject):
        """Store or update an evidence object."""
        self._store[evidence.evidence_id] = evidence

        # Entity index
        for entity in evidence.entity_tags:
            if entity not in self._entity_index:
                self._entity_index[entity] = []
            if evidence.evidence_id not in self._entity_index[entity]:
                self._entity_index[entity].append(evidence.evidence_id)

        # Topic embedding index
        if evidence.topic_embedding is not None:
            self._topic_embeddings.append((evidence.evidence_id, evidence.topic_embedding))

    def query_by_topic(
        self,
        topic_embedding: np.ndarray,
        k: int = 10,
        min_confidence: float = 0.4,
    ) -> List[EvidenceObject]:
        """Retrieve evidence objects closest to a topic centroid."""
        scored = []
        for ev_id, emb in self._topic_embeddings:
            sim = cosine_similarity(topic_embedding, emb)
            ev = self._store.get(ev_id)
            if ev and ev.confidence_score >= min_confidence:
                scored.append((sim, ev))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ev for _, ev in scored[:k]]
        # Update access counters
        for ev in results:
            ev.access_count += 1
            ev.last_accessed = datetime.now(timezone.utc)
        return results

    def query_by_entity(self, entity: str) -> List[EvidenceObject]:
        """Retrieve evidence objects tagged with a specific entity."""
        ev_ids = self._entity_index.get(entity, [])
        results = [self._store[eid] for eid in ev_ids if eid in self._store]
        for ev in results:
            ev.access_count += 1
            ev.last_accessed = datetime.now(timezone.utc)
        return results

    def get_since(self, since: datetime) -> List[EvidenceObject]:
        """Get evidence objects created after a given timestamp."""
        return [
            ev for ev in self._store.values()
            if ev.timestamp >= since
        ]

    def get_all(self) -> List[EvidenceObject]:
        return list(self._store.values())


# ============================================================
# TIER 3: KNOWLEDGE GRAPH (Long-Term)
# ============================================================

@dataclass
class KGEntity:
    """Knowledge graph entity."""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    aliases: List[str] = field(default_factory=list)
    entity_type: str = "concept"  # person | org | concept | event | location | product
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.5
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "aliases": self.aliases,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class TemporalValidity:
    """When a claim is/was valid."""
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    is_current: bool = True
    temporal_qualifier: str = "current"  # permanent | current | historical | projected


@dataclass
class KGClaim:
    """Atomic knowledge graph claim."""
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str = ""
    subject_entity_id: str = ""
    predicate: str = ""
    object_entity_id: Optional[str] = None
    object_value: Optional[str] = None
    source_evidence_ids: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    confidence: float = 0.5
    temporal_validity: TemporalValidity = field(default_factory=TemporalValidity)
    conflict_marker: Optional[str] = None
    status: str = "active"  # active | superseded | disputed | retracted
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "subject_entity_id": self.subject_entity_id,
            "predicate": self.predicate,
            "object_entity_id": self.object_entity_id,
            "object_value": self.object_value,
            "confidence": self.confidence,
            "status": self.status,
            "source_urls": self.source_urls,
            "conflict_marker": self.conflict_marker,
        }


@dataclass
class KGRelationship:
    """Knowledge graph edge."""
    relationship_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: str = ""  # is_a | part_of | causes | contradicts | supports
    weight: float = 0.5
    evidence_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeGraph:
    """
    Tier 3: In-memory knowledge graph with entity/claim/relationship storage.
    Provides conflict resolution and coverage computation.
    Pluggable Postgres persistence via schema.sql extensions.
    """

    def __init__(self):
        self.entities: Dict[str, KGEntity] = {}
        self.claims: Dict[str, KGClaim] = {}
        self.relationships: Dict[str, KGRelationship] = {}
        # Indexes
        self._entity_by_name: Dict[str, str] = {}  # lowercase(name) → entity_id
        self._claims_by_subject: Dict[str, List[str]] = {}
        self._entity_embeddings: List[Tuple[str, np.ndarray]] = []

    # --- Entity operations ---

    def add_entity(self, entity: KGEntity):
        self.entities[entity.entity_id] = entity
        self._entity_by_name[entity.name.lower()] = entity.entity_id
        for alias in entity.aliases:
            self._entity_by_name[alias.lower()] = entity.entity_id
        if entity.embedding is not None:
            self._entity_embeddings.append((entity.entity_id, entity.embedding))

    def find_entity(self, name: str, entity_type: str = None) -> Optional[KGEntity]:
        eid = self._entity_by_name.get(name.lower())
        if eid:
            entity = self.entities.get(eid)
            if entity and (entity_type is None or entity.entity_type == entity_type):
                return entity
        return None

    def get_related_entities(self, query_embedding: np.ndarray, k: int = 15) -> List[KGEntity]:
        """Find entities closest to a query embedding."""
        scored = []
        for eid, emb in self._entity_embeddings:
            sim = cosine_similarity(query_embedding, emb)
            entity = self.entities.get(eid)
            if entity:
                scored.append((sim, entity))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    # --- Claim operations ---

    def add_claim(self, claim: KGClaim):
        self.claims[claim.claim_id] = claim
        if claim.subject_entity_id not in self._claims_by_subject:
            self._claims_by_subject[claim.subject_entity_id] = []
        self._claims_by_subject[claim.subject_entity_id].append(claim.claim_id)

    def get_claims_for_entities(
        self,
        entity_ids: List[str],
        status_filter: List[str] = None,
        min_confidence: float = 0.3,
    ) -> List[KGClaim]:
        """Get all claims related to given entities."""
        if status_filter is None:
            status_filter = ["active", "disputed"]
        claim_ids = set()
        for eid in entity_ids:
            claim_ids.update(self._claims_by_subject.get(eid, []))
        results = []
        for cid in claim_ids:
            claim = self.claims.get(cid)
            if claim and claim.status in status_filter and claim.confidence >= min_confidence:
                results.append(claim)
        return results

    def find_conflicting_claims(self, new_claim_text: str, embedding: np.ndarray = None, threshold: float = 0.80) -> List[KGClaim]:
        """Find existing claims that might conflict with a new claim."""
        # Simple: check by subject entity overlap and high similarity
        # For now, brute-force text overlap; upgrade to embedding-based with FAISS
        conflicts = []
        new_lower = new_claim_text.lower()
        for claim in self.claims.values():
            if claim.status in ("retracted", "superseded"):
                continue
            # Simple word overlap heuristic
            new_words = set(new_lower.split())
            old_words = set(claim.claim_text.lower().split())
            overlap = len(new_words & old_words) / max(len(new_words | old_words), 1)
            if overlap > 0.5:  # Significant word overlap
                conflicts.append(claim)
        return conflicts

    # --- Relationship operations ---

    def add_relationship(self, rel: KGRelationship):
        self.relationships[rel.relationship_id] = rel

    # --- Coverage computation ---

    def compute_coverage(self, query: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Compute how well the knowledge graph covers a query.
        Returns score (0–1) and relevant claims.
        """
        related = self.get_related_entities(query_embedding, k=10)
        if not related:
            return {"score": 0.0, "claims": [], "entity_count": 0}

        claims = self.get_claims_for_entities([e.entity_id for e in related])
        active_claims = [c for c in claims if c.status == "active"]

        # Score based on entity coverage and claim confidence
        entity_score = min(len(related) / 5, 1.0)
        claim_score = min(len(active_claims) / 3, 1.0) if active_claims else 0.0
        avg_confidence = (
            sum(c.confidence for c in active_claims) / len(active_claims)
            if active_claims else 0.0
        )

        coverage = 0.4 * entity_score + 0.3 * claim_score + 0.3 * avg_confidence

        return {
            "score": round(coverage, 4),
            "claims": active_claims,
            "entity_count": len(related),
            "claim_count": len(active_claims),
        }

    # --- Conflict resolution ---

    def resolve_conflict(self, new_claim: KGClaim, existing: KGClaim):
        """
        Resolve conflict between two claims using temporal and source strength.
        """
        new_strength = self._compute_claim_strength(new_claim)
        old_strength = self._compute_claim_strength(existing)

        # Temporal: newer supersedes if existing is not current
        if new_claim.temporal_validity.is_current and not existing.temporal_validity.is_current:
            existing.status = "superseded"
            existing.temporal_validity.valid_until = datetime.now(timezone.utc)
            new_claim.status = "active"
            return

        # Source strength: 30% threshold
        if new_strength > old_strength * 1.3:
            existing.status = "superseded"
            new_claim.status = "active"
        elif old_strength > new_strength * 1.3:
            new_claim.status = "disputed"
            new_claim.conflict_marker = existing.claim_id
        else:
            # Genuine dispute
            new_claim.status = "disputed"
            existing.status = "disputed"
            new_claim.conflict_marker = existing.claim_id
            existing.conflict_marker = new_claim.claim_id

    def _compute_claim_strength(self, claim: KGClaim) -> float:
        source_count = len(claim.source_evidence_ids)
        source_qty = min(source_count / 5, 1.0) * 0.4
        confidence_part = claim.confidence * 0.35
        # Recency
        days_since = (datetime.now(timezone.utc) - claim.last_verified).days
        recency = 1.0 / (1.0 + days_since / 30) * 0.15
        base = 0.10  # Base floor
        return min(1.0, source_qty + confidence_part + recency + base)
