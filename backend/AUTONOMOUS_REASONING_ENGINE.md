# Sentinel-E Autonomous Reasoning Engine — Architecture Specification

**Version**: 4.0  
**Classification**: Implementation-Grade System Design  
**Date**: 2026-02-24  

---

## SYSTEM ARCHITECTURE DIAGRAM

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           SENTINEL-E AUTONOMOUS REASONING ENGINE                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────────────────┐  │
│  │  USER INPUT  │───▶│  INTENT HASHER   │───▶│          EVIDENCE CACHE            │  │
│  └─────────────┘    │                  │    │  ┌──────────┐  ┌───────────────┐   │  │
│         │           │ • Query→Struct   │    │  │ Hit?     │  │ Freshness OK? │   │  │
│         │           │ • SHA3-256 Hash  │    │  │ sim>0.87 │  │ TTL check     │   │  │
│         │           │ • Session Store  │    │  └────┬─────┘  └───────┬───────┘   │  │
│         │           └──────────────────┘    │       │YES             │YES         │  │
│         │                                    │       ▼                ▼            │  │
│         │                                    │  ┌────────────────────────────┐    │  │
│         │                                    │  │  CACHED EVIDENCE REUSE     │    │  │
│         │                                    │  │  (Skip retrieval entirely) │    │  │
│         │                                    │  └────────────────────────────┘    │  │
│         │                                    │       │NO                          │  │
│         │                                    │       ▼                            │  │
│         │                                    │  ┌────────────────────────────┐    │  │
│         │                                    │  │  LIVE RETRIEVAL PIPELINE   │    │  │
│         │                                    │  │  Tavily → Score → Dedup    │    │  │
│         │                                    │  │  → Embed → Cache → Store   │    │  │
│         │                                    │  └────────────────────────────┘    │  │
│         │                                    └─────────────────────────────────────┘  │
│         │                                                                             │
│  ┌──────▼───────────────────────────────────────────────────────────────────────────┐ │
│  │                        THREE-TIER MEMORY ARCHITECTURE                            │ │
│  │                                                                                  │ │
│  │  ┌─────────────────┐  ┌──────────────────────┐  ┌────────────────────────────┐  │ │
│  │  │ TIER 1: SESSION  │  │ TIER 2: EVIDENCE     │  │ TIER 3: KNOWLEDGE GRAPH   │  │ │
│  │  │ (Short-Term)     │  │ (Mid-Term)           │  │ (Long-Term)               │  │ │
│  │  │                  │  │                      │  │                            │  │ │
│  │  │ • Topic cluster  │  │ • Structured objs    │  │ • Entities                │  │ │
│  │  │ • Sliding window │  │ • Entity+Time index  │  │ • Claims                  │  │ │
│  │  │ • Semantic comp  │  │ • Topic embedding    │  │ • Relationships           │  │ │
│  │  │ • Decay weights  │  │ • Confidence scores  │  │ • Conflict markers        │  │ │
│  │  │                  │  │ • Source metadata     │  │ • Temporal validity       │  │ │
│  │  └────────┬─────────┘  └──────────┬───────────┘  └──────────┬─────────────────┘  │ │
│  │           │                       │                          │                    │ │
│  │           └───────────┬───────────┘──────────────────────────┘                    │ │
│  │                       ▼                                                           │ │
│  │              ┌────────────────────┐                                               │ │
│  │              │  CONTEXT COMPILER  │                                               │ │
│  │              │  ┌──────────────┐  │                                               │ │
│  │              │  │ [Verified]   │  │                                               │ │
│  │              │  │ [Updates]    │  │                                               │ │
│  │              │  │ [Conflicts]  │  │                                               │ │
│  │              │  │ [Confidence] │  │                                               │ │
│  │              │  └──────────────┘  │                                               │ │
│  │              └────────┬───────────┘                                               │ │
│  └───────────────────────┼──────────────────────────────────────────────────────────┘ │
│                          ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐│
│  │                        TOPIC BOUNDARY CONTROLLER                                  ││
│  │  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────────────┐       ││
│  │  │ Drift Detector │  │ Follow-Up Anchor│  │ Context Decay Engine         │       ││
│  │  │ cosine(Δ)<0.4  │  │ Intent hash link│  │ weight *= e^(-λ·Δt)         │       ││
│  │  │ → topic shift  │  │ Scope verify    │  │ prune if weight < threshold  │       ││
│  │  └────────────────┘  └─────────────────┘  └──────────────────────────────┘       ││
│  └───────────────────────────────────────────────────────────────────────────────────┘│
│                          │                                                            │
│                          ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐│
│  │                        HALLUCINATION CONTROL GATE                                 ││
│  │  ┌────────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐       ││
│  │  │ Claim Extraction   │  │ Source Verify    │  │ Coverage Gate            │       ││
│  │  │ Atomic decompose   │  │ claim→evidence   │  │ coverage < 0.6 → regen  │       ││
│  │  │ Per-sentence parse │  │ mapping check    │  │ unsupported → strip      │       ││
│  │  └────────────────────┘  └─────────────────┘  └──────────────────────────┘       ││
│  └───────────────────────────────────────────────────────────────────────────────────┘│
│                          │                                                            │
│                          ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐│
│  │                        AUTONOMOUS LIVE INGESTION DAEMON                           ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   ││
│  │  │ Source Watch  │  │ Diff Engine  │  │ Graph Update │  │ Anti-Poison Guard  │   ││
│  │  │ Cron + Event  │  │ Content hash │  │ Centroid     │  │ Trust scoring      │   ││
│  │  │ Rate limited  │  │ comparison   │  │ recalculate  │  │ Anomaly detection  │   ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────────┘   ││
│  └───────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐│
│  │                      MODEL CALL (with compiled context)                           ││
│  │  system_prompt + [Verified Evidence] + [Recent Updates] + [Known Conflicts]       ││
│  │  + [Confidence Metrics] + recent_messages (last N, decayed)                       ││
│  └───────────────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 1 — RETRIEVAL FEEDBACK LOOP

### A. Intent Hashing Layer

**Purpose**: Convert every user query into a deterministic, comparable structural representation that enables exact-match and fuzzy-match cache lookups.

**Algorithm**:

```
FUNCTION hash_intent(query: str, session_id: str) -> IntentHash:
    1. normalized = lowercase(strip(query))
    2. tokens = tokenize(normalized)           // word-level tokenization
    3. stop_removed = remove_stopwords(tokens)  // remove "the", "a", "is", etc.
    4. lemmatized = [lemmatize(t) for t in stop_removed]
    5. sorted_tokens = sort(lemmatized)         // canonical ordering
    6. canonical_repr = join(sorted_tokens, " ")
    7. 
    8. // Deterministic hash for exact-match
    9. exact_hash = SHA3_256(canonical_repr)
   10.
   11. // Semantic embedding for fuzzy-match
   12. embedding = embed(query)                 // all-MiniLM-L6-v2, 384-dim
   13.
   14. // Structural intent features
   15. classification = QueryClassifier.classify(query)  // existing classifier
   16.
   17. RETURN IntentHash(
   18.     exact_hash    = exact_hash,
   19.     embedding     = embedding,
   20.     canonical     = canonical_repr,
   21.     intent_type   = classification.primary_intent,
   22.     retrieval_p   = classification.retrieval_probability,
   23.     temporal_flag = classification.temporal_sensitivity > 0.5,
   24.     session_id    = session_id,
   25.     timestamp     = utcnow()
   26. )
```

**Schema**:

```python
@dataclass
class IntentHash:
    exact_hash: str           # SHA3-256 of canonical form
    embedding: np.ndarray     # 384-dim float32
    canonical: str            # Normalized token string
    intent_type: str          # factual | analytical | creative | temporal | ...
    retrieval_p: float        # 0.0–1.0 probability retrieval needed
    temporal_flag: bool       # True if time-sensitive
    session_id: str
    timestamp: datetime
```

**Session Cache Storage**: Redis key `intent:{session_id}:{exact_hash}`, TTL = 3600s (1 hour), serialized as msgpack.

### B. Evidence Cache System

For every retrieval execution, store:

```python
@dataclass
class EvidenceCacheEntry:
    cache_key: str                        # SHA3-256 of query canonical
    query_canonical: str
    query_embedding: np.ndarray           # 384-dim
    topic_centroid: np.ndarray            # Centroid of all chunk embeddings
    retrieved_chunks: List[EvidenceChunk]  # Ordered by relevance
    source_metadata: List[SourceMeta]
    timestamp: datetime
    ttl_seconds: int                      # Default 1800 (30 min)
    confidence_score: float               # Aggregate evidence confidence
    freshness_class: str                  # "live" | "recent" | "aged" | "stale"
    retrieval_latency_ms: int
    hit_count: int = 0                    # Number of times cache was reused

@dataclass
class EvidenceChunk:
    content: str
    embedding: np.ndarray
    source_url: str
    source_domain: str
    reliability_score: float
    content_hash: str
    position_in_source: int

@dataclass
class SourceMeta:
    url: str
    domain: str
    title: str
    reliability_score: float
    retrieved_at: datetime
    content_hash: str
```

**Cache Hit Detection Algorithm**:

```
FUNCTION check_evidence_cache(intent_hash: IntentHash) -> Optional[EvidenceCacheEntry]:
    // Phase 1: Exact match
    exact_key = f"evidence:{intent_hash.exact_hash}"
    cached = redis.get(exact_key)
    IF cached AND NOT is_expired(cached):
        IF intent_hash.temporal_flag AND cached.freshness_class in ("aged", "stale"):
            // Temporal query needs fresh data — cache miss
            RETURN None
        cached.hit_count += 1
        redis.set(exact_key, cached)
        RETURN cached

    // Phase 2: Semantic similarity scan
    // Load all active cache embeddings from FAISS index
    candidates = faiss_cache_index.search(
        intent_hash.embedding,
        k=5,
        threshold=0.0  // get top-5, filter below
    )

    FOR candidate IN candidates:
        similarity = cosine_similarity(intent_hash.embedding, candidate.query_embedding)
        IF similarity >= SEMANTIC_CACHE_THRESHOLD:  // 0.87
            IF NOT is_expired(candidate):
                IF intent_hash.temporal_flag AND candidate.freshness_class in ("aged", "stale"):
                    CONTINUE  // Skip stale temporal results
                candidate.hit_count += 1
                RETURN candidate

    RETURN None  // Full cache miss — must retrieve
```

**Parameters**:

| Parameter | Value | Rationale |
|---|---|---|
| `SEMANTIC_CACHE_THRESHOLD` | 0.87 | Cosine similarity. ≥0.87 indicates near-identical intent. Empirically tuned — below 0.85 causes false positives on similar-but-different queries. |
| `DEFAULT_TTL` | 1800s (30min) | Balances freshness with API cost. |
| `TEMPORAL_TTL` | 300s (5min) | For time-sensitive queries (stock, news, weather). |
| `STALE_THRESHOLD` | 0.7 × TTL | Entry downgraded to "aged" at 70% of TTL. |
| `MAX_CACHE_ENTRIES` | 1000 | Per-session. LRU eviction when exceeded. |

**Freshness Classification**:

```
FUNCTION classify_freshness(entry: EvidenceCacheEntry) -> str:
    age_seconds = (utcnow() - entry.timestamp).total_seconds()
    ratio = age_seconds / entry.ttl_seconds
    IF ratio < 0.3:   RETURN "live"
    IF ratio < 0.7:   RETURN "recent"
    IF ratio < 1.0:   RETURN "aged"
    RETURN "stale"
```

**Freshness Override Rule**: If a user query contains temporal markers (`today`, `latest`, `current`, `202X`) AND the cached entry's freshness is not "live", force a re-retrieval regardless of semantic similarity.

**Expiration Policy**: Two-phase eviction:
1. **Soft expiry** at TTL: Entry is marked "stale", still usable as fallback if retrieval fails.
2. **Hard eviction** at 2×TTL: Entry purged from cache and FAISS index.

---

## PART 2 — KNOWLEDGE MEMORY ARCHITECTURE

### Tier 1: Session Memory (Short-Term)

**Purpose**: Only retain information relevant to the active topic cluster. Implements semantic compression to minimize token waste.

```python
@dataclass
class SessionMemoryTier:
    topic_cluster_id: str                  # Current active cluster UUID
    topic_embedding: np.ndarray            # Running centroid of cluster
    messages: List[WeightedMessage]        # Sliding window
    max_messages: int = 16                 # Hard limit
    max_tokens: int = 4096                 # Token budget
    compression_threshold: float = 0.65   # Compress when relevance < this

@dataclass
class WeightedMessage:
    role: str                              # user | assistant | system
    content: str
    embedding: np.ndarray
    timestamp: datetime
    relevance_weight: float                # 0.0–1.0, decays over time
    token_count: int
    compressed: bool = False               # True if semantically compressed
    original_content: Optional[str] = None # Pre-compression content
```

**Sliding Window with Semantic Compression**:

```
FUNCTION update_session_memory(memory: SessionMemoryTier, new_msg: WeightedMessage):
    memory.messages.append(new_msg)
    
    // Recalculate topic centroid
    embeddings = [m.embedding for m in memory.messages if m.role == "user"]
    IF embeddings:
        memory.topic_embedding = mean(embeddings)
    
    // Apply decay to all messages
    FOR msg IN memory.messages:
        age = (utcnow() - msg.timestamp).total_seconds()
        msg.relevance_weight = cosine_similarity(msg.embedding, memory.topic_embedding) 
                               * exp(-DECAY_LAMBDA * age / 3600)
    
    // Token pressure check
    total_tokens = sum(m.token_count for m in memory.messages)
    WHILE total_tokens > memory.max_tokens OR len(memory.messages) > memory.max_messages:
        // Find lowest-weight non-compressed message
        candidates = [m for m in memory.messages if not m.compressed and m.relevance_weight < memory.compression_threshold]
        IF candidates:
            target = min(candidates, key=lambda m: m.relevance_weight)
            compressed = semantic_compress(target.content)  // LLM-based 1-sentence summary
            target.original_content = target.content
            target.content = compressed
            target.token_count = count_tokens(compressed)
            target.compressed = True
        ELSE:
            // Remove oldest lowest-weight message entirely
            remove = min(memory.messages, key=lambda m: m.relevance_weight)
            memory.messages.remove(remove)
        total_tokens = sum(m.token_count for m in memory.messages)

    RETURN memory
```

**DECAY_LAMBDA**: 0.15 (15% weight loss per hour of inactivity)

### Tier 2: Persistent Evidence Memory (Mid-Term)

**Purpose**: Store structured evidence objects that survive topic changes within a session. Indexed for fast retrieval by entity, timestamp, and topic.

```python
@dataclass
class EvidenceObject:
    evidence_id: str                       # UUID
    query_origin: str                      # Original query that triggered retrieval
    entity_tags: List[str]                 # Extracted entities (NER)
    topic_tags: List[str]                  # Topic cluster labels
    topic_embedding: np.ndarray            # Topic centroid at time of retrieval
    chunks: List[EvidenceChunk]            # Retrieved evidence chunks
    source_metadata: List[SourceMeta]
    claims_extracted: List[AtomicClaim]    # Decomposed claims
    confidence_score: float
    contradiction_flags: List[str]         # IDs of conflicting evidence objects
    timestamp: datetime
    access_count: int = 0                  # Popularity counter
    last_accessed: datetime = None
```

**Index Structure**:

```
PRIMARY INDEX:    evidence_id → EvidenceObject (hash map)
ENTITY INDEX:     entity_tag → [evidence_id, ...] (inverted index)
TOPIC INDEX:      FAISS index on topic_embedding (approximate nearest neighbor)
TEMPORAL INDEX:   BTree on timestamp (range queries)
COMPOSITE INDEX:  (entity_tag, timestamp) → [evidence_id, ...] (compound)
```

**Storage Backend**: Redis Hashes for hot data, Postgres JSONB for persistence.

```sql
CREATE TABLE IF NOT EXISTS evidence_memory (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_origin TEXT NOT NULL,
    entity_tags JSONB NOT NULL DEFAULT '[]',
    topic_tags JSONB NOT NULL DEFAULT '[]',
    topic_embedding BYTEA,                    -- serialized float32 array
    chunks JSONB NOT NULL DEFAULT '[]',
    source_metadata JSONB NOT NULL DEFAULT '[]',
    claims_extracted JSONB NOT NULL DEFAULT '[]',
    confidence_score FLOAT NOT NULL DEFAULT 0.5,
    contradiction_flags JSONB NOT NULL DEFAULT '[]',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_evidence_memory_entities ON evidence_memory USING GIN(entity_tags);
CREATE INDEX idx_evidence_memory_topics ON evidence_memory USING GIN(topic_tags);
CREATE INDEX idx_evidence_memory_created ON evidence_memory(created_at DESC);
CREATE INDEX idx_evidence_memory_confidence ON evidence_memory(confidence_score DESC);
```

### Tier 3: Knowledge Graph (Long-Term)

**Purpose**: Persistent, growing graph of verified knowledge. Entities, claims, relationships, with temporal validity and conflict resolution.

**Schema**:

```python
@dataclass
class KGEntity:
    entity_id: str                         # UUID
    name: str                              # Canonical name
    aliases: List[str]                     # Alternative names
    entity_type: str                       # person | org | concept | event | location | product
    embedding: np.ndarray                  # Entity embedding
    first_seen: datetime
    last_updated: datetime
    confidence: float                      # 0.0–1.0

@dataclass
class KGClaim:
    claim_id: str                          # UUID
    claim_text: str                        # Atomic claim text
    subject_entity_id: str                 # FK to KGEntity
    predicate: str                         # Relationship type
    object_entity_id: Optional[str]        # FK to KGEntity (if entity-to-entity)
    object_value: Optional[str]            # Literal value (if entity-to-value)
    source_evidence_ids: List[str]         # FK to EvidenceObject
    source_urls: List[str]                 # Direct source URLs
    confidence: float                      # 0.0–1.0
    temporal_validity: TemporalValidity    # When this claim is/was true
    conflict_marker: Optional[str]         # ID of conflicting claim
    status: str                            # "active" | "superseded" | "disputed" | "retracted"
    created_at: datetime
    last_verified: datetime

@dataclass
class TemporalValidity:
    valid_from: Optional[datetime]         # None = always valid
    valid_until: Optional[datetime]        # None = still valid
    is_current: bool                       # True if believed to be currently valid
    temporal_qualifier: str                # "permanent" | "current" | "historical" | "projected"

@dataclass
class KGRelationship:
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str                 # "is_a" | "part_of" | "causes" | "contradicts" | "supports" | ...
    weight: float                          # 0.0–1.0 strength
    evidence_ids: List[str]                # Supporting evidence
    created_at: datetime
```

**Postgres Tables**:

```sql
CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]',
    entity_type VARCHAR(50) NOT NULL,
    embedding BYTEA,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_kg_entities_name ON kg_entities(name);
CREATE INDEX idx_kg_entities_type ON kg_entities(entity_type);

CREATE TABLE IF NOT EXISTS kg_claims (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_text TEXT NOT NULL,
    subject_entity_id UUID NOT NULL,
    predicate VARCHAR(200) NOT NULL,
    object_entity_id UUID,
    object_value TEXT,
    source_evidence_ids JSONB NOT NULL DEFAULT '[]',
    source_urls JSONB NOT NULL DEFAULT '[]',
    confidence FLOAT NOT NULL DEFAULT 0.5,
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,
    is_current BOOLEAN DEFAULT true,
    temporal_qualifier VARCHAR(50) DEFAULT 'current',
    conflict_marker UUID,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_verified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_kg_claims_subject ON kg_claims(subject_entity_id);
CREATE INDEX idx_kg_claims_object ON kg_claims(object_entity_id);
CREATE INDEX idx_kg_claims_status ON kg_claims(status);
CREATE INDEX idx_kg_claims_predicate ON kg_claims(predicate);

CREATE TABLE IF NOT EXISTS kg_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL,
    target_entity_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    weight FLOAT NOT NULL DEFAULT 0.5,
    evidence_ids JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_kg_rel_source ON kg_relationships(source_entity_id);
CREATE INDEX idx_kg_rel_target ON kg_relationships(target_entity_id);
CREATE INDEX idx_kg_rel_type ON kg_relationships(relationship_type);
```

**Update Rules**:

```
FUNCTION update_knowledge_graph(new_evidence: EvidenceObject, graph: KnowledgeGraph):
    // Step 1: Extract entities from evidence chunks
    entities = extract_entities(new_evidence.chunks)  // NER
    
    FOR entity IN entities:
        existing = graph.find_entity(entity.name, entity.type)
        IF existing:
            existing.last_updated = utcnow()
            existing.confidence = ema(existing.confidence, entity.confidence, alpha=0.3)
            // Merge aliases
            existing.aliases = list(set(existing.aliases + entity.aliases))
        ELSE:
            graph.add_entity(entity)
    
    // Step 2: Extract atomic claims
    claims = decompose_claims(new_evidence.chunks)
    
    FOR claim IN claims:
        conflicting = graph.find_conflicting_claims(claim)
        IF conflicting:
            // Conflict resolution
            resolve_conflict(claim, conflicting, graph)
        ELSE:
            graph.add_claim(claim)
    
    // Step 3: Build/update relationships
    relationships = infer_relationships(entities, claims)
    FOR rel IN relationships:
        existing_rel = graph.find_relationship(rel.source, rel.target, rel.type)
        IF existing_rel:
            existing_rel.weight = max(existing_rel.weight, rel.weight)
            existing_rel.evidence_ids.append(new_evidence.evidence_id)
        ELSE:
            graph.add_relationship(rel)
```

**Conflict Resolution Algorithm**:

```
FUNCTION resolve_conflict(new_claim: KGClaim, existing_claims: List[KGClaim], graph: KnowledgeGraph):
    FOR existing IN existing_claims:
        // Score comparison
        new_score = compute_claim_strength(new_claim)
        existing_score = compute_claim_strength(existing)
        
        // Temporal resolution: newer evidence supersedes if both are temporal
        IF new_claim.temporal_validity.is_current AND NOT existing.temporal_validity.is_current:
            existing.status = "superseded"
            existing.temporal_validity.valid_until = utcnow()
            new_claim.status = "active"
            CONTINUE
        
        // Source quality resolution
        IF new_score > existing_score * 1.3:  // 30% stronger evidence
            existing.status = "superseded"
            new_claim.status = "active"
        ELIF existing_score > new_score * 1.3:
            new_claim.status = "disputed"
            new_claim.conflict_marker = existing.claim_id
        ELSE:
            // Genuine dispute — mark both
            new_claim.status = "disputed"
            existing.status = "disputed"
            new_claim.conflict_marker = existing.claim_id
            existing.conflict_marker = new_claim.claim_id
    
    graph.add_claim(new_claim)


FUNCTION compute_claim_strength(claim: KGClaim) -> float:
    source_count = len(claim.source_evidence_ids)
    avg_source_reliability = mean([get_source_reliability(url) for url in claim.source_urls])
    recency_bonus = 1.0 / (1.0 + days_since(claim.last_verified) / 30)
    
    strength = (
        0.4 * min(source_count / 5, 1.0)     // Source quantity (capped at 5)
        + 0.35 * avg_source_reliability        // Source quality
        + 0.15 * claim.confidence              // Extraction confidence
        + 0.10 * recency_bonus                 // Freshness
    )
    RETURN clamp(strength, 0.0, 1.0)
```

---

## PART 3 — FEEDING BACK INTO MODEL CONTEXT

### Context Injection Strategy

Before each model call, the system compiles a structured context block that replaces naive full-history injection.

```
FUNCTION compile_model_context(
    session: SessionMemoryTier,
    evidence_memory: EvidenceMemory,
    knowledge_graph: KnowledgeGraph,
    current_query: str,
    max_context_tokens: int = 6144
) -> CompiledContext:

    // Step 1: Determine topic cluster
    query_embedding = embed(current_query)
    topic_cluster = identify_topic_cluster(query_embedding, session.topic_embedding)
    
    // Step 2: Retrieve relevant evidence (from Tier 2)
    relevant_evidence = evidence_memory.query(
        topic_embedding=topic_cluster.centroid,
        entity_filter=extract_entities_fast(current_query),
        k=10,
        min_confidence=0.4
    )
    
    // Step 3: Retrieve relevant graph nodes (from Tier 3)
    relevant_entities = knowledge_graph.get_related_entities(
        query_embedding=query_embedding,
        k=15
    )
    relevant_claims = knowledge_graph.get_claims_for_entities(
        entity_ids=[e.entity_id for e in relevant_entities],
        status_filter=["active", "disputed"],
        min_confidence=0.3
    )
    conflict_flags = [c for c in relevant_claims if c.status == "disputed"]
    
    // Step 4: Compile structured context blocks
    context = CompiledContext()
    token_budget = max_context_tokens
    
    // Block 1: Verified Evidence (highest priority)
    verified_block = format_verified_evidence(relevant_evidence, max_tokens=token_budget * 0.40)
    context.add_block("verified_evidence", verified_block)
    token_budget -= count_tokens(verified_block)
    
    // Block 2: Recent Updates (new information since last query)
    updates_block = format_recent_updates(
        evidence_memory.get_since(session.last_query_time),
        max_tokens=token_budget * 0.20
    )
    context.add_block("recent_updates", updates_block)
    token_budget -= count_tokens(updates_block)
    
    // Block 3: Known Conflicts (critical for preventing hallucination)
    IF conflict_flags:
        conflicts_block = format_conflicts(conflict_flags, max_tokens=token_budget * 0.15)
        context.add_block("known_conflicts", conflicts_block)
        token_budget -= count_tokens(conflicts_block)
    
    // Block 4: Confidence Metrics
    confidence_block = format_confidence_summary(relevant_evidence, relevant_claims)
    context.add_block("confidence_metrics", confidence_block)
    token_budget -= count_tokens(confidence_block)
    
    // Block 5: Recent conversation (only high-weight messages)
    conversation_block = format_conversation(
        [m for m in session.messages if m.relevance_weight > 0.3],
        max_tokens=token_budget
    )
    context.add_block("conversation", conversation_block)
    
    RETURN context
```

**Compiled Context Format** (injected as system message):

```
[Verified Evidence]
• {claim_text} (confidence: 0.87, sources: 3, domain: arxiv.org, nature.com) 
• {claim_text} (confidence: 0.72, sources: 2, domain: reuters.com)

[Recent Updates]  
• {entity} updated: {change_description} (2h ago, source: bbc.com)

[Known Conflicts]
⚠ DISPUTED: "{claim_a}" vs "{claim_b}" — sources disagree (severity: 0.7)

[Confidence Metrics]
Overall evidence confidence: 0.78
Source agreement: 0.65
Graph coverage: 12 entities, 8 active claims
Freshness: 85% live/recent
```

**Prevention Rules**:

1. **No full history re-injection**: Only messages with `relevance_weight > 0.3` enter context.
2. **No irrelevant context**: Evidence must pass topic cluster similarity check (`> 0.5`).
3. **Token explosion guard**: Hard budget of `max_context_tokens` with proportional allocation per block.

---

## PART 4 — FOLLOW-UP CONTEXT STABILITY

### A. Topic Boundary Detection

```
FUNCTION detect_topic_boundary(
    current_query_embedding: np.ndarray,
    session: SessionMemoryTier,
    threshold: float = 0.40
) -> TopicBoundaryResult:

    // Compute similarity to current topic cluster centroid
    sim = cosine_similarity(current_query_embedding, session.topic_embedding)
    
    // Compute similarity to last N user messages
    recent_user_msgs = [m for m in session.messages[-5:] if m.role == "user"]
    IF recent_user_msgs:
        recent_centroid = mean([m.embedding for m in recent_user_msgs])
        local_sim = cosine_similarity(current_query_embedding, recent_centroid)
    ELSE:
        local_sim = sim
    
    // Combined drift score (weighted)
    drift_score = 1.0 - (0.6 * sim + 0.4 * local_sim)
    
    IF drift_score > (1.0 - threshold):  // drift > 0.60 → topic shift
        RETURN TopicBoundaryResult(
            is_shift=True,
            drift_score=drift_score,
            action="archive_and_reset",
            previous_cluster_id=session.topic_cluster_id
        )
    ELIF drift_score > (1.0 - threshold) * 0.7:  // Moderate drift
        RETURN TopicBoundaryResult(
            is_shift=False,
            drift_score=drift_score,
            action="expand_cluster",
            previous_cluster_id=session.topic_cluster_id
        )
    ELSE:
        RETURN TopicBoundaryResult(
            is_shift=False,
            drift_score=drift_score,
            action="continue",
            previous_cluster_id=session.topic_cluster_id
        )


FUNCTION handle_topic_shift(session: SessionMemoryTier, evidence_memory: EvidenceMemory):
    // Archive current cluster
    archive = TopicArchive(
        cluster_id=session.topic_cluster_id,
        topic_embedding=session.topic_embedding,
        messages=session.messages.copy(),
        evidence_refs=[e.evidence_id for e in evidence_memory.get_by_topic(session.topic_cluster_id)],
        archived_at=utcnow()
    )
    session.archives.append(archive)
    
    // Reset session memory for new topic
    session.topic_cluster_id = new_uuid()
    session.topic_embedding = None  // Will be set from new query
    session.messages = []           // Fresh start
    
    // Keep only last message for transition context
    // (prevents total context loss on shift)
```

### B. Follow-Up Anchoring

```
FUNCTION anchor_followup(
    current_intent: IntentHash,
    session: SessionMemoryTier,
    intent_history: List[IntentHash]
) -> FollowUpAnchor:

    IF NOT intent_history:
        RETURN FollowUpAnchor(anchor_type="initial", linked_intent=None)
    
    last_intent = intent_history[-1]
    
    // Check if this is a follow-up (high similarity to last intent)
    sim = cosine_similarity(current_intent.embedding, last_intent.embedding)
    
    IF sim > 0.70:  // Likely follow-up
        // Verify scope match: same topic cluster?
        scope_match = session.topic_cluster_id == last_intent.session_id
        
        IF scope_match OR sim > 0.85:
            RETURN FollowUpAnchor(
                anchor_type="linked",
                linked_intent=last_intent.exact_hash,
                similarity=sim,
                reuse_evidence=True  // Reuse previous evidence
            )
        ELSE:
            // Scope mismatch — might be follow-up to archived topic
            RETURN FollowUpAnchor(
                anchor_type="ambiguous",
                linked_intent=last_intent.exact_hash,
                similarity=sim,
                reuse_evidence=False,
                clarification_needed=True,
                clarification_prompt=f"Are you continuing the discussion about '{last_intent.canonical}' or starting a new topic?"
            )
    ELSE:
        RETURN FollowUpAnchor(anchor_type="new_topic", linked_intent=None)
```

### C. Context Decay Logic

```
FUNCTION apply_context_decay(session: SessionMemoryTier, current_time: datetime):
    // Exponential decay with topic relevance modulation
    
    FOR msg IN session.messages:
        age_hours = (current_time - msg.timestamp).total_seconds() / 3600
        
        // Base time decay
        time_decay = exp(-DECAY_LAMBDA * age_hours)  // DECAY_LAMBDA = 0.15
        
        // Topic relevance (cosine to current topic centroid)
        IF session.topic_embedding IS NOT None:
            topic_relevance = cosine_similarity(msg.embedding, session.topic_embedding)
        ELSE:
            topic_relevance = 0.5
        
        // Role-based persistence boost
        role_boost = 1.0
        IF msg.role == "system": role_boost = 1.5   // System messages persist longer
        IF msg.compressed: role_boost *= 0.8         // Compressed messages lose weight
        
        // Final weight
        msg.relevance_weight = clamp(
            time_decay * topic_relevance * role_boost,
            0.0, 1.0
        )
    
    // Prune messages below minimum weight threshold
    PRUNE_THRESHOLD = 0.05
    session.messages = [m for m in session.messages if m.relevance_weight >= PRUNE_THRESHOLD]
```

**Drift Detection Algorithm Summary**:

| Metric | Threshold | Action |
|---|---|---|
| `cosine(query, topic_centroid) < 0.40` | Hard shift | Archive + reset |
| `0.40 ≤ cosine < 0.58` | Soft drift | Expand cluster |
| `cosine ≥ 0.58` | On-topic | Continue |
| `followup_sim > 0.85` | Strong follow-up | Reuse evidence, same scope |
| `0.70 < followup_sim ≤ 0.85` | Probable follow-up | Verify scope |
| `followup_sim ≤ 0.70` | New topic | New intent chain |

---

## PART 5 — AUTONOMOUS LIVE INGESTION

### Ingestion Daemon Architecture

```
CLASS LiveIngestionDaemon:
    // Runs as an async background task, independent of user interaction

    PROPERTIES:
        source_watchers: List[SourceWatcher]     // Configured data sources
        ingestion_queue: AsyncQueue              // Bounded queue, max 500 items
        rate_limiter: TokenBucketLimiter         // 10 requests/minute per source
        trust_scorer: SourceTrustScorer
        poison_guard: AntiPoisonGuard
        knowledge_graph: KnowledgeGraph
        evidence_memory: EvidenceMemory
        running: bool = False
        last_run: Dict[str, datetime] = {}       // Per-source last execution time

    FUNCTION start():
        running = True
        // Launch concurrent watchdogs
        await gather(
            _schedule_loop(),
            _event_listener(),
            _queue_processor()
        )

    FUNCTION _schedule_loop():
        // Cron-style scheduled ingestion
        WHILE running:
            FOR watcher IN source_watchers:
                IF should_run(watcher):
                    IF rate_limiter.acquire(watcher.source_id):
                        ingestion_queue.put(IngestionTask(watcher))
                        last_run[watcher.source_id] = utcnow()
            await sleep(60)  // Check every 60s

    FUNCTION _event_listener():
        // Webhook/SSE/WebSocket triggered ingestion
        WHILE running:
            event = await event_bus.receive()
            IF event.type == "new_data":
                watcher = get_watcher_for_event(event)
                IF rate_limiter.acquire(watcher.source_id):
                    ingestion_queue.put(IngestionTask(watcher, event=event))

    FUNCTION _queue_processor():
        // Process ingestion tasks sequentially
        WHILE running:
            task = await ingestion_queue.get()
            TRY:
                await _process_ingestion(task)
            EXCEPT Exception as e:
                log.error(f"Ingestion failed: {e}")
            FINALLY:
                ingestion_queue.task_done()
```

**Diff-Based Update Engine**:

```
FUNCTION _process_ingestion(task: IngestionTask):
    // Step 1: Fetch new data
    new_data = await task.watcher.fetch()
    
    // Step 2: Content hash for diff detection
    new_hash = sha256(new_data.content)
    stored_hash = await redis.get(f"content_hash:{task.watcher.source_id}")
    
    IF new_hash == stored_hash:
        // No change — skip entirely
        RETURN
    
    // Step 3: Anti-poisoning check BEFORE processing
    IF NOT poison_guard.check(new_data):
        log.warning(f"Poisoning detected in {task.watcher.source_id}")
        RETURN
    
    // Step 4: Chunk and embed
    chunks = chunk_text(new_data.content, chunk_size=512, overlap=64)
    embeddings = batch_embed(chunks)
    
    // Step 5: Compare against stored knowledge
    FOR i, chunk IN enumerate(chunks):
        existing = knowledge_graph.find_similar_claims(embeddings[i], threshold=0.90)
        
        IF existing:
            // Update or contradiction detection
            FOR claim IN existing:
                IF is_contradiction(chunk.text, claim.claim_text):
                    // Mark conflict
                    claim.status = "disputed"
                    claim.conflict_marker = new_claim_id
                    log.info(f"Contradiction detected: {claim.claim_id}")
                ELIF is_update(chunk.text, claim.claim_text):
                    // Supersede old claim
                    claim.status = "superseded"
                    claim.temporal_validity.valid_until = utcnow()
        
        // Store new evidence
        evidence_obj = create_evidence_object(chunk, embeddings[i], task.watcher.source_meta)
        evidence_memory.store(evidence_obj)
        
        // Update knowledge graph
        update_knowledge_graph(evidence_obj, knowledge_graph)
    
    // Step 6: Recalculate affected topic centroids
    affected_topics = get_affected_topics(chunks, knowledge_graph)
    FOR topic IN affected_topics:
        topic.centroid = recalculate_centroid(topic)
    
    // Step 7: Store new content hash
    await redis.set(f"content_hash:{task.watcher.source_id}", new_hash, ttl=86400)
```

**Source Trust Scoring**:

```
CLASS SourceTrustScorer:
    FUNCTION score(source: SourceMeta, historical_accuracy: float = None) -> float:
        base_score = TRUSTED_DOMAINS.get(source.domain, 0.5)
        
        // Historical accuracy override (from past verification)
        IF historical_accuracy IS NOT None:
            base_score = 0.6 * base_score + 0.4 * historical_accuracy
        
        // Recency bonus
        age_days = (utcnow() - source.retrieved_at).days
        recency = 1.0 / (1.0 + age_days / 30)
        
        // Penalize if source previously flagged
        penalty = 0.0
        IF source.domain IN flagged_sources:
            penalty = 0.3
        
        RETURN clamp(base_score * recency - penalty, 0.0, 1.0)
```

**Anti-Poisoning Safeguards**:

```
CLASS AntiPoisonGuard:
    FUNCTION check(data: IngestedData) -> bool:
        // 1. Content anomaly detection
        IF len(data.content) < 50:
            RETURN False  // Suspiciously short
        IF len(data.content) > 1_000_000:
            RETURN False  // Suspiciously large
        
        // 2. Injection pattern detection
        injection_patterns = [
            r"ignore previous instructions",
            r"you are now",
            r"disregard all",
            r"new system prompt",
            r"<script>",
            r"javascript:",
        ]
        FOR pattern IN injection_patterns:
            IF regex_search(pattern, data.content, IGNORECASE):
                RETURN False
        
        // 3. Source velocity check (too many updates = suspicious)
        recent_updates = get_update_count(data.source_id, window=timedelta(hours=1))
        IF recent_updates > MAX_UPDATES_PER_HOUR:  // 20
            RETURN False
        
        // 4. Semantic drift detection (content suddenly changes topic dramatically)
        IF data.source_id IN known_sources:
            historical_centroid = get_historical_centroid(data.source_id)
            new_embedding = embed(data.content[:1000])
            drift = 1.0 - cosine_similarity(new_embedding, historical_centroid)
            IF drift > 0.7:  // Dramatic content shift
                log.warning(f"Semantic drift detected for {data.source_id}: {drift}")
                RETURN False
        
        RETURN True
```

**Rate Limiting** (Token Bucket):

```
CLASS TokenBucketLimiter:
    // Per source: 10 tokens/minute, bucket size = 10
    FUNCTION acquire(source_id: str) -> bool:
        bucket = buckets.get(source_id, TokenBucket(capacity=10, refill_rate=10/60))
        RETURN bucket.consume(1)
```

**Loop Prevention**: 
- Each ingestion task has a unique `task_id` tracked in a Set.
- If `task_id` already processed within the dedup window (5 min), skip.
- Queue is bounded (max 500), oldest tasks evicted if full.

---

## PART 6 — REDUNDANCY ELIMINATION

### Semantic Cache Architecture

The entire retrieval path is gated by a multi-level cache:

```
FUNCTION retrieve_with_cache(query: str, session: Session) -> RetrievalResult:
    intent = hash_intent(query, session.id)
    
    // Level 1: Exact hash match (O(1) Redis lookup)
    l1_hit = redis.get(f"evidence:{intent.exact_hash}")
    IF l1_hit:
        metrics.record("cache_hit_l1")
        RETURN deserialize(l1_hit)
    
    // Level 2: Semantic similarity match (FAISS ANN search)
    l2_candidates = evidence_cache_index.search(intent.embedding, k=3)
    FOR candidate IN l2_candidates:
        sim = cosine_similarity(intent.embedding, candidate.embedding)
        IF sim >= 0.87 AND NOT is_expired(candidate) AND NOT needs_freshness(intent, candidate):
            metrics.record("cache_hit_l2")
            // Promote to L1 for future exact matches
            redis.set(f"evidence:{intent.exact_hash}", serialize(candidate), ttl=candidate.remaining_ttl)
            RETURN candidate
    
    // Level 3: Knowledge graph lookup (skip retrieval if graph coverage sufficient)
    graph_coverage = knowledge_graph.compute_coverage(query, intent.embedding)
    IF graph_coverage.score >= 0.80 AND NOT intent.temporal_flag:
        metrics.record("cache_hit_l3_graph")
        RETURN RetrievalResult(
            source="knowledge_graph",
            evidence=graph_coverage.claims,
            retrieval_skipped=True,
            coverage_score=graph_coverage.score
        )
    
    // Full cache miss — execute live retrieval
    metrics.record("cache_miss")
    result = await live_retrieve(query, intent)
    
    // Store in cache
    cache_entry = build_cache_entry(intent, result)
    redis.set(f"evidence:{intent.exact_hash}", serialize(cache_entry), ttl=cache_entry.ttl_seconds)
    evidence_cache_index.add(cache_entry)
    
    // Feed back into knowledge graph
    schedule_graph_update(result)
    
    RETURN result
```

**Expiry Policy**:

| Entry Type | Default TTL | Temporal TTL | Stale Grace | Hard Evict |
|---|---|---|---|---|
| Factual query | 1800s (30m) | 300s (5m) | +300s | +600s |
| Analytical query | 3600s (1h) | N/A | +600s | +1200s |
| Conversational | N/A (not cached) | N/A | N/A | N/A |
| Creative | N/A (not cached) | N/A | N/A | N/A |

**Freshness Override Logic**:

```
FUNCTION needs_freshness(intent: IntentHash, cached: EvidenceCacheEntry) -> bool:
    // Rule 1: Temporal queries always need fresh data
    IF intent.temporal_flag:
        IF classify_freshness(cached) != "live":
            RETURN True
    
    // Rule 2: Breaking news patterns
    IF intent.intent_type == "temporal":
        RETURN True
    
    // Rule 3: High retrieval probability + stale cache
    IF intent.retrieval_p > 0.8 AND classify_freshness(cached) in ("aged", "stale"):
        RETURN True
    
    RETURN False
```

---

## PART 7 — HALLUCINATION CONTROL

### Claim-to-Source Verification Pipeline

```
FUNCTION verify_response(
    response_text: str,
    evidence: List[EvidenceChunk],
    knowledge_claims: List[KGClaim],
    mode: str
) -> VerificationResult:

    // Step 1: Decompose response into atomic sentences
    sentences = split_into_sentences(response_text)
    
    // Step 2: Classify each sentence
    verified_sentences = []
    unsupported_sentences = []
    opinion_sentences = []
    
    FOR sentence IN sentences:
        sent_embedding = embed(sentence)
        classification = classify_sentence_type(sentence)
        
        IF classification in ("factual_claim", "causal_claim", "statistical_claim"):
            // Must be verifiable
            support = find_supporting_evidence(sent_embedding, evidence, knowledge_claims)
            
            IF support.max_similarity >= 0.75:
                verified_sentences.append(VerifiedSentence(
                    text=sentence,
                    support_score=support.max_similarity,
                    source_refs=support.source_refs,
                    status="verified"
                ))
            ELIF support.max_similarity >= 0.55:
                verified_sentences.append(VerifiedSentence(
                    text=sentence,
                    support_score=support.max_similarity,
                    source_refs=support.source_refs,
                    status="partially_supported"
                ))
            ELSE:
                unsupported_sentences.append(UnsupportedSentence(
                    text=sentence,
                    best_match_score=support.max_similarity,
                    status="unsupported"
                ))
        
        ELIF classification in ("opinion", "hedged_claim"):
            opinion_sentences.append(sentence)
        
        // Else: procedural, connective, etc. — pass through
    
    // Step 3: Compute coverage
    total_factual = len(verified_sentences) + len(unsupported_sentences)
    coverage = len(verified_sentences) / total_factual IF total_factual > 0 ELSE 1.0
    
    // Step 4: Decision
    IF coverage < COVERAGE_THRESHOLD:  // 0.60
        RETURN VerificationResult(
            status="regenerate",
            coverage=coverage,
            unsupported=unsupported_sentences,
            message="Insufficient evidence coverage. Regeneration required."
        )
    
    // Step 5: Strip unsupported factual claims
    clean_response = response_text
    FOR unsupported IN unsupported_sentences:
        IF mode == "evidence":
            // In evidence mode, replace with explicit uncertainty
            clean_response = clean_response.replace(
                unsupported.text,
                f"[Unverified: {unsupported.text}]"
            )
        ELSE:
            // In standard mode, soften the language
            clean_response = clean_response.replace(
                unsupported.text,
                soften_claim(unsupported.text)  // "X is Y" → "X may be Y (unverified)"
            )
    
    RETURN VerificationResult(
        status="verified",
        coverage=coverage,
        clean_response=clean_response,
        verified_count=len(verified_sentences),
        unsupported_count=len(unsupported_sentences),
        traceability_map=build_traceability_map(verified_sentences)
    )


FUNCTION find_supporting_evidence(
    sent_embedding: np.ndarray,
    evidence: List[EvidenceChunk],
    claims: List[KGClaim]
) -> SupportResult:
    
    best_similarity = 0.0
    source_refs = []
    
    // Check against evidence chunks
    FOR chunk IN evidence:
        sim = cosine_similarity(sent_embedding, chunk.embedding)
        IF sim > best_similarity:
            best_similarity = sim
            source_refs = [chunk.source_url]
        ELIF sim > 0.60:
            source_refs.append(chunk.source_url)
    
    // Check against knowledge graph claims
    FOR claim IN claims:
        claim_embedding = embed(claim.claim_text)
        sim = cosine_similarity(sent_embedding, claim_embedding)
        IF sim > best_similarity:
            best_similarity = sim
            source_refs = claim.source_urls
        ELIF sim > 0.60:
            source_refs.extend(claim.source_urls)
    
    RETURN SupportResult(
        max_similarity=best_similarity,
        source_refs=list(set(source_refs))
    )
```

**Confidence Scoring**:

```
FUNCTION compute_response_confidence(verification: VerificationResult) -> float:
    IF verification.status == "regenerate":
        RETURN 0.0
    
    base = verification.coverage
    
    // Penalty for partially supported claims
    partial_count = sum(1 for v in verification.verified if v.status == "partially_supported")
    partial_penalty = partial_count * 0.05
    
    // Bonus for high-reliability sources
    high_rel_count = sum(1 for v in verification.verified if v.avg_source_reliability > 0.8)
    reliability_bonus = min(high_rel_count * 0.03, 0.15)
    
    RETURN clamp(base - partial_penalty + reliability_bonus, 0.0, 1.0)
```

**Regeneration Trigger**:

```
FUNCTION handle_low_coverage(
    query: str,
    evidence: List[EvidenceChunk],
    attempt: int = 1,
    max_attempts: int = 2
) -> str:
    IF attempt >= max_attempts:
        // Fail gracefully — acknowledge uncertainty
        RETURN generate_with_uncertainty_acknowledgment(query, evidence)
    
    // Try additional retrieval with modified query
    expanded_query = expand_query(query)  // Add context terms
    new_evidence = await live_retrieve(expanded_query)
    combined_evidence = deduplicate(evidence + new_evidence)
    
    // Regenerate with more context
    response = await generate(query, combined_evidence)
    verification = verify_response(response, combined_evidence)
    
    IF verification.coverage >= COVERAGE_THRESHOLD:
        RETURN verification.clean_response
    ELSE:
        RETURN handle_low_coverage(query, combined_evidence, attempt + 1)
```

---

## PART 8 — PERFORMANCE OPTIMIZATION

### Embedding Pre-Computation

```
CLASS EmbeddingPrecomputer:
    // Batch embed on write, not on read
    
    FUNCTION on_message_received(message: str) -> np.ndarray:
        // Embed immediately, store alongside message
        embedding = self.model.encode(message, normalize_embeddings=True)
        RETURN embedding
    
    FUNCTION batch_precompute(texts: List[str]) -> List[np.ndarray]:
        // Batch encoding is 3-5x faster than sequential
        RETURN self.model.encode(texts, normalize_embeddings=True, batch_size=32)
    
    FUNCTION warm_cache(session: Session):
        // Pre-compute embeddings for expected follow-up topics
        recent_topics = extract_topic_keywords(session.messages[-5:])
        anticipated_queries = expand_topic_variations(recent_topics)
        embeddings = self.batch_precompute(anticipated_queries)
        FOR q, emb IN zip(anticipated_queries, embeddings):
            warm_cache.store(q, emb, ttl=300)
```

### Topic Centroid Indexing

```
CLASS TopicCentroidIndex:
    // FAISS IVF index for fast topic lookup
    
    FUNCTION __init__(self):
        self.dimension = 384
        self.nlist = 32                     // Number of Voronoi cells
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.id_map = {}                    // FAISS ID → topic_cluster_id
    
    FUNCTION add_centroid(topic_id: str, centroid: np.ndarray):
        normalized = centroid / np.linalg.norm(centroid)
        idx = self.index.ntotal
        self.index.add(normalized.reshape(1, -1))
        self.id_map[idx] = topic_id
    
    FUNCTION search(query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        normalized = query_embedding / np.linalg.norm(query_embedding)
        distances, indices = self.index.search(normalized.reshape(1, -1), k)
        results = []
        FOR i IN range(k):
            IF indices[0][i] >= 0:
                topic_id = self.id_map.get(indices[0][i])
                IF topic_id:
                    results.append((topic_id, float(distances[0][i])))
        RETURN results
```

### Retrieval Warm Cache

```
FUNCTION warm_retrieval_cache(session: Session, evidence_memory: EvidenceMemory):
    // Predict likely next queries based on conversation trajectory
    
    IF len(session.messages) < 3:
        RETURN  // Not enough context to predict
    
    recent_topics = extract_entities_fast(session.messages[-3:])
    
    FOR topic IN recent_topics:
        // Pre-fetch evidence for predicted follow-up topics
        cached = evidence_memory.query_by_entity(topic)
        IF NOT cached:
            // Schedule background pre-fetch
            schedule_background_task(
                prefetch_evidence,
                args=(topic,),
                priority="low"
            )
```

### Async Ingestion Pipeline

```
CLASS AsyncIngestionPipeline:
    FUNCTION __init__(self):
        self.embed_pool = ThreadPoolExecutor(max_workers=4)
        self.process_queue = asyncio.Queue(maxsize=100)
    
    ASYNC FUNCTION ingest(data: IngestedData):
        // Stage 1: Chunk (CPU-bound → thread pool)
        chunks = await run_in_executor(self.embed_pool, chunk_text, data.content)
        
        // Stage 2: Embed (CPU-bound → thread pool, batched)
        embeddings = await run_in_executor(self.embed_pool, batch_embed, chunks)
        
        // Stage 3: Score and store (IO-bound → async)
        scored = await score_sources(chunks, embeddings)
        await evidence_memory.store_batch(scored)
        
        // Stage 4: Graph update (async, can be deferred)
        await self.process_queue.put(GraphUpdateTask(scored))
```

### Parallel Evidence Scoring

```
ASYNC FUNCTION score_evidence_parallel(chunks: List[EvidenceChunk]) -> List[ScoredChunk]:
    // Score reliability, relevance, and freshness in parallel
    
    tasks = []
    FOR chunk IN chunks:
        tasks.append(asyncio.create_task(score_single_chunk(chunk)))
    
    results = await asyncio.gather(*tasks)
    
    // Sort by composite score
    results.sort(key=lambda r: r.composite_score, reverse=True)
    RETURN results

ASYNC FUNCTION score_single_chunk(chunk: EvidenceChunk) -> ScoredChunk:
    reliability = score_domain(chunk.source_url)
    relevance = chunk.similarity_to_query  // Already computed during retrieval
    freshness = compute_freshness_score(chunk.retrieved_at)
    
    composite = (
        0.45 * relevance +
        0.30 * reliability +
        0.15 * freshness +
        0.10 * chunk.content_density  // Information density heuristic
    )
    
    RETURN ScoredChunk(chunk=chunk, composite_score=composite)
```

### Token Optimization Strategy

```
FUNCTION optimize_token_budget(
    context: CompiledContext,
    model_max_tokens: int,
    response_reserve: int = 2048
) -> CompiledContext:
    
    available = model_max_tokens - response_reserve
    
    // Priority-based allocation
    ALLOCATIONS = {
        "system_prompt":       0.10,   // 10% — fixed system instructions
        "verified_evidence":   0.35,   // 35% — primary evidence
        "recent_updates":      0.10,   // 10% — new information
        "known_conflicts":     0.05,   //  5% — dispute markers
        "confidence_metrics":  0.05,   //  5% — confidence summary
        "conversation":        0.25,   // 25% — conversation history
        "user_query":          0.10,   // 10% — current query + context
    }
    
    FOR block_name, ratio IN ALLOCATIONS.items():
        block = context.get_block(block_name)
        IF block:
            max_tokens = int(available * ratio)
            IF count_tokens(block.content) > max_tokens:
                block.content = truncate_intelligently(block.content, max_tokens)
                // Truncation preserves sentence boundaries and highest-relevance content
    
    RETURN context
```

---

## PART 9 — MEMORY FLOW DIAGRAM

```
USER QUERY
    │
    ▼
┌────────────────┐
│ INTENT HASHER  │──── exact_hash ──── session_cache
│ (SHA3 + embed) │                         │
└───────┬────────┘                         │
        │                                  │
        ▼                                  ▼
┌────────────────┐              ┌─────────────────┐
│ EVIDENCE CACHE │◄────────────▶│ CACHE HIT CHECK │
│ L1: Redis      │              │ L1: Exact hash  │
│ L2: FAISS ANN  │              │ L2: Semantic sim │
│ L3: KG lookup  │              │ L3: Graph cover │
└───────┬────────┘              └────────┬────────┘
        │                                │
        │ MISS                           │ HIT
        ▼                                ▼
┌────────────────┐              ┌─────────────────┐
│ LIVE RETRIEVAL │              │ CACHED EVIDENCE  │
│ Tavily API     │              │ (skip retrieval) │
│ Score + Dedup  │              └────────┬────────┘
│ Embed chunks   │                       │
└───────┬────────┘                       │
        │                                │
        ├────────────────────────────────┤
        ▼                                │
┌────────────────┐                       │
│ FEEDBACK LOOP  │                       │
│ Store evidence │                       │
│ Update cache   │                       │
│ Update KG      │                       │
│ Update centroid│                       │
└───────┬────────┘                       │
        │                                │
        ├────────────────────────────────┤
        ▼                                ▼
┌─────────────────────────────────────────────┐
│            TOPIC BOUNDARY CHECK             │
│ cosine(query, cluster) < 0.40 → SHIFT      │
│ follow-up anchor check                      │
│ context decay application                   │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│            CONTEXT COMPILER                 │
│ [Verified Evidence]  — 35% token budget     │
│ [Recent Updates]     — 10% token budget     │
│ [Known Conflicts]    —  5% token budget     │
│ [Confidence Metrics] —  5% token budget     │
│ [Conversation]       — 25% token budget     │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              MODEL CALL                     │
│ system + compiled_context + query           │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│         HALLUCINATION CONTROL GATE          │
│ Decompose → Verify → Score → Strip/Regen   │
│ coverage < 0.60 → regenerate                │
│ unsupported claims → soften/tag             │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              RESPONSE OUTPUT                │
│ + Traceability map                          │
│ + Confidence score                          │
│ + Source citations                          │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│            FEEDBACK INTO MEMORY             │
│ Update Session Memory (Tier 1)              │
│ Store new evidence (Tier 2)                 │
│ Update Knowledge Graph (Tier 3)             │
│ Update intent history                       │
│ Recalculate topic centroid                  │
└─────────────────────────────────────────────┘
```

---

## FAILURE MODE HANDLING

### Critical Failure Modes and Mitigations

| Failure Mode | Detection | Mitigation | Fallback |
|---|---|---|---|
| **Redis unavailable** | Connection timeout > 2s | Switch to in-memory LRU cache (1000 entries max) | All cache operations become no-ops; live retrieval every time |
| **FAISS index corrupted** | Index load fails or returns dimension mismatch | Rebuild from Tier 2 evidence memory (Postgres) | Fall back to brute-force cosine similarity on cached embeddings |
| **Tavily API down** | HTTP 5xx or timeout > 10s | Retry 2x with exponential backoff (2s, 4s) | Return "offline mode" response using only cached evidence + knowledge graph |
| **Embedding model OOM** | CUDA/CPU OOM exception | Reduce batch size from 32 → 8 → 1 | Cache last-known embedding; use TF-IDF fallback for similarity |
| **Knowledge graph inconsistency** | Circular conflict markers | Run conflict resolution sweep (O(n) on disputed claims) | Quarantine all disputed claims; serve only "active" status claims |
| **Ingestion loop detected** | Same `task_id` processed 3x in 5min | Kill task, add source to cooldown list (30min) | Skip source ingestion; alert admin |
| **Token budget exceeded** | `count_tokens(context) > max_context_tokens` | Progressive truncation: trim conversation → trim evidence → trim conflicts | Use only system prompt + user query + top-1 evidence chunk |
| **Coverage below threshold** | `verification.coverage < 0.60` after 2 regenerations | Acknowledge uncertainty explicitly in response | Return partial response with clear "[insufficient evidence]" markers |
| **Hallucination detected** | Unsupported factual claims in response | Strip or soften unsupported sentences | In evidence mode: refuse to output unverified claims. In standard mode: hedge language |
| **Topic boundary false positive** | User says "continuing from before" but system archived | Re-score against archived clusters; restore if sim > 0.6 | Always keep last 2 archived clusters recoverable for 1 hour |
| **Semantic cache poisoning** | Cache entry serves wrong topic repeatedly (negative feedback) | Invalidate cache entry; reduce TTL by 50% for that topic | Bypass cache for 3 subsequent queries to that topic cluster |

### Circuit Breaker Pattern

```
CLASS RetrievalCircuitBreaker:
    STATE: CLOSED | OPEN | HALF_OPEN
    failure_count: int = 0
    failure_threshold: int = 5
    reset_timeout: int = 60   // seconds
    last_failure: datetime = None
    
    FUNCTION call(retrieval_fn, *args) -> Result:
        IF state == OPEN:
            IF (utcnow() - last_failure).seconds > reset_timeout:
                state = HALF_OPEN
            ELSE:
                RETURN fallback_result()
        
        TRY:
            result = await retrieval_fn(*args)
            IF state == HALF_OPEN:
                state = CLOSED
                failure_count = 0
            RETURN result
        EXCEPT:
            failure_count += 1
            last_failure = utcnow()
            IF failure_count >= failure_threshold:
                state = OPEN
            RETURN fallback_result()
```

---

## MODE COHERENCE ACROSS ALL SENTINEL MODES

The system must maintain consistent behavior across all modes while adapting the aggressiveness of each component:

| Component | Standard | Research/Debate | Research/Evidence | Research/Glass | Research/Stress |
|---|---|---|---|---|---|
| Cache threshold | 0.87 | 0.87 | 0.80 (lower = more retrieval) | 0.87 | 0.87 |
| Hallucination gate | Soften unsupported | Soften unsupported | Strip unsupported | Strip unsupported | Soften unsupported |
| Coverage threshold | 0.50 | 0.55 | 0.70 (strict) | 0.60 | 0.50 |
| Context budget | 4096 tokens | 6144 tokens | 8192 tokens | 6144 tokens | 4096 tokens |
| Evidence priority | 30% of budget | 35% | 45% | 35% | 25% |
| Conflict visibility | Hidden | Research panel | Inline warnings | Full exposure | Summary only |
| Graph depth | 1 hop | 2 hops | 3 hops | 2 hops | 1 hop |
| Decay lambda | 0.15 | 0.10 (slower decay) | 0.08 (slowest) | 0.12 | 0.20 (fastest) |
| Regen attempts | 1 | 2 | 3 | 2 | 1 |

---

## INTEGRATION POINTS WITH EXISTING CODEBASE

| New Component | Integrates With | Integration Method |
|---|---|---|
| `IntentHasher` | `backend/retrieval/cognitive_rag.py::QueryClassifier` | Wraps classifier output into `IntentHash` struct |
| `EvidenceCache` | `backend/retrieval/cognitive_rag.py::CognitiveRAG.process()` | Inserted before `._search()` call as cache gate |
| `SessionMemoryTier` | `backend/memory/memory_engine.py::ShortTermMemory` | Replaces current fixed-window with semantic-decay window |
| `EvidenceMemory` | `backend/core/evidence_engine.py::EvidenceEngine` | Wraps evidence results into persistent indexed storage |
| `KnowledgeGraph` | `backend/storage/schema.sql` (new tables) | New Postgres tables + FAISS indexes |
| `ContextCompiler` | `backend/memory/memory_engine.py::MemoryEngine.build_prompt_context()` | Replaces flat context with structured block compilation |
| `TopicBoundaryDetector` | `backend/core/session_intelligence.py::SessionIntelligence` | Integrates with domain inference and expertise scoring |
| `HallucinationGate` | `backend/core/confidence_engine.py`, `backend/core/evidence_engine.py` | Post-generation verification before response delivery |
| `LiveIngestionDaemon` | `backend/core/ingestion.py::IngestionEngine` | Background async daemon feeding into evidence memory + KG |
| `ModeCoherenceConfig` | `backend/core/mode_config.py::ModeConfig` | Extends `ModeConfig` with per-mode tuning parameters |

---

*End of Architecture Specification*
