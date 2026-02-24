-- ============================================
-- SENTINEL-E AUTONOMOUS REASONING ENGINE
-- Knowledge Graph + Evidence Memory Schema Extension
-- ============================================
-- Extends schema.sql with tables for the three-tier memory system
-- Apply AFTER schema.sql

-- ============================================
-- EVIDENCE MEMORY (Tier 2) 
-- ============================================
CREATE TABLE IF NOT EXISTS evidence_memory (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_origin TEXT NOT NULL,
    entity_tags JSONB NOT NULL DEFAULT '[]',
    topic_tags JSONB NOT NULL DEFAULT '[]',
    topic_embedding BYTEA,
    chunks JSONB NOT NULL DEFAULT '[]',
    source_metadata JSONB NOT NULL DEFAULT '[]',
    claims_extracted JSONB NOT NULL DEFAULT '[]',
    confidence_score FLOAT NOT NULL DEFAULT 0.5,
    contradiction_flags JSONB NOT NULL DEFAULT '[]',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_evidence_memory_entities ON evidence_memory USING GIN(entity_tags);
CREATE INDEX IF NOT EXISTS idx_evidence_memory_topics ON evidence_memory USING GIN(topic_tags);
CREATE INDEX IF NOT EXISTS idx_evidence_memory_created ON evidence_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_memory_confidence ON evidence_memory(confidence_score DESC);

-- ============================================
-- KNOWLEDGE GRAPH ENTITIES (Tier 3)
-- ============================================
CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]',
    entity_type VARCHAR(50) NOT NULL DEFAULT 'concept',
    embedding BYTEA,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_entities_name_type ON kg_entities(name, entity_type);

-- ============================================
-- KNOWLEDGE GRAPH CLAIMS (Tier 3)
-- ============================================
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

CREATE INDEX IF NOT EXISTS idx_kg_claims_subject ON kg_claims(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_claims_object ON kg_claims(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_claims_status ON kg_claims(status);
CREATE INDEX IF NOT EXISTS idx_kg_claims_predicate ON kg_claims(predicate);
CREATE INDEX IF NOT EXISTS idx_kg_claims_current ON kg_claims(is_current) WHERE is_current = true;

-- ============================================
-- KNOWLEDGE GRAPH RELATIONSHIPS (Tier 3)
-- ============================================
CREATE TABLE IF NOT EXISTS kg_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL,
    target_entity_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    weight FLOAT NOT NULL DEFAULT 0.5,
    evidence_ids JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kg_rel_source ON kg_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_rel_target ON kg_relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_rel_type ON kg_relationships(relationship_type);

-- ============================================
-- EVIDENCE CACHE METADATA
-- ============================================
CREATE TABLE IF NOT EXISTS evidence_cache_log (
    cache_key VARCHAR(64) PRIMARY KEY,
    query_canonical TEXT NOT NULL,
    intent_type VARCHAR(50),
    hit_count INTEGER NOT NULL DEFAULT 0,
    ttl_seconds INTEGER NOT NULL DEFAULT 1800,
    confidence_score FLOAT,
    freshness_class VARCHAR(20),
    source_count INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_hit_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cache_log_intent ON evidence_cache_log(intent_type);
CREATE INDEX IF NOT EXISTS idx_cache_log_created ON evidence_cache_log(created_at DESC);

-- ============================================
-- INTENT HISTORY
-- ============================================
CREATE TABLE IF NOT EXISTS intent_history (
    intent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    exact_hash VARCHAR(64) NOT NULL,
    canonical TEXT,
    intent_type VARCHAR(50),
    retrieval_p FLOAT,
    temporal_flag BOOLEAN DEFAULT false,
    cache_hit BOOLEAN DEFAULT false,
    cache_level VARCHAR(10),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_intent_history_session ON intent_history(session_id);
CREATE INDEX IF NOT EXISTS idx_intent_history_hash ON intent_history(exact_hash);
CREATE INDEX IF NOT EXISTS idx_intent_history_created ON intent_history(created_at DESC);

-- ============================================
-- TOPIC BOUNDARY EVENTS
-- ============================================
CREATE TABLE IF NOT EXISTS topic_boundary_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    previous_cluster_id VARCHAR(255),
    new_cluster_id VARCHAR(255),
    drift_score FLOAT NOT NULL,
    action VARCHAR(50) NOT NULL,
    similarity_global FLOAT,
    similarity_local FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_topic_events_session ON topic_boundary_events(session_id);
CREATE INDEX IF NOT EXISTS idx_topic_events_created ON topic_boundary_events(created_at DESC);

-- ============================================
-- HALLUCINATION VERIFICATION LOG
-- ============================================
CREATE TABLE IF NOT EXISTS hallucination_verification_log (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    run_id UUID,
    mode VARCHAR(50),
    status VARCHAR(20) NOT NULL,
    coverage FLOAT NOT NULL,
    verified_count INTEGER NOT NULL DEFAULT 0,
    unsupported_count INTEGER NOT NULL DEFAULT 0,
    opinion_count INTEGER NOT NULL DEFAULT 0,
    confidence_score FLOAT,
    regenerated BOOLEAN DEFAULT false,
    traceability_map JSONB DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_halluc_session ON hallucination_verification_log(session_id);
CREATE INDEX IF NOT EXISTS idx_halluc_status ON hallucination_verification_log(status);
CREATE INDEX IF NOT EXISTS idx_halluc_created ON hallucination_verification_log(created_at DESC);

-- ============================================
-- VIEWS
-- ============================================

-- Knowledge graph entity summary
CREATE OR REPLACE VIEW kg_entity_summary AS
SELECT 
    e.entity_id,
    e.name,
    e.entity_type,
    e.confidence,
    COUNT(DISTINCT c.claim_id) as claim_count,
    COUNT(DISTINCT c.claim_id) FILTER (WHERE c.status = 'disputed') as disputed_count,
    COUNT(DISTINCT r.relationship_id) as relationship_count,
    e.first_seen,
    e.last_updated
FROM kg_entities e
LEFT JOIN kg_claims c ON e.entity_id = c.subject_entity_id
LEFT JOIN kg_relationships r ON e.entity_id = r.source_entity_id OR e.entity_id = r.target_entity_id
GROUP BY e.entity_id, e.name, e.entity_type, e.confidence, e.first_seen, e.last_updated;

-- Cache performance metrics
CREATE OR REPLACE VIEW cache_performance AS
SELECT 
    intent_type,
    COUNT(*) as total_queries,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
    ROUND(100.0 * SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as hit_rate_pct,
    cache_level,
    DATE(created_at) as query_date
FROM intent_history
GROUP BY intent_type, cache_level, DATE(created_at)
ORDER BY query_date DESC, hit_rate_pct DESC;

-- ============================================
-- DONE
-- ============================================
