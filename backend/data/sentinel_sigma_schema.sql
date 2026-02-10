-- ============================================================
-- SENTINEL-Σ (SIGMA) ANALYST DATABASE SCHEMA
-- ============================================================
-- Design Goals:
-- 1. Structural Analysis: Focus on hypotheses and collapse events
-- 2. Longitudinal Tracking: Analyze failure patterns over time
-- 3. Machine-Native: Optimized for automated querying
-- ============================================================

-- 1. Run Metadata
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_summary TEXT,
    models_used TEXT,       -- JSON string: ["qwen", "groq", "mistral"]
    notes TEXT
);

-- 2. Evidence Registry
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    modality TEXT CHECK(modality IN ('text', 'image', 'pdf', 'code')),
    source TEXT,
    content_hash TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

-- 3. Atomic Claims (Explicit Assertions)
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    model TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

-- 4. Latent Hypotheses (Implicit Assumptions) - CRITICAL TABLE
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    model TEXT NOT NULL,
    hypothesis_text TEXT NOT NULL,
    dependency_strength REAL DEFAULT 0.5,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

-- 5. Hypothesis Intersection Graph (Shared Dependencies)
CREATE TABLE IF NOT EXISTS hypothesis_graph (
    graph_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    supporting_models TEXT, -- JSON string: ["groq", "mistral"]
    centrality_score REAL,
    is_single_point_of_failure BOOLEAN DEFAULT 0,
    FOREIGN KEY(run_id) REFERENCES runs(run_id),
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- 6. Stress Tests (Gradient / Asymmetric)
CREATE TABLE IF NOT EXISTS stress_tests (
    stress_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stress_type TEXT,   -- 'image_blur', 'clause_removal', 'asymmetric_context'
    stress_level REAL,  -- Scale 0.0-1.0
    description TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

-- 7. Collapse Events (Structural Failures)
CREATE TABLE IF NOT EXISTS collapse_events (
    collapse_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    stress_id TEXT NOT NULL,
    collapse_order INTEGER, -- 1 = collapsed first (most fragile)
    FOREIGN KEY(run_id) REFERENCES runs(run_id),
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id),
    FOREIGN KEY(stress_id) REFERENCES stress_tests(stress_id)
);

-- 8. Final Metrics
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT PRIMARY KEY,
    consensus_integrity_score REAL, -- 0.0-1.0
    hypothesis_fragility_index REAL, -- Higher = more fragile
    classification TEXT, -- 'structural_consensus', 'false_consensus', 'fragile_agreement'
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

-- Indexes for Analysis
CREATE INDEX IF NOT EXISTS idx_hypotheses_run ON hypotheses(run_id);
CREATE INDEX IF NOT EXISTS idx_collapse_hypothesis ON collapse_events(hypothesis_id);
