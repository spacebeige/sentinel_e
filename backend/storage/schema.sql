-- ============================================
-- SENTINEL-E / SENTINEL-Σ POSTGRES SCHEMA
-- ============================================
-- Comprehensive schema for boundary profiling, claims analysis, and human feedback
-- All tables use UUID primary keys
-- Time-series capable with indexed timestamps
-- Supports persistent boundary profiling and correlation analysis
--
-- CRITICAL: NO foreign-key cascades (to prevent accidental data loss)
-- Schema is vendor-neutral Postgres

-- ============================================
-- 1. MODELS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    vendor VARCHAR(100),  -- e.g., "OpenAI", "Groq", "Mistral", "Local"
    version VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_vendor ON models(vendor);

-- ============================================
-- 2. RUNS TABLE (Execution Metadata)
-- ============================================
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope VARCHAR(50) NOT NULL,  -- "standard" or "experimental"
    mode VARCHAR(100),  -- e.g., "full", "shadow_boundaries", "critical_boundaries", "hypothesis_only"
    input_text TEXT,
    status VARCHAR(50),  -- "complete", "failed", "incomplete"
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    error_message TEXT,
    metadata JSONB,  -- Arbitrary run metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_runs_scope ON runs(scope);
CREATE INDEX idx_runs_started_at ON runs(started_at DESC);
CREATE INDEX idx_runs_status ON runs(status);

-- ============================================
-- 3. CLAIMS TABLE (Atomic Claims from Hypotheses)
-- ============================================
CREATE TABLE IF NOT EXISTS claims (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,  -- FK (no cascade)
    model_id UUID,  -- FK (no cascade) - which model produced this claim
    claim_text TEXT NOT NULL,
    claim_type VARCHAR(100),  -- e.g., "causal_claim", "factual_claim", "predictive_claim"
    extraction_round INTEGER,
    is_hypothesis BOOLEAN DEFAULT false,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_claims_run_id ON claims(run_id);
CREATE INDEX idx_claims_model_id ON claims(model_id);
CREATE INDEX idx_claims_claim_type ON claims(claim_type);
CREATE INDEX idx_claims_created_at ON claims(created_at DESC);

-- ============================================
-- 4. BOUNDARY REQUIREMENTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS boundary_requirements (
    req_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_type VARCHAR(100) NOT NULL UNIQUE,
    required_grounding JSONB NOT NULL,  -- Array of grounding types, e.g., ["temporal_precedence", "mechanism"]
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_boundary_requirements_claim_type ON boundary_requirements(claim_type);

-- ============================================
-- 5. GROUNDING OBSERVATIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS grounding_observations (
    obs_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL,  -- FK (no cascade)
    observation_text TEXT NOT NULL,
    observation_type VARCHAR(100),  -- e.g., "source_citation", "corroboration", "evidence"
    source VARCHAR(255),
    confidence_score FLOAT,  -- 0-1
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_grounding_observations_claim_id ON grounding_observations(claim_id);
CREATE INDEX idx_grounding_observations_type ON grounding_observations(observation_type);

-- ============================================
-- 6. BOUNDARY VIOLATIONS TABLE (Central to Analysis)
-- ============================================
CREATE TABLE IF NOT EXISTS boundary_violations (
    violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,  -- FK (no cascade)
    claim_id UUID,  -- FK (no cascade)
    boundary_id VARCHAR(255),  -- Unique boundary identifier (from BoundaryDetector)
    severity_level VARCHAR(20),  -- "critical", "high", "medium", "low", "minimal"
    severity_score FLOAT NOT NULL,  -- 0-100
    grounding_score FLOAT,  -- 0-100 (completeness of available grounding)
    required_grounding JSONB,  -- Array of what was needed
    missing_grounding JSONB,  -- Array of what was absent
    human_review_required BOOLEAN DEFAULT false,
    violation_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_by_human BOOLEAN DEFAULT false,
    review_notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_boundary_violations_run_id ON boundary_violations(run_id);
CREATE INDEX idx_boundary_violations_claim_id ON boundary_violations(claim_id);
CREATE INDEX idx_boundary_violations_severity ON boundary_violations(severity_score DESC);
CREATE INDEX idx_boundary_violations_violation_timestamp ON boundary_violations(violation_timestamp DESC);
CREATE INDEX idx_boundary_violations_human_review ON boundary_violations(human_review_required);

-- ============================================
-- 7. MODEL BOUNDARY PROFILES TABLE  
-- ============================================
-- Persistent, time-series boundary behavior per model
CREATE TABLE IF NOT EXISTS model_boundary_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,  -- FK (no cascade)
    profile_date DATE NOT NULL,
    total_claims INTEGER,
    critical_violations INTEGER,
    high_violations INTEGER,
    medium_violations INTEGER,
    low_violations INTEGER,
    mean_severity FLOAT,  -- Average severity of all violations
    median_severity FLOAT,
    max_severity FLOAT,
    violation_rate FLOAT,  -- violations / total_claims
    grounding_completeness FLOAT,  -- Mean grounding score
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_boundary_profiles_model_id ON model_boundary_profiles(model_id);
CREATE INDEX idx_model_boundary_profiles_date ON model_boundary_profiles(profile_date DESC);
CREATE UNIQUE INDEX idx_model_boundary_profiles_unique ON model_boundary_profiles(model_id, profile_date);

-- ============================================
-- 8. HUMAN FEEDBACK TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS human_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,  -- FK (no cascade)
    feedback VARCHAR(10) NOT NULL,  -- "up" (👍) or "down" (👎)
    reason TEXT,  -- Optional; max 500 chars for 👎 feedback
    user_id VARCHAR(255),  -- Anonymous user identifier (hashed)
    ip_address INET,  -- For rate-limiting and fraud detection
    feedback_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_human_feedback_run_id ON human_feedback(run_id);
CREATE INDEX idx_human_feedback_feedback_timestamp ON human_feedback(feedback_timestamp DESC);
CREATE INDEX idx_human_feedback_feedback_type ON human_feedback(feedback);

-- ============================================
-- 9. REFUSAL DECISIONS TABLE
-- ============================================
-- Log refusal decisions for analysis
CREATE TABLE IF NOT EXISTS refusal_decisions (
    refusal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,  -- FK (no cascade)
    refused BOOLEAN NOT NULL,
    refusal_reason VARCHAR(500),
    boundary_severity FLOAT,
    severity_level VARCHAR(20),
    violation_count INTEGER,
    refusal_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_refusal_decisions_run_id ON refusal_decisions(run_id);
CREATE INDEX idx_refusal_decisions_refused ON refusal_decisions(refused);
CREATE INDEX idx_refusal_decisions_refusal_timestamp ON refusal_decisions(refusal_timestamp DESC);

-- ============================================
-- 10. ANALYSIS SESSIONS TABLE
-- ============================================
-- Tracks multi-run experimental sessions
CREATE TABLE IF NOT EXISTS analysis_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_name VARCHAR(255),
    analyst_id VARCHAR(255),  -- Anonymous analyst identifier
    scope VARCHAR(50),  -- "standard" or "experimental"
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    run_count INTEGER DEFAULT 0,
    total_claims INTEGER DEFAULT 0,
    total_violations INTEGER DEFAULT 0,
    mean_severity FLOAT,
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_sessions_started_at ON analysis_sessions(started_at DESC);
CREATE INDEX idx_analysis_sessions_scope ON analysis_sessions(scope);

-- ============================================
-- 11. SESSION_RUNS JUNCTION TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS session_runs (
    session_id UUID NOT NULL,  -- FK (no cascade)
    run_id UUID NOT NULL,  -- FK (no cascade)
    PRIMARY KEY (session_id, run_id)
);

CREATE INDEX idx_session_runs_session_id ON session_runs(session_id);
CREATE INDEX idx_session_runs_run_id ON session_runs(run_id);

-- ============================================
-- VIEWS FOR ANALYSIS
-- ============================================

-- Summary of violations per run
CREATE OR REPLACE VIEW run_violation_summary AS
SELECT 
    r.run_id,
    r.scope,
    COUNT(DISTINCT bv.violation_id) as total_violations,
    COUNT(DISTINCT bv.violation_id) FILTER (WHERE bv.severity_level = 'critical') as critical_count,
    COUNT(DISTINCT bv.violation_id) FILTER (WHERE bv.severity_level = 'high') as high_count,
    COUNT(DISTINCT bv.violation_id) FILTER (WHERE bv.severity_level = 'medium') as medium_count,
    COUNT(DISTINCT bv.violation_id) FILTER (WHERE bv.severity_level = 'low') as low_count,
    AVG(bv.severity_score) as mean_severity,
    MAX(bv.severity_score) as max_severity,
    r.started_at,
    r.completed_at
FROM runs r
LEFT JOIN boundary_violations bv ON r.run_id = bv.run_id
GROUP BY r.run_id, r.scope, r.started_at, r.completed_at;

-- Model performance vs boundary severity
CREATE OR REPLACE VIEW model_violation_trends AS
SELECT 
    m.model_id,
    m.name,
    m.vendor,
    DATE(r.started_at) as profile_date,
    COUNT(DISTINCT c.claim_id) as total_claims,
    COUNT(DISTINCT bv.violation_id) as total_violations,
    ROUND(100.0 * COUNT(DISTINCT bv.violation_id) / NULLIF(COUNT(DISTINCT c.claim_id), 0), 2) as violation_rate,
    AVG(bv.severity_score) as mean_severity,
    MAX(bv.severity_score) as max_severity
FROM models m
LEFT JOIN claims c ON m.model_id = c.model_id
LEFT JOIN runs r ON c.run_id = r.run_id
LEFT JOIN boundary_violations bv ON c.claim_id = bv.claim_id
WHERE r.started_at IS NOT NULL
GROUP BY m.model_id, m.name, m.vendor, DATE(r.started_at)
ORDER BY profile_date DESC, mean_severity DESC;

-- Feedback correlation with severity
CREATE OR REPLACE VIEW feedback_severity_correlation AS
SELECT 
    hf.feedback,
    COUNT(*) as feedback_count,
    AVG(bv.severity_score) as mean_violation_severity,
    COUNT(DISTINCT hf.run_id) as unique_runs
FROM human_feedback hf
JOIN runs r ON hf.run_id = r.run_id
LEFT JOIN boundary_violations bv ON r.run_id = bv.run_id
GROUP BY hf.feedback;

-- ============================================
-- GRANTS (adjust as needed for your roles)
-- ============================================
-- Grant SELECT on all tables to read-only roles
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO read_only_role;
-- Grant full access to app role
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_role;

-- ============================================
-- DONE
-- ============================================
-- To apply this schema to your Postgres instance:
-- psql -h <host> -U <user> -d <database> -f schema.sql
