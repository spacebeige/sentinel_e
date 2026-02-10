# Sentinel-Σ (Sigma) Database Schema
# PostgreSQL

CREATE TABLE IF NOT EXISTS sentinel_runs (
    run_id UUID PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode VARCHAR(50) NOT NULL CHECK (mode IN ('standard', 'experimental')),
    input_hash VARCHAR(64),
    status VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS sigma_hypotheses (
    hypothesis_id UUID PRIMARY KEY,
    run_id UUID REFERENCES sentinel_runs(run_id),
    model_source VARCHAR(50),
    statement TEXT,
    confidence FLOAT
);

CREATE TABLE IF NOT EXISTS sigma_graph_edges (
    edge_id UUID PRIMARY KEY,
    run_id UUID REFERENCES sentinel_runs(run_id),
    source_node VARCHAR(255),
    target_node VARCHAR(255),
    weight FLOAT
);
