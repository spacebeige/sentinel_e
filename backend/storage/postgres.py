import os
import logging
import json

# Try importing asyncpg, but don't crash if missing (graceful degradation)
try:
    import asyncpg
    HAS_POSTGRES_LIB = True
except ImportError:
    HAS_POSTGRES_LIB = False

logger = logging.getLogger("PostgresClient")

class PostgresClient:
    def __init__(self):
        self.connection_string = os.getenv("POSTGRES_URL")
        self.pool = None
        self.connected = False

    async def connect(self):
        if not HAS_POSTGRES_LIB:
            logger.warning("asyncpg not installed. Postgres DISABLED.")
            return

        if not self.connection_string:
            logger.warning("POSTGRES_URL not found. Running in stateless mode.")
            return

        try:
            logger.info(f"Connecting to Postgres...")
            self.pool = await asyncpg.create_pool(self.connection_string)
            self.connected = True
            logger.info("Postgres Connected.")
            await self.initialize_schema()
        except Exception as e:
            logger.error(f"Postgres Connection Failed: {e}. Falling back to stateless.")
            self.connected = False

    async def initialize_schema(self):
        """Create necessary tables if they don't exist."""
        if not self.connected:
            return
            
        queries = [
            """
            CREATE TABLE IF NOT EXISTS sentinel_runs (
                run_id VARCHAR(255) PRIMARY KEY,
                mode VARCHAR(50),
                input_hash VARCHAR(255),
                status VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS sigma_hypotheses (
                hypothesis_id VARCHAR(255) PRIMARY KEY,
                run_id VARCHAR(255),
                model_source VARCHAR(50),
                statement TEXT,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS boundary_violations (
                id SERIAL PRIMARY KEY,
                run_id VARCHAR(255),
                claim_id VARCHAR(255),
                severity_level VARCHAR(50),
                severity_score FLOAT,
                grounding_score FLOAT,
                required_grounding JSONB,
                missing_grounding JSONB,
                human_review_required BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
             """
            CREATE TABLE IF NOT EXISTS refusal_decisions (
                id SERIAL PRIMARY KEY,
                run_id VARCHAR(255),
                refused BOOLEAN,
                refusal_reason TEXT,
                boundary_severity FLOAT,
                severity_level VARCHAR(50),
                violation_count INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]
        
        try:
            async with self.pool.acquire() as conn:
                for q in queries:
                    await conn.execute(q)
            logger.info("Postgres schema initialized.")
        except Exception as e:
            logger.error(f"Schema Initialization Failed: {e}")

    async def write_run(self, run_id: str, mode: str, input_hash: str, status: str):
        if not self.connected:
            return
            
        query = """
        INSERT INTO sentinel_runs (run_id, mode, input_hash, status)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (run_id) DO NOTHING;
        """
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(query, run_id, mode, input_hash, status)
        except Exception as e:
            logger.error(f"DB Write Failed: {e}")

    async def write_hypotheses(self, run_id: str, hypotheses_map: dict):
        if not self.connected:
            return

        # hypotheses_map: { "ModelName": ["H1", "H2"] }
        query = """
        INSERT INTO sigma_hypotheses (hypothesis_id, run_id, model_source, statement, confidence)
        VALUES ($1, $2, $3, $4, 1.0)
        """
        
        try:
            async with self.pool.acquire() as conn:
                for model, hyps in hypotheses_map.items():
                    for h in hyps:
                        import uuid
                        h_id = str(uuid.uuid4())
                        await conn.execute(query, h_id, run_id, model, h)
        except Exception as e:
            logger.error(f"Hypothesis DB Write Failed: {e}")

    async def write_boundary_violations(self, run_id: str, claim_id: str, violations: dict):
        """Persist boundary violations to database."""
        if not self.connected:
            return
        
        query = """
        INSERT INTO boundary_violations 
        (run_id, claim_id, severity_level, severity_score, grounding_score, 
         required_grounding, missing_grounding, human_review_required)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT DO NOTHING;
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    run_id,
                    claim_id,
                    violations.get("severity_level"),
                    violations.get("severity_score"),
                    violations.get("grounding_score"),
                    json.dumps(violations.get("required_grounding", [])),
                    json.dumps(violations.get("missing_grounding", [])),
                    violations.get("human_review_required", False),
                )
        except Exception as e:
            logger.error(f"Boundary Violation DB Write Failed: {e}")

    async def update_model_boundary_profile(self, model_id: str, profile_date: str, metrics: dict):
        """Update model boundary profile for a given date."""
        if not self.connected:
            return
        
        query = """
        INSERT INTO model_boundary_profiles 
        (model_id, profile_date, total_claims, critical_violations, high_violations,
         medium_violations, low_violations, mean_severity, median_severity, max_severity,
         violation_rate, grounding_completeness)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (model_id, profile_date) DO UPDATE SET
            total_claims = $3,
            critical_violations = $4,
            high_violations = $5,
            medium_violations = $6,
            low_violations = $7,
            mean_severity = $8,
            median_severity = $9,
            max_severity = $10,
            violation_rate = $11,
            grounding_completeness = $12;
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    model_id,
                    profile_date,
                    metrics.get("total_claims", 0),
                    metrics.get("critical_violations", 0),
                    metrics.get("high_violations", 0),
                    metrics.get("medium_violations", 0),
                    metrics.get("low_violations", 0),
                    metrics.get("mean_severity", 0.0),
                    metrics.get("median_severity", 0.0),
                    metrics.get("max_severity", 0.0),
                    metrics.get("violation_rate", 0.0),
                    metrics.get("grounding_completeness", 0.0),
                )
        except Exception as e:
            logger.error(f"Model Profile DB Update Failed: {e}")

    async def write_refusal_decision(self, run_id: str, refused: bool, refusal_reason: str, 
                                     boundary_severity: float, severity_level: str, violation_count: int):
        """Log refusal decisions for audit trail."""
        if not self.connected:
            return
        
        query = """
        INSERT INTO refusal_decisions 
        (run_id, refused, refusal_reason, boundary_severity, severity_level, violation_count)
        VALUES ($1, $2, $3, $4, $5, $6);
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, run_id, refused, refusal_reason, boundary_severity, severity_level, violation_count)
        except Exception as e:
            logger.error(f"Refusal Decision DB Write Failed: {e}")

    async def write_human_feedback(self, run_id: str, feedback: str, reason: str = None, user_id: str = None):
        """Persist human feedback for learning."""
        if not self.connected:
            return
        
        query = """
        INSERT INTO human_feedback 
        (run_id, feedback, reason, user_id)
        VALUES ($1, $2, $3, $4);
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, run_id, feedback, reason, user_id)
        except Exception as e:
            logger.error(f"Human Feedback DB Write Failed: {e}")

    async def query_model_history(self, model_id: str, days: int = 30) -> list:
        """Query historical boundary violations for a model."""
        if not self.connected:
            return []
        
        query = """
        SELECT severity_score, severity_level, violation_timestamp
        FROM boundary_violations
        WHERE run_id IN (
            SELECT run_id FROM runs WHERE metadata->>'model_id' = $1
        )
        AND violation_timestamp > NOW() - INTERVAL '1 day' * $2
        ORDER BY violation_timestamp DESC;
        """
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, model_id, days)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Model History Query Failed: {e}")
            return []

    async def close(self):
        if self.pool:
            await self.pool.close()
