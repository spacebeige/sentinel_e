"""
============================================================
Battle Platform Routes — Sentinel-E v2
============================================================
FastAPI router exposing all Battle Platform endpoints.

Endpoints:
    GET  /battle/leaderboard             — ELO rankings
    GET  /battle/leaderboard/{model_id}  — Single model entry
    POST /battle/vote                    — Accept a user vote
    GET  /battle/dataset/stats           — Dataset statistics
    GET  /battle/dataset/recent          — Last N records
    GET  /battle/ops                     — Real-time ops snapshot
    GET  /battle/ops/health              — System health summary
    GET  /battle/ops/latency             — Model latency percentiles
    POST /battle/company/submit          — Submit company eval job
    GET  /battle/company/status/{job_id} — Poll company job
    GET  /battle/company/list            — List all company jobs
    POST /battle/company/run/{job_id}    — Execute a submitted job
    POST /battle/benchmark/run           — Trigger benchmark session
    GET  /battle/benchmark/history       — Recent benchmark reports
    GET  /battle/benchmark/trend/{model} — Model score trend
    GET  /battle/models/tiers            — Tiered model registry
    POST /battle/metrics                 — Evaluate raw outputs
============================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

logger = logging.getLogger("BattleRoutes")

router = APIRouter(prefix="/battle", tags=["Battle Platform v2"])


# ── Request/Response Models ───────────────────────────────────

class VoteRequest(BaseModel):
    record_id: str
    user_vote: str  # model_id the user voted for


class CompanyJobRequest(BaseModel):
    company_name: str
    model_name: str
    model_endpoint: str
    api_key_value: str
    api_key_header: str = "Authorization"


class MetricsRequest(BaseModel):
    """Evaluate raw model outputs directly (no debate required)."""
    outputs: List[Dict[str, Any]] = Field(
        description="List of {model_id, model_name, position, reasoning, confidence}"
    )


# ── Leaderboard ───────────────────────────────────────────────

@router.get("/leaderboard")
async def get_leaderboard(top_n: Optional[int] = None) -> Dict[str, Any]:
    """
    Return the ELO leaderboard.

    Example response:
        {
          "leaderboard": [
            {"rank": 1, "model_id": "llama-3.3", "elo_score": 1560, ...},
            {"rank": 2, "model_id": "mixtral-8x7b", "elo_score": 1532, ...}
          ]
        }
    """
    try:
        from ranking.elo_engine import get_elo_engine
        engine = get_elo_engine()
        return engine.get_leaderboard_dict()
    except Exception as exc:
        logger.error("Leaderboard error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/leaderboard/{model_id}")
async def get_model_ranking(model_id: str) -> Dict[str, Any]:
    """Return ELO entry for a specific model."""
    try:
        from ranking.elo_engine import get_elo_engine
        entry = get_elo_engine().get_model_entry(model_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return entry.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── User Vote ─────────────────────────────────────────────────

@router.post("/vote")
async def submit_vote(req: VoteRequest) -> Dict[str, Any]:
    """
    Accept a user vote for a debate record.

    The vote overrides the automated winner and updates ELO accordingly.
    """
    try:
        from evaluation.dataset import get_evaluation_dataset
        success = get_evaluation_dataset().accept_vote(
            record_id=req.record_id,
            user_vote=req.user_vote,
        )
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Record '{req.record_id}' not found",
            )
        return {"status": "accepted", "record_id": req.record_id, "user_vote": req.user_vote}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Evaluation Dataset ────────────────────────────────────────

@router.get("/dataset/stats")
async def get_dataset_stats() -> Dict[str, Any]:
    """Return dataset statistics: record count, win distribution, vote rate."""
    try:
        from evaluation.dataset import get_evaluation_dataset
        return get_evaluation_dataset().statistics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/dataset/recent")
async def get_recent_records(n: int = 20) -> Dict[str, Any]:
    """Return the N most recent evaluation records."""
    try:
        from evaluation.dataset import get_evaluation_dataset
        records = get_evaluation_dataset().load_recent(n=min(n, 200))
        return {"records": [r.model_dump() for r in records], "count": len(records)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Operations Dashboard ──────────────────────────────────────

@router.get("/ops")
async def get_ops_snapshot(window_seconds: float = 3600.0) -> Dict[str, Any]:
    """
    Return real-time operations snapshot.

    Includes: avg debate time, consensus confidence, conflict rate,
    error rate, per-model latencies, token usage.
    """
    try:
        from monitoring.ops_dashboard import get_ops_dashboard
        snapshot = get_ops_dashboard().get_snapshot(window_seconds=window_seconds)
        return snapshot.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ops/health")
async def get_system_health() -> Dict[str, Any]:
    """Return system health summary (healthy / degraded / critical)."""
    try:
        from monitoring.ops_dashboard import get_ops_dashboard
        return get_ops_dashboard().system_health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ops/latency")
async def get_latency_summary(window_seconds: float = 3600.0) -> Dict[str, Any]:
    """Return per-model latency percentiles (p50, p95, p99)."""
    try:
        from monitoring.ops_dashboard import get_ops_dashboard
        return get_ops_dashboard().get_model_latency_summary(window_seconds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ops/confidence-trend")
async def get_confidence_trend(
    window_seconds: float = 86400.0,
    bucket_count: int = 24,
) -> Dict[str, Any]:
    """Return consensus confidence trend (24-hour bucketed average)."""
    try:
        from monitoring.ops_dashboard import get_ops_dashboard
        trend = get_ops_dashboard().get_confidence_trend(window_seconds, bucket_count)
        return {"trend": trend}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Company Evaluation Pipeline ───────────────────────────────

@router.post("/company/submit")
async def submit_company_job(req: CompanyJobRequest) -> Dict[str, Any]:
    """
    Submit a company model for evaluation.

    Returns a job_id for status polling.
    """
    try:
        from evaluation.company_pipeline import get_company_pipeline
        from core.ensemble_schemas import CompanyEvaluationJob
        job = CompanyEvaluationJob(
            company_name=req.company_name,
            model_name=req.model_name,
            model_endpoint=req.model_endpoint,
            api_key_value=req.api_key_value,
            api_key_header=req.api_key_header,
        )
        job_id = get_company_pipeline().submit_job(job)
        return {"status": "submitted", "job_id": job_id, "model_name": req.model_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/company/run/{job_id}")
async def run_company_job(job_id: str) -> Dict[str, Any]:
    """
    Execute a submitted company evaluation job.

    This is a long-running operation (20+ API calls). In production,
    use the background task queue. In dev, runs synchronously.
    """
    try:
        from evaluation.company_pipeline import get_company_pipeline
        report = await get_company_pipeline().run_job(job_id=job_id)
        return report.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Company job %s failed: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/company/status/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Poll the status of a company evaluation job."""
    try:
        from evaluation.company_pipeline import get_company_pipeline
        status = get_company_pipeline().get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return status
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/company/list")
async def list_company_jobs() -> Dict[str, Any]:
    """Return a summary list of all company evaluation jobs."""
    try:
        from evaluation.company_pipeline import get_company_pipeline
        return {"jobs": get_company_pipeline().list_jobs()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Automated Benchmarking ────────────────────────────────────

@router.post("/benchmark/run")
async def run_benchmark_session(
    prompt_override: Optional[List[Dict[str, str]]] = Body(None),
) -> Dict[str, Any]:
    """
    Trigger an automated benchmark session.

    Optionally provide custom prompts (overrides the built-in bank).
    """
    try:
        from evaluation.benchmark_pipeline import get_benchmark_pipeline
        pipeline = get_benchmark_pipeline()
        report = await pipeline.run_benchmark_session(
            prompt_override=prompt_override or None,
        )
        return report
    except Exception as exc:
        logger.error("Benchmark run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/benchmark/history")
async def get_benchmark_history(n: int = 10) -> Dict[str, Any]:
    """Return the N most recent benchmark reports."""
    try:
        from evaluation.benchmark_pipeline import get_benchmark_pipeline
        reports = get_benchmark_pipeline().get_reports(n=min(n, 100))
        return {"reports": reports, "count": len(reports)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/benchmark/trend/{model_id}")
async def get_model_trend(model_id: str, n_sessions: int = 30) -> Dict[str, Any]:
    """
    Return score trend for a model across the last N benchmark sessions.

    Used to detect temporal drift in model performance.
    """
    try:
        from evaluation.benchmark_pipeline import get_benchmark_pipeline
        trend = get_benchmark_pipeline().get_model_trend(
            model_id=model_id, n_sessions=min(n_sessions, 365)
        )
        return {"model_id": model_id, "trend": trend}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Model Registry / Tiers ────────────────────────────────────

@router.get("/models/tiers")
async def get_model_tiers() -> Dict[str, Any]:
    """
    Return the full tiered model registry.

    Tiers:
      1 — Anchor Models  (primary reasoning reference)
      2 — Debate Models  (diverse argument generators)
      3 — Specialist     (domain-specific reasoning)
    """
    try:
        from metacognitive.cognitive_gateway import (
            COGNITIVE_MODEL_REGISTRY, MODEL_DEBATE_TIERS
        )
        result: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
        for model_key, tier in MODEL_DEBATE_TIERS.items():
            spec = COGNITIVE_MODEL_REGISTRY.get(model_key)
            if not spec:
                continue
            result[tier].append({
                "id": model_key,
                "name": spec.name,
                "model_id": spec.model_id,
                "provider": spec.provider,
                "role": spec.role.value,
                "enabled": spec.enabled and spec.active,
                "context_window": spec.context_window,
            })
        return {
            "tier_1_anchor": result[1],
            "tier_2_debate": result[2],
            "tier_3_specialist": result[3],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/models/select")
async def select_models_for_debate(
    prompt_type: str = "general",
    max_models: int = 6,
) -> Dict[str, Any]:
    """
    Return the dynamically-selected model set for a given prompt type.

    Used by the frontend to preview which models will debate.
    """
    try:
        from metacognitive.cognitive_gateway import get_tiered_models_for_debate
        selected = get_tiered_models_for_debate(
            prompt_type=prompt_type,
            max_models=min(max_models, 6),
        )
        return {
            "prompt_type": prompt_type,
            "max_models": max_models,
            "selected": selected,
            "count": len(selected),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Direct Metrics Evaluation ─────────────────────────────────

@router.post("/metrics")
async def evaluate_metrics(req: MetricsRequest) -> Dict[str, Any]:
    """
    Evaluate reasoning quality metrics for a set of raw model outputs.

    Useful for testing/debugging the MetricsEngine directly.
    """
    try:
        from analysis.metrics_engine import MetricsEngine
        from core.ensemble_schemas import StructuredModelOutput, StanceVector

        engine = MetricsEngine()
        outputs = []
        for item in req.outputs:
            outputs.append(StructuredModelOutput(
                model_id=item.get("model_id", "unknown"),
                model_name=item.get("model_name", item.get("model_id", "unknown")),
                position=item.get("position", ""),
                reasoning=item.get("reasoning", ""),
                confidence=float(item.get("confidence", 0.5)),
                raw_output=item.get("position", "") + " " + item.get("reasoning", ""),
            ))

        results = engine.evaluate_all(outputs)
        return {
            "metrics": [engine.to_frontend_dict(m) for m in results],
            "count": len(results),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
