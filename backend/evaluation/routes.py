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
    GET  /battle/models/select           — Dynamic model selection
    POST /battle/metrics                 — Evaluate raw outputs
    POST /battle/debate                  — Full multi-model debate (Debate Mode)
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


class DebateRequest(BaseModel):
    """Request body for POST /battle/debate."""
    query: str = Field(..., description="User question to debate across the model ensemble")
    chat_id: Optional[str] = Field(None, description="Session ID for multi-turn context")
    history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Conversation history for multi-turn context",
    )
    prompt_type: str = Field(
        "general",
        description="Prompt category: general | code | logical | evidence | depth | conceptual",
    )
    max_models: int = Field(
        7,
        ge=3,
        le=7,
        description="Max models for this debate (3–7, default 7)",
    )
    include_charts: bool = Field(
        False,
        description="If True, generate and embed base64 visualisation charts in the response",
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


@router.get("/ops/model-reliability")
async def get_model_reliability() -> Dict[str, Any]:
    """
    Return per-model reliability stats: success rate, avg latency,
    total invocations, provider, tier, and current status.
    """
    try:
        from metacognitive.cognitive_gateway import (
            COGNITIVE_MODEL_REGISTRY,
            MODEL_DEBATE_TIERS,
        )
        from monitoring.ops_dashboard import get_ops_dashboard

        dashboard = get_ops_dashboard()
        health = dashboard.system_health()
        latency = dashboard.get_model_latency_summary(window_seconds=3600.0)

        reliability = {}
        for key, spec in COGNITIVE_MODEL_REGISTRY.items():
            model_latency = latency.get(key, {})
            reliability[key] = {
                "name": spec.name,
                "provider": spec.provider,
                "tier": MODEL_DEBATE_TIERS.get(key, 0),
                "enabled": spec.enabled,
                "active": spec.active,
                "model_id": spec.model_id,
                "context_window": spec.context_window,
                "p50_latency_ms": model_latency.get("p50", 0),
                "p95_latency_ms": model_latency.get("p95", 0),
                "error_rate": health.get("per_model_error_rate", {}).get(key, 0.0),
            }

        return {"models": reliability}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ops/cache-stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Return response cache statistics: size, hit rate, entries."""
    try:
        from optimization import get_response_cache
        cache = get_response_cache()
        return cache.stats()
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
      3 — Fallback       (reliability guarantee models)
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
            "tier_3_fallback": result[3],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/models/select")
async def select_models_for_debate(
    prompt_type: str = "general",
    max_models: int = 7,
) -> Dict[str, Any]:
    """
    Return the dynamically-selected model set for a given prompt type.

    Used by the frontend to preview which models will debate.
    """
    try:
        from metacognitive.cognitive_gateway import get_tiered_models_for_debate
        selected = get_tiered_models_for_debate(
            prompt_type=prompt_type,
            max_models=min(max_models, 7),
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


# ── Debate Mode ───────────────────────────────────────────────

@router.post("/debate")
async def run_debate(req: DebateRequest) -> Dict[str, Any]:
    """
    Execute a full multi-model debate and return the BattleVisualizationPayload.

    Pipeline:
      1. Select models via get_tiered_models_for_debate (2 anchors + 3 debate + 1 specialist).
      2. Run CognitiveCoreEngine.process() → EnsembleResponse (3-round debate).
      3. Build BattleVisualizationPayload via BattleVisualizationEngine.build().
      4. Return the full frontend-ready dict including per-model reasoning metrics,
         consensus scores, timeline, conflict edges, and optional base64 charts.

    Example request:
        POST /battle/debate
        {
          "query": "Should we prioritise energy security or decarbonisation?",
          "prompt_type": "general",
          "max_models": 6,
          "include_charts": false
        }

    Example response:
        {
          "prompt": "Should we …",
          "prompt_type": "general",
          "models_selected": ["llama-3.3", "deepseek-chat", "groq-small", …],
          "round_outputs": { "1": [...], "2": [...], "3": [...] },
          "reasoning_metrics": [...],
          "consensus_scores": [...],
          "consensus_stability_score": 0.72,
          "agreement_heatmap": [[...]],
          "model_labels": [...],
          "conflict_edges": [...],
          "debate_timeline": [...],
          "winner": "deepseek-chat",
          "winner_score": 0.84,
          "models": [
            {"id": "llama-3.3", "name": "Llama 3.3 70B", "tier": 1, ...},
            …
          ]
        }
    """
    import uuid as _uuid

    try:
        from metacognitive.cognitive_gateway import (
            COGNITIVE_MODEL_REGISTRY,
            MODEL_DEBATE_TIERS,
            get_tiered_models_for_debate,
        )
        from core.ensemble_schemas import MIN_DEBATE_ROUNDS
        from viz.battle_visualization import BattleVisualizationEngine

        # ── 0. Check debate result cache ───────────────────────
        from optimization import get_response_cache
        _cache = get_response_cache()
        import hashlib as _hashlib
        _cache_key = f"debate:{_hashlib.sha256(f'{req.query}:{req.prompt_type}:{req.max_models}'.encode()).hexdigest()}"
        cached = _cache.get(_cache_key)
        if cached:
            logger.info(f"Debate cache hit for query: {req.query[:60]}...")
            cached["cache_hit"] = True
            return cached

        # ── 1. Select tiered models ────────────────────────────
        selected_keys = get_tiered_models_for_debate(
            prompt_type=req.prompt_type,
            max_models=req.max_models,
        )

        if len(selected_keys) < 3:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Insufficient enabled models for debate: "
                    f"got {len(selected_keys)}, need ≥3. "
                    f"Check API key configuration."
                ),
            )

        # ── 2. Build models metadata list (for frontend) ───────
        models_meta = []
        for key in selected_keys:
            spec = COGNITIVE_MODEL_REGISTRY.get(key)
            if spec:
                models_meta.append({
                    "id": key,
                    "name": spec.name,
                    "provider": spec.provider,
                    "role": spec.role.value,
                    "tier": MODEL_DEBATE_TIERS.get(key, 2),
                })

        # ── 3. Run debate via CognitiveCoreEngine ──────────────
        # The cognitive engine is a global singleton wired in main.py.
        # We access it via the module-level reference injected at startup.
        import backend.main as _main  # type: ignore[import]
        engine = getattr(_main, "cognitive_orchestrator_engine", None)

        if engine is None:
            # Attempt lazy initialisation for test environments
            from models.mco_bridge import MCOModelBridge
            from metacognitive.orchestrator import MetaCognitiveOrchestrator
            _orch = MetaCognitiveOrchestrator()
            _bridge = MCOModelBridge(_orch.cognitive_gateway)
            from core.cognitive_orchestrator import CognitiveCoreEngine
            engine = CognitiveCoreEngine(model_bridge=_bridge)

        chat_id = req.chat_id or str(_uuid.uuid4())

        # ── 2b. Evidence retrieval (non-blocking) ──────────────
        evidence_context = None
        try:
            from core.evidence_debate_pipeline import gather_evidence
            evidence_context = await gather_evidence(req.query, max_results=5)
            if evidence_context and evidence_context.search_executed:
                logger.info(
                    f"Evidence gathered: {evidence_context.source_count} sources, "
                    f"confidence={evidence_context.evidence_confidence:.4f}"
                )
        except Exception as ev_exc:
            logger.warning(f"Evidence retrieval skipped: {ev_exc}")

        ensemble_response = await engine.process(
            query=req.query,
            chat_id=chat_id,
            rounds=MIN_DEBATE_ROUNDS,
            history=req.history,
        )

        # ── 4. Build BattleVisualizationPayload ────────────────
        # Convert EnsembleResponse debate rounds into the round_outputs
        # dict expected by BattleVisualizationEngine.build().
        round_outputs: Dict[str, List[Dict[str, Any]]] = {}
        if hasattr(ensemble_response, "debate_rounds"):
            for dr in ensemble_response.debate_rounds:
                rkey = str(getattr(dr, "round_number", "?"))
                round_outputs[rkey] = [
                    {
                        "model": getattr(pos, "model_id", ""),
                        "output": getattr(pos, "reasoning", ""),
                        "tokens_used": getattr(pos, "tokens_used", 0),
                    }
                    for pos in getattr(dr, "positions", [])
                ]

        # Final round structured outputs for metrics/consensus computation
        final_outputs = []
        if hasattr(ensemble_response, "structured_outputs"):
            final_outputs = ensemble_response.structured_outputs
        elif round_outputs:
            last_key = sorted(round_outputs.keys())[-1]
            from core.ensemble_schemas import StructuredModelOutput
            for item in round_outputs[last_key]:
                final_outputs.append(StructuredModelOutput(
                    model_id=item.get("model", "unknown"),
                    model_name=item.get("model", "unknown"),
                    position=item.get("output", ""),
                    reasoning=item.get("output", ""),
                    raw_output=item.get("output", ""),
                ))

        viz_engine = BattleVisualizationEngine()
        payload = viz_engine.build(
            prompt=req.query,
            prompt_type=req.prompt_type,
            round_outputs=round_outputs,
            final_round_outputs=final_outputs,
            include_charts=req.include_charts,
        )

        result = viz_engine.to_frontend_dict(payload)

        # Augment with models metadata for dynamic frontend rendering
        result["models"] = models_meta
        result["chat_id"] = chat_id

        # Augment with evidence data for frontend evidence panel
        if evidence_context and evidence_context.search_executed:
            result["evidence"] = evidence_context.to_frontend_dict()
        else:
            result["evidence"] = None

        # ── 5. Anchor Model Pass (post-debate evaluation) ──────
        # Runs heavyweight reasoning models ONCE to evaluate debate
        # quality and produce calibrated final synthesis.
        # Skipped if no anchor API keys are configured.
        result["anchor_pass"] = None
        try:
            from core.anchor_pass import get_anchor_engine
            anchor_engine = get_anchor_engine()
            if anchor_engine.has_anchors():
                # Build positions summary for anchor prompt
                positions_lines = []
                if final_outputs:
                    for fo in final_outputs[:7]:
                        pos_text = getattr(fo, "position", "") or ""
                        positions_lines.append(
                            f"- {getattr(fo, 'model_name', fo.model_id)}: {pos_text[:200]}"
                        )
                positions_summary = "\n".join(positions_lines) or "(No positions)"

                evidence_summary = ""
                if evidence_context and evidence_context.search_executed:
                    evidence_summary = "\n".join(
                        f"- {s.get('title', 'Source')}: {s.get('snippet', '')[:150]}"
                        for s in (evidence_context.to_frontend_dict().get("sources", []))[:5]
                    )

                anchor_result = await anchor_engine.evaluate(
                    query=req.query,
                    debate_synthesis=getattr(ensemble_response, "formatted_output", "") or payload.winner or "",
                    debate_metrics={
                        "model_count": len(models_meta),
                        "round_count": len(round_outputs),
                        "consensus_strength": payload.consensus_stability_score,
                        "disagreement": result.get("reasoning_analytics", {}).get("disagreement_strength", 0) if isinstance(result.get("reasoning_analytics"), dict) else 0,
                        "tokens_spent": sum(
                            item.get("tokens_used", 0)
                            for items in round_outputs.values()
                            for item in items
                        ),
                    },
                    positions_summary=positions_summary,
                    evidence_summary=evidence_summary,
                )

                if anchor_result.anchor_count > 0:
                    result["anchor_pass"] = {
                        "anchor_count": anchor_result.anchor_count,
                        "avg_quality_score": anchor_result.avg_quality_score,
                        "avg_confidence": anchor_result.avg_confidence,
                        "anchor_agreement": anchor_result.anchor_agreement,
                        "dominant_verdict": anchor_result.dominant_verdict,
                        "combined_synthesis": anchor_result.combined_synthesis,
                        "evaluations": [
                            {
                                "anchor_model": e.anchor_model,
                                "anchor_name": e.anchor_name,
                                "quality_score": e.quality_score,
                                "confidence": e.confidence,
                                "verdict": e.verdict,
                                "verdict_reason": e.verdict_reason,
                                "reasoning_flaws": e.reasoning_flaws,
                                "synthesis": e.synthesis[:500],
                                "latency_ms": round(e.latency_ms, 1),
                            }
                            for e in anchor_result.evaluations
                        ],
                    }
                    logger.info(
                        f"Anchor pass complete: {anchor_result.anchor_count} anchors, "
                        f"quality={anchor_result.avg_quality_score:.2f}, "
                        f"verdict={anchor_result.dominant_verdict}"
                    )
        except Exception as anchor_exc:
            logger.warning(f"Anchor pass skipped: {anchor_exc}")

        # Record battle in ELO engine (async, non-blocking)
        try:
            from ranking.elo_engine import get_elo_engine
            elo = get_elo_engine()
            if payload.consensus_scores:
                top = payload.consensus_scores[0]
                for score in payload.consensus_scores[1:]:
                    elo.record_battle_result(
                        winner_id=top.model,
                        loser_id=score.model,
                        score_diff=top.composite_score - score.composite_score,
                    )
        except Exception as elo_exc:
            logger.warning(f"ELO update skipped: {elo_exc}")

        # Cache the debate result (TTL: 30 minutes)
        try:
            result["cache_hit"] = False
            _cache.set(_cache_key, result, ttl=1800)
        except Exception as cache_exc:
            logger.warning(f"Debate cache store failed: {cache_exc}")

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Debate execution failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
