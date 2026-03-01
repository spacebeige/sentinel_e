"""
============================================================
Meta-Cognitive Orchestrator — FastAPI Routes
============================================================
Exposes the distributed cognitive architecture as API endpoints.
Integrates with the existing Sentinel-E gateway.

Routes:
  POST /api/mco/run         — Standard/Experimental execution
  POST /api/mco/experimental — Experimental mode (full exposure)
  GET  /api/mco/session/{id} — Session state inspection
  GET  /api/mco/graph/{id}   — Knowledge graph subgraph
  GET  /api/mco/models       — Available model registry
  GET  /api/mco/daemon/status — Background daemon status
  POST /api/mco/daemon/start  — Start background daemon
  POST /api/mco/daemon/stop   — Stop background daemon
============================================================
"""

import logging
from typing import Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth import get_current_user
from database.connection import get_db
from database.crud import (
    create_chat, get_chat, add_message, update_chat_metadata,
)
from utils.chat_naming import generate_chat_name

from metacognitive.schemas import (
    OperatingMode,
    OrchestratorRequest,
    OrchestratorResponse,
)
from metacognitive.orchestrator import MetaCognitiveOrchestrator
from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
from metacognitive.background_daemon import BackgroundDaemon

logger = logging.getLogger("MCO-Routes")

# ── Router ───────────────────────────────────────────────────
router = APIRouter(prefix="/api/mco", tags=["Meta-Cognitive Orchestrator"])

# ── Global references (set during app startup) ──────────────
_orchestrator: Optional[MetaCognitiveOrchestrator] = None
_daemon: Optional[BackgroundDaemon] = None
_cognitive_engine = None  # CognitiveCoreEngine for debate mode


def set_orchestrator(orch: MetaCognitiveOrchestrator):
    global _orchestrator
    _orchestrator = orch


def set_daemon(daemon: BackgroundDaemon):
    global _daemon
    _daemon = daemon


def set_cognitive_engine(engine):
    """Wire the CognitiveCoreEngine for debate sub-mode delegation."""
    global _cognitive_engine
    _cognitive_engine = engine


def _get_orchestrator() -> MetaCognitiveOrchestrator:
    if not _orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Meta-Cognitive Orchestrator not initialized",
        )
    return _orchestrator


# ============================================================
# MAIN EXECUTION
# ============================================================

@router.post("/run")
async def mco_run(
    query: str = Body(...),
    mode: str = Body("standard"),
    sub_mode: Optional[str] = Body(None),
    chat_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None),
    force_retrieval: bool = Body(False),
    selected_model: Optional[str] = Body(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """
    Main Meta-Cognitive Orchestrator execution endpoint.
    Routes through the mandatory 10-step protocol.

    Standard Mode: Returns highest-scoring aggregated answer.
    Experimental Mode: Returns all outputs with full scoring breakdown.
    Single Model Focus: Only the selected model executes.
    """
    orch = _get_orchestrator()

    # Validate selected_model if provided
    if selected_model:
        if selected_model not in COGNITIVE_MODEL_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {selected_model}",
            )
        spec = COGNITIVE_MODEL_REGISTRY[selected_model]
        if not spec.enabled:
            raise HTTPException(
                status_code=400,
                detail=f"Model not enabled: {selected_model}",
            )

    # Resolve operating mode
    try:
        op_mode = OperatingMode(mode)
    except ValueError:
        op_mode = OperatingMode.STANDARD

    # Resolve chat
    chat = None
    if chat_id:
        try:
            chat = await get_chat(db, UUID(chat_id))
        except (ValueError, Exception):
            pass

    if not chat:
        chat_name = generate_chat_name(query, f"mco-{mode}")
        chat = await create_chat(
            db, chat_name, f"mco-{mode}",
            user_id=user["user_id"],
        )

    await add_message(db, chat.id, "user", query)

    # ══════════════════════════════════════════════════════════
    # DEBATE MODE DELEGATION
    # When sub_mode is "debate" and CognitiveOrchestrator is available,
    # delegate to it for multi-round structured debate (StructuredDebateEngine).
    # This produces real rounds with rebuttals, drift/rift metrics, etc.
    # ══════════════════════════════════════════════════════════
    effective_sub_mode = sub_mode
    if effective_sub_mode == "debate" and _cognitive_engine is not None:
        logger.info(f"Debate mode: delegating to CognitiveOrchestrator for chat {chat.id}")
        try:
            from core.ensemble_schemas import EnsembleFailure
            ensemble_response = await _cognitive_engine.process(
                query=query,
                chat_id=str(chat.id),
                rounds=3,
            )
        except EnsembleFailure as ef:
            logger.error(f"Debate ensemble hard failure: {ef}")
            ensemble_response = ef.to_response()
        except Exception as debate_err:
            logger.error(f"Debate delegation failed, falling back to MCO: {debate_err}")
            ensemble_response = None

        if ensemble_response is not None:
            payload = ensemble_response.to_frontend_payload()
            formatted_output = ensemble_response.formatted_output

            # Persist
            await add_message(db, chat.id, "assistant", formatted_output)
            await update_chat_metadata(
                db, chat.id,
                priority_answer=formatted_output,
                machine_metadata={
                    "engine": "CognitiveCoreEngine",
                    "mode": "debate",
                    "sub_mode": "debate",
                    "models_executed": ensemble_response.models_executed,
                    "debate_rounds": ensemble_response.debate_result.total_rounds,
                },
                rounds=ensemble_response.debate_result.total_rounds,
            )

            confidence = ensemble_response.confidence.final_confidence
            ens_entropy = ensemble_response.ensemble_metrics.disagreement_entropy
            ens_fragility = ensemble_response.ensemble_metrics.fragility_score

            omega_metadata = payload.get("omega_metadata", {})
            omega_metadata.update({
                "version": "7.1.0-cognitive",
                "mode": "debate",
                "sub_mode": "debate",
                "confidence": confidence,
                "entropy": ens_entropy,
                "fragility": ens_fragility,
                "fragility_index": ens_fragility,
                "ensemble_metrics": payload.get("ensemble_metrics", {}),
                "debate_result": payload.get("debate_result", {}),
                "debate_rounds": payload.get("debate_rounds", []),
                "model_outputs": payload.get("model_outputs", []),
                "agreement_matrix": payload.get("agreement_matrix", {}),
                "drift_metrics": payload.get("drift_metrics", {}),
                "tactical_map": payload.get("tactical_map", []),
                "confidence_graph": payload.get("confidence_graph", payload.get("calibrated_confidence", {})),
                "session_intelligence": payload.get("session_intelligence", {}),
                "session_analytics": payload.get("session_analytics", {}),
                "model_status": payload.get("model_status", []),
                "reasoning_trace": {
                    "engine": "CognitiveCoreEngine",
                    "pipeline": "cognitive_v7_debate",
                    "models_executed": ensemble_response.models_executed,
                    "models_succeeded": ensemble_response.models_succeeded,
                    "models_failed": ensemble_response.models_failed,
                    "debate_rounds": ensemble_response.debate_result.total_rounds,
                },
            })

            # Update MCO session analytics
            if _orchestrator and hasattr(_orchestrator, 'session_engine'):
                try:
                    _orchestrator.session_engine.update_analytics(
                        session_id=str(chat.id),
                        mode="debate",
                        drift_value=ensemble_response.debate_result.drift_index,
                        rift_value=ensemble_response.debate_result.rift_index,
                        disagreement_value=ens_entropy,
                    )
                except Exception:
                    pass

            return {
                "chat_id": str(chat.id),
                "session_id": str(chat.id),
                "mode": "debate",
                "sub_mode": "debate",
                "formatted_output": formatted_output,
                "aggregated_answer": formatted_output,
                "confidence": round(confidence, 4),
                "data": {"priority_answer": formatted_output},
                "omega_metadata": omega_metadata,
                "session_state": {
                    "session_id": str(chat.id),
                    "debate_rounds": ensemble_response.debate_result.total_rounds,
                    "drift_score": round(ensemble_response.debate_result.drift_index, 4),
                    "message_count": ensemble_response.session_intelligence.message_count,
                    "boundary_history_count": ensemble_response.session_intelligence.boundary_hits,
                    "reasoning_depth": ensemble_response.session_intelligence.depth or "N/A",
                },
                "boundary_result": {
                    "risk_level": (
                        "LOW" if confidence > 0.7
                        else "MEDIUM" if confidence > 0.4
                        else "HIGH"
                    ),
                    "severity_score": int((1 - confidence) * 100),
                },
                "models_executed": ensemble_response.models_executed,
                "models_succeeded": ensemble_response.models_succeeded,
                "models_failed": ensemble_response.models_failed,
            }

    # ══════════════════════════════════════════════════════════
    # STANDARD MCO PIPELINE (non-debate modes)
    # ══════════════════════════════════════════════════════════

    # Build request
    request = OrchestratorRequest(
        session_id=session_id or str(chat.id),
        query=query,
        mode=op_mode,
        sub_mode=sub_mode,
        chat_id=str(chat.id),
        force_retrieval=force_retrieval,
        selected_model=selected_model,
    )

    # Execute 10-step protocol
    try:
        response = await orch.process(request)
    except RuntimeError as e:
        logger.error(f"MCO execution error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"MCO execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # Guard: no blank responses
    if not response.aggregated_answer or not response.aggregated_answer.strip():
        raise HTTPException(
            status_code=502,
            detail=f"Model '{response.winning_model}' returned empty output",
        )

    # Persist assistant response
    await add_message(db, chat.id, "assistant", response.aggregated_answer)
    await update_chat_metadata(
        db, chat.id,
        priority_answer=response.aggregated_answer,
        machine_metadata={
            "mco_version": "1.0.0",
            "mode": response.mode.value,
            "winning_model": response.winning_model,
            "winning_score": response.winning_score,
            "drift_score": response.drift_score,
            "volatility_score": response.volatility_score,
            "refinement_cycles": response.refinement_cycles,
            "latency_ms": response.latency_ms,
            "model_count": len(response.all_results),
        },
        rounds=1,
    )

    # Build API response
    result = {
        "chat_id": str(chat.id),
        "session_id": response.session_id,
        "mode": response.mode.value,
        "sub_mode": response.sub_mode or sub_mode,
        "aggregated_answer": response.aggregated_answer,
        "formatted_output": response.aggregated_answer,
        "winning_model": response.winning_model,
        "winning_score": round(response.winning_score, 4),
        "drift_score": round(response.drift_score, 4),
        "volatility_score": round(response.volatility_score, 4),
        "refinement_cycles": response.refinement_cycles,
        "latency_ms": round(response.latency_ms, 1),
        "knowledge_bundle_size": len(response.knowledge_bundle),
        "retrieval_confidence": round(response.retrieval_confidence, 4),
        "selected_model": selected_model,
        "confidence": round(response.winning_score, 4),
    }

    # ── Build omega_metadata (unified frontend contract) ────
    all_outputs_serialized = [
        {
            "model_name": r.output.model_name,
            "raw_output": r.output.raw_output,
            "tokens_used": r.output.tokens_used,
            "latency_ms": round(r.output.latency_ms, 1),
            "success": r.output.success,
            "error": r.output.error,
            "score": {
                "topic_alignment": round(r.score.topic_alignment, 4),
                "knowledge_grounding": round(r.score.knowledge_grounding, 4),
                "specificity": round(r.score.specificity, 4),
                "confidence_calibration": round(r.score.confidence_calibration, 4),
                "drift_penalty": round(r.score.drift_penalty, 4),
                "final_score": round(r.score.final_score, 4),
            },
        }
        for r in response.all_results
    ]

    scoring_serialized = [
        {
            "model": s.model_name,
            "T": round(s.topic_alignment, 4),
            "K": round(s.knowledge_grounding, 4),
            "S": round(s.specificity, 4),
            "C": round(s.confidence_calibration, 4),
            "D": round(s.drift_penalty, 4),
            "final": round(s.final_score, 4),
        }
        for s in (response.scoring_breakdown or [])
    ]

    divergence = response.divergence_metrics or {}

    omega_metadata = {
        "mode": response.mode.value,
        "sub_mode": response.sub_mode or sub_mode,
        "confidence": round(response.winning_score, 4),
        "winning_model": response.winning_model,
        "model_count": len(response.all_results),
        "latency_ms": round(response.latency_ms, 1),
        "drift_score": round(response.drift_score, 4),
        "volatility_score": round(response.volatility_score, 4),
        "session_state": {
            "session_id": response.session_id,
            "refinement_cycles": response.refinement_cycles,
            "drift_score": round(response.drift_score, 4),
            "volatility_score": round(response.volatility_score, 4),
            "inferred_domain": divergence.get("domain_classification", None),
        },
        "all_outputs": all_outputs_serialized,
        "scoring_breakdown": scoring_serialized,
        "divergence_metrics": divergence,
    }

    # Build sub-mode-specific structured results for frontend components
    effective_sub_mode = response.sub_mode or sub_mode

    if response.mode == OperatingMode.EXPERIMENTAL or effective_sub_mode:
        # Build debate_result (DebateView/DebateArena consumes this)
        # PHASE 6: Filter out empty/failed model outputs from debate positions
        _valid_results = [
            r for r in response.all_results
            if r.output.success and r.output.raw_output and r.output.raw_output.strip()
        ]
        omega_metadata["debate_result"] = {
            "rounds": [[
                {
                    "model_id": r.output.model_name,
                    "model_label": r.output.model_name,
                    "model_name": r.output.model_name,
                    "model_color": "",
                    "round_num": 1,
                    "position": r.output.raw_output[:300] if r.output.raw_output else "",
                    "argument": r.output.raw_output,
                    "assumptions": [],
                    "risks": [],
                    "rebuttals": "",
                    "position_shift": "none",
                    "weaknesses_found": "",
                    "confidence": round(r.score.final_score, 4),
                    "latency_ms": round(r.output.latency_ms, 2) if hasattr(r.output, 'latency_ms') else 0.0,
                    "role": r.output.model_name,
                }
                for r in _valid_results
            ]],
            "models_used": [r.output.model_name for r in _valid_results],
            "scores": {
                s.model_name: round(s.final_score, 4)
                for s in (response.scoring_breakdown or [])
            },
            "analysis": {
                "synthesis": response.aggregated_answer[:500] if response.aggregated_answer else "",
                "conflict_axes": [],
                "disagreement_strength": divergence.get("max_divergence", 0),
                "convergence_level": divergence.get("convergence", "moderate"),
                "convergence_detail": "",
                "logical_stability": 0.5,
                "strongest_argument": response.winning_model or "",
                "weakest_argument": "",
                "confidence_recalibration": round(response.winning_score, 4) if response.winning_score else 0.5,
                "drift_index": round(response.drift_score, 4),
                "rift_index": 0.0,
                "confidence_spread": 0.0,
                "fragility_score": 0.0,
                "per_model_drift": {},
                "per_round_rift": [],
                "per_round_disagreement": [],
                "overall_confidence": round(
                    sum(r.score.final_score for r in _valid_results) / len(_valid_results), 4
                ) if _valid_results else 0.5,
            },
        }

        # Build aggregation_result (standard structured display)
        omega_metadata["aggregation_result"] = {
            "winner": response.winning_model,
            "winner_score": round(response.winning_score, 4),
            "answer": response.aggregated_answer,
            "model_scores": {
                s.model_name: round(s.final_score, 4)
                for s in (response.scoring_breakdown or [])
            },
        }

        # Build forensic_result (EvidenceConsole consumes this)
        omega_metadata["forensic_result"] = {
            "models_analyzed": len(response.all_results),
            "scoring_breakdown": scoring_serialized,
            "divergence": divergence,
            "winning_model": response.winning_model,
            "winning_score": round(response.winning_score, 4),
        }

        # Build audit_result (GlassConsole consumes this)
        omega_metadata["audit_result"] = {
            "all_outputs": all_outputs_serialized,
            "scoring_breakdown": scoring_serialized,
            "divergence_metrics": divergence,
            "drift_score": round(response.drift_score, 4),
            "volatility_score": round(response.volatility_score, 4),
            "refinement_cycles": response.refinement_cycles,
        }

    elif response.mode == OperatingMode.STANDARD:
        # Standard mode: minimal aggregation_result
        omega_metadata["aggregation_result"] = {
            "winner": response.winning_model,
            "winner_score": round(response.winning_score, 4),
            "answer": response.aggregated_answer,
            "model_scores": {
                s.model_name: round(s.final_score, 4)
                for s in (response.scoring_breakdown or [])
            },
        }

    result["omega_metadata"] = omega_metadata
    result["data"] = {"priority_answer": response.aggregated_answer}
    result["session_state"] = omega_metadata["session_state"]
    result["boundary_result"] = {
        "severity_score": 0,
        "flags": [],
    }

    # Single Model Focus: add focus_model identifier
    if selected_model:
        result["focus_model"] = response.winning_model

    # Experimental mode: also expose flat all_outputs for backward compat
    if response.mode == OperatingMode.EXPERIMENTAL:
        result["all_outputs"] = all_outputs_serialized
        result["divergence_metrics"] = divergence
        result["scoring_breakdown"] = scoring_serialized

    return result


@router.post("/experimental")
async def mco_experimental(
    query: str = Body(...),
    sub_mode: Optional[str] = Body(None),
    chat_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None),
    force_retrieval: bool = Body(False),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """
    Convenience endpoint for Experimental Mode.
    All models run in parallel. No arbitration override.
    All outputs displayed. Full scoring metrics exposed.
    """
    return await mco_run(
        query=query,
        mode="experimental",
        sub_mode=sub_mode,
        chat_id=chat_id,
        session_id=session_id,
        force_retrieval=force_retrieval,
        db=db,
        user=user,
    )


# ============================================================
# SESSION INSPECTION
# ============================================================

@router.get("/session/{session_id}")
async def mco_session_state(
    session_id: str,
    user: Dict = Depends(get_current_user),
):
    """Inspect Meta-Cognitive session state."""
    orch = _get_orchestrator()
    session = orch.session_engine.get_session(session_id)

    if not session:
        # Try Redis
        session = await orch.session_engine.restore_from_redis(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "mode": session.mode.value,
        "turn_count": session.turn_count,
        "drift_score": round(session.drift_score, 4),
        "volatility_score": round(session.volatility_score, 4),
        "refinement_cycles": session.refinement_cycles,
        "active_goals": [
            {"id": g.id, "description": g.description, "status": g.status}
            for g in session.structured_goals
        ],
        "unresolved_questions": [
            {"id": q.id, "question": q.question, "priority": q.priority, "attempts": q.attempts}
            for q in session.unresolved_questions
        ],
        "memory_block_count": len(session.memory_blocks),
        "behavioral_history_count": len(session.behavioral_history),
        "has_centroid": bool(session.topic_centroid_embedding),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


@router.get("/session/{session_id}/analytics")
async def mco_session_analytics(
    session_id: str,
    user: Dict = Depends(get_current_user),
):
    """Get rich session analytics including drift/rift trends, latency history, and confidence metrics."""
    orch = _get_orchestrator()
    analytics = orch.session_engine.get_session_analytics(session_id)

    if not analytics:
        raise HTTPException(status_code=404, detail="Session not found or no analytics available")

    return analytics


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

@router.get("/graph/{session_id}")
async def mco_knowledge_graph(
    session_id: str,
    user: Dict = Depends(get_current_user),
):
    """Get knowledge graph subgraph for a session."""
    orch = _get_orchestrator()
    subgraph = orch.knowledge_graph.get_session_subgraph(session_id)
    stats = orch.knowledge_graph.stats()

    return {
        "session_id": session_id,
        "subgraph": subgraph,
        "global_stats": stats,
    }


# ============================================================
# MODEL REGISTRY
# ============================================================

@router.get("/models")
async def mco_models(user: Dict = Depends(get_current_user)):
    """List available cognitive models with enabled/disabled status."""
    return {
        "models": [
            {
                "key": key,
                "name": spec.name,
                "model_id": spec.model_id,
                "provider": spec.provider,
                "role": spec.role.value,
                "context_window": spec.context_window,
                "max_output_tokens": spec.max_output_tokens,
                "active": spec.active,
                "enabled": spec.enabled,
            }
            for key, spec in COGNITIVE_MODEL_REGISTRY.items()
        ]
    }


# ============================================================
# BACKGROUND DAEMON
# ============================================================

@router.get("/daemon/status")
async def daemon_status(user: Dict = Depends(get_current_user)):
    """Get background daemon status."""
    if not _daemon:
        return {"status": "not_configured"}
    return {
        "running": _daemon.is_running,
        "interval_seconds": _daemon.interval,
        "iterations": _daemon._iterations,
    }


@router.post("/daemon/start")
async def daemon_start(user: Dict = Depends(get_current_user)):
    """Start the background daemon."""
    if not _daemon:
        raise HTTPException(status_code=503, detail="Daemon not configured")
    if _daemon.is_running:
        return {"status": "already_running"}
    _daemon.start()
    return {"status": "started"}


@router.post("/daemon/stop")
async def daemon_stop(user: Dict = Depends(get_current_user)):
    """Stop the background daemon."""
    if not _daemon:
        raise HTTPException(status_code=503, detail="Daemon not configured")
    if not _daemon.is_running:
        return {"status": "not_running"}
    _daemon.stop()
    return {"status": "stopped"}


# ============================================================
# BEHAVIORAL ANALYTICS
# ============================================================

@router.get("/analytics/{session_id}")
async def mco_analytics(
    session_id: str,
    user: Dict = Depends(get_current_user),
):
    """Get behavioral analytics for a session."""
    orch = _get_orchestrator()
    session = orch.session_engine.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Aggregate behavioral data
    model_stats = {}
    for record in session.behavioral_history:
        name = record.model_name
        if name not in model_stats:
            model_stats[name] = {
                "invocations": 0,
                "avg_score": 0.0,
                "avg_grounding": 0.0,
                "avg_specificity": 0.0,
                "total_drift": 0.0,
            }
        stats = model_stats[name]
        stats["invocations"] += 1
        stats["avg_score"] = (
            (stats["avg_score"] * (stats["invocations"] - 1) + record.final_score)
            / stats["invocations"]
        )
        stats["avg_grounding"] = (
            (stats["avg_grounding"] * (stats["invocations"] - 1) + record.grounding_score)
            / stats["invocations"]
        )
        stats["avg_specificity"] = (
            (stats["avg_specificity"] * (stats["invocations"] - 1) + record.specificity)
            / stats["invocations"]
        )
        stats["total_drift"] += record.drift_penalty

    return {
        "session_id": session_id,
        "turn_count": session.turn_count,
        "model_performance": model_stats,
        "drift_history": [
            {
                "model": r.model_name,
                "timestamp": r.timestamp,
                "score": round(r.final_score, 4),
                "drift": round(r.drift_penalty, 4),
            }
            for r in session.behavioral_history[-20:]
        ],
    }
