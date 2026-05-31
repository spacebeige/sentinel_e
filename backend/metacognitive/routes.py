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
import traceback
from typing import Dict, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth import get_current_user, get_optional_user
from database.connection import get_db
from database.crud import (
    create_chat, get_chat, add_message, update_chat_metadata, get_chat_messages,
)
from utils.chat_naming import generate_chat_name
from utils.output_sanitizer import sanitize_output, sanitize_json_response
from core.context_builder import get_context_builder
from core.document_cognition import build_document_cognition

from metacognitive.schemas import (
    OperatingMode,
    OrchestratorRequest,
    OrchestratorResponse,
)
from metacognitive.orchestrator import MetaCognitiveOrchestrator
from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
import metacognitive.cognitive_gateway as cg
from metacognitive.background_daemon import BackgroundDaemon

logger = logging.getLogger("MCO-Routes")

# ── Router ───────────────────────────────────────────────────
router = APIRouter(prefix="/api/mco", tags=["Meta-Cognitive Orchestrator"])

# ── Global references (set during app startup) ──────────────
_orchestrator: Optional[MetaCognitiveOrchestrator] = None
_daemon: Optional[BackgroundDaemon] = None
_cognitive_engine = None  # CognitiveCoreEngine for debate mode

try:
    from core.orchestration_run import create_orchestration_run, CognitivePhase
    from core.runtime_event_bus import create_run_bus
    from core.cognitive_artifact_builder import build_cognitive_artifact
    from memory.layered_memory import get_deliberative_memory, get_tactical_memory
    _COGNITIVE_RUNTIME_ENABLED = True
except ImportError as runtime_import_error:
    logger.warning("Cognitive runtime bridge unavailable in MCO routes: %s", runtime_import_error)
    create_orchestration_run = None
    CognitivePhase = None
    create_run_bus = None
    build_cognitive_artifact = None
    get_deliberative_memory = None
    get_tactical_memory = None
    _COGNITIVE_RUNTIME_ENABLED = False

try:
    from memory.behavioral_memory import BehavioralMemoryManager
except ImportError:
    BehavioralMemoryManager = None


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


def _sanitize_mco_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize MCO response to remove internal reasoning tags.
    
    Cleans:
      - formatted_output
      - aggregated_answer
      - raw_output (in nested structures)
      - Various sub-results (forensic, audit, synthesis, etc.)
    """
    if not isinstance(response, dict):
        return response
    
    # Clone to avoid mutating original
    response = dict(response)
    
    # Sanitize top-level string fields
    if "formatted_output" in response and isinstance(response["formatted_output"], str):
        response["formatted_output"] = sanitize_output(response["formatted_output"])
        logger.debug(f"Sanitized formatted_output, length: {len(response['formatted_output'])}")
    
    if "aggregated_answer" in response and isinstance(response["aggregated_answer"], str):
        response["aggregated_answer"] = sanitize_output(response["aggregated_answer"])
    
    if "answer" in response and isinstance(response["answer"], str):
        response["answer"] = sanitize_output(response["answer"])
    
    # Sanitize nested omega_metadata
    if "omega_metadata" in response and isinstance(response["omega_metadata"], dict):
        omega = dict(response["omega_metadata"])
        
        # Sanitize various sub-results
        if "forensic_result" in omega and isinstance(omega["forensic_result"], dict):
            if "answer" in omega["forensic_result"]:
                omega["forensic_result"]["answer"] = sanitize_output(omega["forensic_result"]["answer"])
        
        if "audit_result" in omega and isinstance(omega["audit_result"], dict):
            # Sanitize all outputs
            if "all_outputs" in omega["audit_result"] and isinstance(omega["audit_result"]["all_outputs"], list):
                sanitized_outputs = []
                for out in omega["audit_result"]["all_outputs"]:
                    if isinstance(out, dict):
                        out_copy = dict(out)
                        if "raw_output" in out_copy:
                            out_copy["raw_output"] = sanitize_output(out_copy["raw_output"])
                        sanitized_outputs.append(out_copy)
                    else:
                        sanitized_outputs.append(out)
                omega["audit_result"]["all_outputs"] = sanitized_outputs
        
        if "synthesis_result" in omega and isinstance(omega["synthesis_result"], dict):
            syn = omega["synthesis_result"]
            if "claude_synthesis" in syn:
                omega["synthesis_result"]["claude_synthesis"] = sanitize_output(syn["claude_synthesis"])
            if "refined_output" in syn:
                omega["synthesis_result"]["refined_output"] = sanitize_output(syn["refined_output"])
        
        # Sanitize all_outputs at top level of omega_metadata
        if "all_outputs" in omega and isinstance(omega["all_outputs"], list):
            sanitized_outputs = []
            for out in omega["all_outputs"]:
                if isinstance(out, dict):
                    out_copy = dict(out)
                    if "raw_output" in out_copy:
                        out_copy["raw_output"] = sanitize_output(out_copy["raw_output"])
                    sanitized_outputs.append(out_copy)
                else:
                    sanitized_outputs.append(out)
            omega["all_outputs"] = sanitized_outputs
        
        # Sanitize debate results if present
        if "debate_result" in omega and isinstance(omega["debate_result"], dict):
            # Keep debate_result as-is (structured data)
            pass
        
        response["omega_metadata"] = omega
    
    # Sanitize data payload
    if "data" in response and isinstance(response["data"], dict):
        data = dict(response["data"])
        if "priority_answer" in data:
            data["priority_answer"] = sanitize_output(data["priority_answer"])
        response["data"] = data
    
    # Sanitize all_outputs at top level (experimental mode)
    if "all_outputs" in response and isinstance(response["all_outputs"], list):
        sanitized_outputs = []
        for out in response["all_outputs"]:
            if isinstance(out, dict):
                out_copy = dict(out)
                if "raw_output" in out_copy:
                    out_copy["raw_output"] = sanitize_output(out_copy["raw_output"])
                sanitized_outputs.append(out_copy)
            else:
                sanitized_outputs.append(out)
        response["all_outputs"] = sanitized_outputs
    
    return response


def _start_runtime_run(chat_id: str, user_id: str, query: str, execution_path: str = "pending"):
    if not _COGNITIVE_RUNTIME_ENABLED:
        return None, None

    try:
        run = create_orchestration_run(
            chat_id=str(chat_id),
            user_id=user_id,
            query_preview=(query or "")[:80],
            execution_path=execution_path,
        )
        bus = create_run_bus(run.orchestration_run_id)
        _transition_runtime(run, bus, CognitivePhase.OBSERVE, {"source": "api.mco.run"})
        _transition_runtime(run, bus, CognitivePhase.ANALYZE, {"source": "api.mco.run"})
        return run, bus
    except Exception as runtime_err:
        logger.debug("[MCO Runtime] Failed to start runtime run: %s", runtime_err)
        return None, None


def _transition_runtime(run, bus, phase, payload: Optional[Dict[str, Any]] = None) -> None:
    if run is None or phase is None:
        return

    payload = payload or {}
    try:
        run.transition_to(phase, payload)
        if bus is not None:
            bus.publish({
                "event_type": "phase_transition",
                "phase": phase.value,
                "phase_label": run.phase_label,
                "payload": payload,
                "run_id": run.orchestration_run_id,
            })
    except Exception as runtime_err:
        logger.debug("[MCO Runtime] Phase transition failed: %s", runtime_err)


def _record_runtime_memory(run, bus, layer: str, key: str, preview: str, relevance_score: float = 1.0) -> None:
    if run is None:
        return

    try:
        _transition_runtime(run, bus, CognitivePhase.RETRIEVE_MEMORY, {"layer": layer, "key": key})
        run.record_memory_retrieval(
            layer=layer,
            key=key,
            content_preview=(preview or "")[:200],
            relevance_score=relevance_score,
        )
    except Exception as runtime_err:
        logger.debug("[MCO Runtime] Memory recording failed: %s", runtime_err)


def _record_runtime_routing(
    run,
    bus,
    *,
    execution_path: str,
    reason: str,
    query_complexity: str,
    selected_model: Optional[str] = None,
    debate_requested: bool = False,
    cache_hit: bool = False,
) -> None:
    if run is None:
        return

    decision = {
        "path": execution_path,
        "reason": reason,
        "query_complexity": query_complexity,
        "selected_model": selected_model,
        "debate_requested": debate_requested,
        "cache_hit": cache_hit,
    }
    try:
        _transition_runtime(run, bus, CognitivePhase.ROUTE, decision)
        run.record_routing_decision(decision)
    except Exception as runtime_err:
        logger.debug("[MCO Runtime] Routing recording failed: %s", runtime_err)


def _attach_runtime_metadata(result: Dict[str, Any], run, artifact: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if run is None:
        return result

    omega_metadata = dict(result.get("omega_metadata", {}) or {})
    omega_metadata["orchestration_run"] = run.to_frontend_dict()
    if artifact:
        omega_metadata["cognitive_artifact"] = artifact
    result["omega_metadata"] = omega_metadata
    return result


def _complete_runtime_run(run, bus, *, reflection: str = "") -> None:
    if run is None:
        return

    try:
        if reflection:
            _transition_runtime(run, bus, CognitivePhase.REFLECT)
            run.record_reflection(reflection)
        _transition_runtime(run, bus, CognitivePhase.STORE_SNAPSHOT)
        run.mark_completed()
    except Exception as runtime_err:
        logger.debug("[MCO Runtime] Completion failed: %s", runtime_err)
    finally:
        if bus is not None:
            bus.close()


def _fail_runtime_run(run, bus, error: str, error_code: str = "MCO_RUNTIME_ERROR") -> None:
    if run is None:
        return

    try:
        run.mark_failed(error, error_code=error_code)
        if bus is not None:
            bus.publish({
                "event_type": "orchestration_failed",
                "phase": run.cognitive_phase.value,
                "error": error,
                "error_code": error_code,
                "run_id": run.orchestration_run_id,
            })
            bus.close()
    except Exception as runtime_err:
        logger.debug("[MCO Runtime] Failure handling failed: %s", runtime_err)


def _build_mco_cognitive_artifact(response, run) -> Dict[str, Any]:
    all_results = list(response.all_results or [])
    divergence = response.divergence_metrics or {}
    knowledge_bundle = list(response.knowledge_bundle or [])
    alternatives = []

    for result in all_results[:6]:
        raw_output = getattr(result.output, "raw_output", "") or ""
        if not raw_output.strip():
            continue
        alternatives.append({
            "model_id": result.output.model_name,
            "model_name": result.output.model_name,
            "position": raw_output[:280],
            "confidence": round(getattr(result.score, "final_score", 0.0), 4),
        })

    winner_model = response.winning_model or ""
    non_winner_alternatives = [item for item in alternatives if item["model_name"] != winner_model][:3]
    contradiction_density = float(divergence.get("max_divergence", 0.0) or 0.0)
    convergence = divergence.get("convergence", "moderate") or "moderate"
    reliability = "high" if response.winning_score >= 0.8 else "medium" if response.winning_score >= 0.55 else "low"
    participation = sum(1 for result in all_results if getattr(result.output, "success", False)) / max(len(all_results), 1)

    return {
        "primary_conclusion": (response.aggregated_answer or "")[:500],
        "reasoning_topology": [
            {
                "model_id": result.output.model_name,
                "model_name": result.output.model_name,
                "confidence": round(getattr(result.score, "final_score", 0.0), 4),
                "reasoning_steps": [sanitize_output((getattr(result.output, "raw_output", "") or "")[:240])],
                "assumptions": [],
                "vulnerabilities": [],
            }
            for result in all_results[:5]
            if (getattr(result.output, "raw_output", "") or "").strip()
        ],
        "evidence_matrix": [
            {
                "type": "knowledge_block",
                "title": kb.source or f"Knowledge {index + 1}",
                "content_preview": (kb.content or "")[:180],
                "reliability": round(getattr(kb, "confidence", 0.0), 4),
            }
            for index, kb in enumerate(knowledge_bundle[:5])
        ],
        "contradiction_analysis": {
            "density": round(contradiction_density, 4),
            "severity": "high" if contradiction_density > 0.65 else "medium" if contradiction_density > 0.35 else "low",
            "summary": f"MCO convergence={convergence}; drift={response.drift_score:.2f}; volatility={response.volatility_score:.2f}.",
            "unresolved_conflicts": [item["position"][:160] for item in non_winner_alternatives[:3]],
            "resolved_conflicts": [],
            "requires_verification": contradiction_density > 0.5 or response.volatility_score > 0.5,
        },
        "alternative_perspectives": non_winner_alternatives,
        "verification_results": {
            "stability_index": round(max(0.0, 1.0 - contradiction_density), 4),
            "fragility_score": round(min(1.0, response.volatility_score), 4),
            "calibration_method": "mco_arbitration",
            "reliability_assessment": reliability,
            "verification_passed": contradiction_density < 0.6,
        },
        "confidence_evolution": [
            {"phase": "route", "value": round(max(0.25, response.winning_score * 0.72), 4), "method": "heuristic"},
            {"phase": "synthesize", "value": round(response.winning_score, 4), "method": "mco_arbitration"},
        ],
        "reflective_cognition": (
            f"The MCO pipeline completed {response.refinement_cycles} refinement cycle(s) with "
            f"{len(all_results)} model result(s). Confidence settled at {response.winning_score:.2f}."
        ),
        "memory_continuity_links": [
            {
                "layer": retrieval.get("layer", "unknown"),
                "key": retrieval.get("key", ""),
                "preview": (retrieval.get("content_preview") or retrieval.get("preview") or "")[:100],
                "relevance": round(float(retrieval.get("relevance_score", retrieval.get("relevance", 1.0)) or 1.0), 4),
            }
            for retrieval in (run.to_frontend_dict().get("memory_retrievals", []) if run else [])[:10]
        ],
        "orchestration_identity": run.to_summary() if run else {},
        "quality_indicators": {
            "models_executed": len(all_results),
            "models_succeeded": sum(1 for result in all_results if getattr(result.output, "success", False)),
            "participation_rate": round(participation, 4),
            "debate_rounds": 0,
            "stability_index": round(max(0.0, 1.0 - contradiction_density), 4),
            "response_grade": (
                "A" if participation >= 0.8 and response.winning_score >= 0.8 else
                "B" if participation >= 0.6 and response.winning_score >= 0.6 else
                "C" if participation >= 0.4 else
                "D"
            ),
        },
    }



# ============================================================
# MAIN EXECUTION
# ============================================================

@router.post("/run")
async def mco_run(
    background_tasks: BackgroundTasks,
    payload: Optional[Dict[str, Any]] = Body(None),
    db: AsyncSession = Depends(get_db),
    user: Optional[Dict] = Depends(get_optional_user),
):
    """
    Safe wrapper for Meta-Cognitive Orchestrator execution.
    Ensures malformed payloads / auth gaps never crash the endpoint.
    """
    logger.info("Entering /api/mco/run")
    logger.info("/api/mco/run user authenticated=%s", bool(user))

    try:
        if not isinstance(payload, dict):
            logger.warning("/api/mco/run invalid payload type: %s", type(payload).__name__)
            return JSONResponse(status_code=400, content={"detail": "Invalid payload. Expected JSON object."})

        logger.info("/api/mco/run payload keys=%s", list(payload.keys()))

        registry = COGNITIVE_MODEL_REGISTRY
        if not registry or not hasattr(registry, "keys"):
            logger.warning("/api/mco/run registry unavailable or invalid: %s", type(registry).__name__)
            return JSONResponse(
                status_code=503,
                content={"detail": "Model registry unavailable.", "models": []},
            )
        logger.info("/api/mco/run registry keys=%s", list(registry.keys()))

        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return JSONResponse(status_code=400, content={"detail": "Field 'query' is required and must be a non-empty string."})

        mode = payload.get("mode", "standard")
        sub_mode = payload.get("sub_mode")
        chat_id = payload.get("chat_id")
        session_id = payload.get("session_id")
        force_retrieval = bool(payload.get("force_retrieval", False))
        selected_model = payload.get("selected_model") or payload.get("model")
        image_b64 = payload.get("image_b64")
        image_mime = payload.get("image_mime")
        rounds = int(payload.get("rounds") or 3)
        response_style = payload.get("response_style")
        preferences = payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {}

        safe_user = user or {
            "id": "00000000-0000-0000-0000-000000000000",
            "user_id": "00000000-0000-0000-0000-000000000000",
            "email": "anonymous@local",
            "name": "Anonymous User",
            "role": "guest",
            "provider": "system",
            "authenticated": False,
        }

        if not user:
            try:
                from gateway.auth_v2 import ensure_user_exists
                await ensure_user_exists(safe_user, db)
            except Exception as e:
                logger.error(f"Failed to ensure anonymous user: {e}")

        result_payload = await _mco_run_impl(
            query=query,
            mode=mode,
            sub_mode=sub_mode,
            chat_id=chat_id,
            session_id=session_id,
            force_retrieval=force_retrieval,
            selected_model=selected_model,
            image_b64=image_b64,
            image_mime=image_mime,
            rounds=rounds,
            response_style=response_style,
            preferences=preferences,
            db=db,
            user=safe_user,
        )

        if BehavioralMemoryManager and safe_user.get("user_id") != "00000000-0000-0000-0000-000000000000":
            background_tasks.add_task(
                BehavioralMemoryManager.update_profile_async,
                db=db,
                user_id=safe_user["user_id"],
                interaction_metrics={
                    "model": result_payload.get("data", {}).get("machine_metadata", {}).get("mode", mode),
                    "query_complexity": "complex" if len(query) > 100 else "simple",
                    "latency_ms": 0, # Could extract actual latency from omega_metadata if available
                }
            )

        return result_payload
    except HTTPException as http_exc:
        logger.error("/api/mco/run HTTPException: %s", http_exc.detail)
        return JSONResponse(status_code=http_exc.status_code, content={"detail": str(http_exc.detail)})
    except Exception as exc:
        print("MCO RUN ERROR:", repr(exc))
        import traceback
        tb = traceback.format_exc()
        print("TRACEBACK:", tb)
        logger.error("Unhandled error in /api/mco/run: %s", exc)
        logger.error(tb)
        
        exc_str = str(exc).lower() + " " + repr(exc).lower()
        error_type = "orchestration exception"
        
        if "timeout" in exc_str:
            error_type = "provider timeout"
        elif "api key" in exc_str:
            error_type = "missing API key"
        elif "auth" in exc_str and "provider" in exc_str:
            error_type = "provider auth failure"
        elif "resolve" in exc_str and "model" in exc_str:
            error_type = "resolver failure"
        elif "sqlalchemy" in exc_str or "asyncpg" in exc_str or "database" in exc_str or "password" in exc_str or "connection" in exc_str:
            error_type = "persistence exception"
            
        return JSONResponse(
            status_code=500, 
            content={
                "detail": error_type,
                "error": str(exc)
            }
        )


async def _mco_run_impl(
    query: str = Body(...),
    mode: str = Body("standard"),
    sub_mode: Optional[str] = Body(None),
    chat_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None),
    force_retrieval: bool = Body(False),
    selected_model: Optional[str] = Body(None),
    image_b64: Optional[str] = Body(None),
    image_mime: Optional[str] = Body(None),
    rounds: int = Body(3),
    response_style: Optional[str] = Body(None),
    preferences: Optional[Dict[str, Any]] = Body(None),
    db: AsyncSession = Depends(get_db),
    user: Optional[Dict] = None,
):
    """
    Main Meta-Cognitive Orchestrator execution endpoint.
    Routes through the mandatory 10-step protocol.

    Standard Mode: Returns highest-scoring aggregated answer.
    Experimental Mode: Returns all outputs with full scoring breakdown.
    Single Model Focus: Only the selected model executes.
    """
    orch = _get_orchestrator()

    # Load and apply user settings
    from api.endpoints_v2 import SETTINGS_SCHEMA
    from database.crud import get_user_preferences
    
    user_settings = {}
    if user and user.get("user_id"):
        db_settings = await get_user_preferences(db, user.get("user_id"))
        user_settings = {k: v.get("default") for k, v in SETTINGS_SCHEMA.items()}
        for k, v in db_settings.items():
            if k in user_settings:
                user_settings[k] = v

    # Apply defaults if parameters are missing or set to basic defaults
    if not selected_model and user_settings.get("default_model"):
        selected_model = user_settings.get("default_model")
        
    # The frontend usually sends "standard" if no mode is selected.
    if mode == "standard" and user_settings.get("default_mode") and user_settings.get("default_mode") != "standard":
        # Check if the default mode is a sub_mode (like debate, pro, evidence)
        default_m = user_settings.get("default_mode")
        if default_m in ["debate", "evidence", "glass", "synthesis", "pro"]:
            mode = "experimental"
            sub_mode = default_m
        else:
            mode = default_m
            
    if rounds == 3 and user_settings.get("debate_rounds"):
        rounds = user_settings.get("debate_rounds")
        
    if not response_style and user_settings.get("response_style"):
        response_style = user_settings.get("response_style")

    # Validate selected_model if provided
    if selected_model:
        resolved_model = cg.resolve_model_key(selected_model)
        if not resolved_model:
            # Fallback to system default if setting is invalid
            selected_model = None
        else:
            selected_model = resolved_model
            spec = COGNITIVE_MODEL_REGISTRY.get(selected_model)
            if not spec or not spec.enabled:
                selected_model = None

    # Resolve operating mode
    try:
        op_mode = OperatingMode(mode)
    except ValueError:
        op_mode = OperatingMode.STANDARD

    # Resolve chat
    chat = None
    if chat_id:
        try:
            chat = await get_chat(db, UUID(chat_id), user_id=user.get("user_id"))
            if chat:
                db.expunge(chat)
        except (ValueError, Exception):
            pass

    if not chat:
        chat_name = generate_chat_name(query, f"mco-{mode}")
        chat = await create_chat(
            db, chat_name, f"mco-{mode}",
            user_id=user["user_id"],
        )
        db.expunge(chat)

    orch_run, orch_bus = _start_runtime_run(
        str(chat.id),
        user.get("user_id", "00000000-0000-0000-0000-000000000000"),
        query,
    )

    document_cognition: Dict[str, Any] = {"available": False}
    if image_b64:
        try:
            _transition_runtime(
                orch_run,
                orch_bus,
                CognitivePhase.VERIFY,
                {"source": "document_cognition", "mime": image_mime},
            )
            document_cognition = await build_document_cognition(
                image_b64,
                image_mime,
                max_context_chars=5000,
            )
            if document_cognition.get("available"):
                orch_run.record_memory_retrieval(
                    "working",
                    "document_cognition",
                    document_cognition.get("semantic_context", ""),
                    0.88,
                ) if orch_run else None
                if orch_bus:
                    orch_bus.publish({
                        "event_type": "document_cognition_extracted",
                        "phase": "verify",
                        "document_type": document_cognition.get("document_type"),
                        "extraction_method": document_cognition.get("extraction_method"),
                        "text_char_count": document_cognition.get("text_char_count", 0),
                    })
        except Exception as doc_err:
            logger.warning("Document cognition extraction skipped: %s", doc_err)
            document_cognition = {
                "available": False,
                "error": str(doc_err)[:200],
            }

    await add_message(
        db,
        chat.id,
        user.get("user_id", "00000000-0000-0000-0000-000000000000"),
        "user",
        query,
        image_b64=image_b64,
        image_mime=image_mime,
    )

    # Build user-aware context before any model execution
    contextual_query = query
    context_meta: Dict[str, Any] = {
        "document_cognition": {
            key: value
            for key, value in document_cognition.items()
            if key != "semantic_context"
        }
    } if document_cognition.get("available") else {}
    try:
        recent_messages = await get_chat_messages(db, chat.id, user_id=user.get("user_id"))
        recent_payload = [
            {"role": m.role, "content": m.content}
            for m in (recent_messages[-12:] if recent_messages else [])
            if m and m.content
        ]

        builder = get_context_builder(max_tokens=2048, model=(selected_model or "llama33-70b"))
        built = await builder.build_context(
            db=db,
            user_id=user.get("user_id", "00000000-0000-0000-0000-000000000000"),
            query=query,
            recent_messages=recent_payload,
            semantic_search_results=None,
        )
        built_context = (built or {}).get("context", "")
        if built_context and isinstance(built_context, str) and built_context.strip():
            contextual_query = f"{built_context}\n\n[CURRENT USER QUERY]\n{query}"
            context_meta = {
                "context_applied": True,
                "token_usage": built.get("token_usage", {}),
                "model": built.get("model"),
                "available_tokens": built.get("available_tokens"),
            }
            _record_runtime_memory(
                orch_run,
                orch_bus,
                "episodic",
                "context_builder",
                built_context,
                0.92,
            )
        else:
            context_meta = {"context_applied": False}
    except Exception as ctx_err:
        logger.warning(f"Context build skipped: {ctx_err}")
        context_meta = {"context_applied": False, "error": str(ctx_err)[:200]}

    runtime_preferences = dict(preferences or {})
    if response_style:
        runtime_preferences["response_style"] = response_style
    if runtime_preferences:
        preference_lines = []
        style = runtime_preferences.get("response_style")
        if style:
            preference_lines.append(f"Response style: {style}.")
        default_mode = runtime_preferences.get("default_mode")
        if default_mode:
            preference_lines.append(f"Default mode preference: {default_mode}.")
        if preference_lines:
            contextual_query = (
                "[USER RUNTIME PREFERENCES]\n"
                + "\n".join(preference_lines)
                + "\n\n"
                + contextual_query
            )
            context_meta["runtime_preferences"] = runtime_preferences

    if document_cognition.get("semantic_context"):
        contextual_query = (
            f"[DOCUMENT COGNITION]\n{document_cognition['semantic_context']}"
            f"\n\n{contextual_query}"
        )
        context_meta["document_cognition"] = {
            key: value
            for key, value in document_cognition.items()
            if key != "semantic_context"
        }

    # ══════════════════════════════════════════════════════════
    # QUERY COMPLEXITY CHECK — Skip debate for trivial queries
    # ══════════════════════════════════════════════════════════
    from core.query_router import classify_query_complexity
    query_complexity = classify_query_complexity(query)

    # ══════════════════════════════════════════════════════════
    # DEBATE MODE DELEGATION
    # When sub_mode is "debate" and CognitiveOrchestrator is available,
    # delegate to it for multi-round structured debate (StructuredDebateEngine).
    # This produces real rounds with rebuttals, drift/rift metrics, etc.
    # Skip debate for trivial queries even if debate mode is requested.
    # ══════════════════════════════════════════════════════════
    effective_sub_mode = sub_mode
    if effective_sub_mode in ("debate", "pro") and query_complexity == "trivial":
        logger.info(f"Trivial query in debate mode — skipping debate for chat {chat.id}")
        effective_sub_mode = None  # Fall through to standard MCO pipeline

    # If user selected a specific model, never route to debate engine
    if selected_model and effective_sub_mode in ("debate", "pro"):
        logger.info(f"Single model '{selected_model}' selected — skipping debate")
        effective_sub_mode = None

    predicted_execution_path = (
        "ensemble" if effective_sub_mode in ("debate", "pro") else
        "fast_standard" if query_complexity == "trivial" and not selected_model and not image_b64 else
        "single_model" if selected_model else
        "standard_mco"
    )
    _record_runtime_routing(
        orch_run,
        orch_bus,
        execution_path=predicted_execution_path,
        reason="debate_sub_mode" if effective_sub_mode in ("debate", "pro") else "mco_standard_pipeline",
        query_complexity=query_complexity,
        selected_model=selected_model,
        debate_requested=effective_sub_mode in ("debate", "pro"),
    )

    skip_mco = False
    
    if effective_sub_mode in ("debate", "pro") and _cognitive_engine is not None:
        logger.info(f"Debate mode: delegating to CognitiveOrchestrator for chat {chat.id}")
        _transition_runtime(orch_run, orch_bus, CognitivePhase.SPAWN_AGENTS, {"path": "ensemble"})
        try:
            from core.ensemble_schemas import EnsembleFailure
            debate_rounds = max(1, min(int(rounds or 3), 10))
            ensemble_response = await _cognitive_engine.process(
                query=contextual_query,
                chat_id=str(chat.id),
                rounds=debate_rounds,
                image_b64=image_b64,
                image_mime=image_mime,
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
                "context_builder": context_meta,
            })

            # Persist
            await add_message(
                db,
                chat.id,
                user.get("user_id", "00000000-0000-0000-0000-000000000000"),
                "assistant",
                formatted_output,
                reasoning_json=omega_metadata,
            )
            await update_chat_metadata(
                db, chat.id,
                priority_answer=formatted_output,
                machine_metadata=omega_metadata,
                rounds=ensemble_response.debate_result.total_rounds,
            )

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

            if orch_run is not None:
                try:
                    orch_run.models_executed = ensemble_response.models_executed
                    orch_run.models_succeeded = ensemble_response.models_succeeded
                    orch_run.models_failed = ensemble_response.models_failed
                    orch_run.active_agents = [
                        getattr(output, "model_name", getattr(output, "model_id", "unknown"))
                        for output in (ensemble_response.model_outputs or [])
                    ]
                    for output in (ensemble_response.model_outputs or []):
                        orch_run.record_provider_call(
                            model_id=getattr(output, "model_id", getattr(output, "model_name", "unknown")),
                            model_name=getattr(output, "model_name", "unknown"),
                            provider=getattr(output, "provider", "unknown"),
                            latency_ms=float(getattr(output, "latency_ms", 0.0) or 0.0),
                            succeeded=bool(getattr(output, "succeeded", False)),
                            error=getattr(output, "error", None),
                            input_tokens=int(getattr(output, "input_tokens", 0) or 0),
                            output_tokens=int(getattr(output, "output_tokens", 0) or 0),
                        )
                    orch_run.record_confidence_snapshot(
                        phase="post_debate",
                        value=confidence,
                        method="calibrated_ensemble",
                    )
                    orch_run.record_debate_round(
                        round_number=getattr(ensemble_response.debate_result, "total_rounds", 0),
                        positions=[
                            {
                                "model_id": getattr(output, "model_id", getattr(output, "model_name", "unknown")),
                                "model_name": getattr(output, "model_name", "unknown"),
                                "confidence": round(float(getattr(output, "confidence", 0.0) or 0.0), 4),
                            }
                            for output in (ensemble_response.model_outputs or [])
                        ],
                        contradiction_density=float(getattr(ensemble_response.ensemble_metrics, "contradiction_density", 0.0) or 0.0),
                        drift_index=float(getattr(ensemble_response.debate_result, "drift_index", 0.0) or 0.0),
                    )
                    if getattr(ensemble_response.ensemble_metrics, "contradiction_density", 0.0) > 0.0:
                        _transition_runtime(
                            orch_run,
                            orch_bus,
                            CognitivePhase.VERIFY,
                            {"contradiction_density": round(float(ensemble_response.ensemble_metrics.contradiction_density), 4)},
                        )
                    _transition_runtime(orch_run, orch_bus, CognitivePhase.SYNTHESIZE, {"method": "ensemble_weighted"})
                    orch_run.record_synthesis_start("ensemble_weighted", ensemble_response.models_succeeded)
                    orch_run.record_synthesis_complete(len(formatted_output or ""))
                    runtime_artifact = build_cognitive_artifact(
                        ensemble_response=ensemble_response,
                        orchestration_run=orch_run,
                    ) if build_cognitive_artifact is not None else None
                    if runtime_artifact:
                        omega_metadata["cognitive_artifact"] = runtime_artifact
                    omega_metadata["orchestration_run"] = orch_run.to_frontend_dict()
                    _complete_runtime_run(
                        orch_run,
                        orch_bus,
                        reflection=(runtime_artifact or {}).get("reflective_cognition", ""),
                    )
                except Exception as runtime_err:
                    logger.debug("[MCO Runtime] Debate runtime enrichment failed: %s", runtime_err)

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

            debate_result = {
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

            if effective_sub_mode == "debate":
                debate_result = _attach_runtime_metadata(
                    debate_result,
                    orch_run,
                    omega_metadata.get("cognitive_artifact"),
                )
                # Sanitize before returning
                debate_result = _sanitize_mco_response(debate_result)
                return debate_result
                
            elif effective_sub_mode == "pro":
                # Adapt EnsembleResponse into OrchestratorResponse for chaining
                class DummyOutput:
                    def __init__(self, raw, name):
                        self.raw_output = raw
                        self.model_name = name
                        self.success = True
                        self.error = None
                        self.latency_ms = 0
                        
                class DummyScore:
                    def __init__(self, score):
                        self.final_score = score
                        self.topic_alignment = score
                        self.knowledge_grounding = score
                        self.specificity = score
                        self.confidence_calibration = score
                        self.drift_penalty = 0

                class DummyResult:
                    def __init__(self, out, sc):
                        self.output = out
                        self.score = sc

                class DummyBreakdown:
                    def __init__(self, name, score):
                        self.model_name = name
                        self.final_score = score
                        self.topic_alignment = score
                        self.knowledge_grounding = score
                        self.specificity = score
                        self.confidence_calibration = score
                        self.drift_penalty = 0
                        self.score = score

                adapted_results = []
                adapted_scoring = []
                for m in (ensemble_response.model_outputs or []):
                    name = getattr(m, "model_name", "unknown")
                    raw = getattr(m, "raw_output", "") or getattr(m, "position", "")
                    conf = getattr(m, "confidence", 0.8)
                    adapted_results.append(DummyResult(DummyOutput(raw, name), DummyScore(conf)))
                    adapted_scoring.append(DummyBreakdown(name, conf))
                
                class AdaptedResponse:
                    def __init__(self):
                        self.all_results = adapted_results
                        self.scoring_breakdown = adapted_scoring
                        self.divergence_metrics = {"max_divergence": ens_entropy, "convergence": "high" if ens_entropy < 0.3 else "low"}
                        self.aggregated_answer = formatted_output
                        self.winning_model = getattr(ensemble_response.debate_result, "winning_model", "ensemble")
                        self.drift_score = getattr(ensemble_response.debate_result, "drift_index", 0.0)
                        self.volatility_score = ens_fragility
                        self.mode = OperatingMode.EXPERIMENTAL
                        self.sub_mode = "pro"
                        self.session_id = str(chat.id)
                        self.refinement_cycles = getattr(ensemble_response.debate_result, "total_rounds", 1)
                        self.winning_score = confidence
                        self.latency_ms = 0
                
                # Expose adapted response to the rest of the pipeline
                response = AdaptedResponse()
                skip_mco = True
                pro_omega_metadata = omega_metadata

    # ══════════════════════════════════════════════════════════
    # FAST-PATH: Trivial queries bypass the full 10-step protocol
    # Uses a single fast model for instant response (e.g., "hi", "hello")
    # ══════════════════════════════════════════════════════════
    if query_complexity == "trivial" and not selected_model and not image_b64:
        import time as _time
        _fast_start = _time.monotonic()
        logger.info(f"Fast-path: trivial query '{query[:30]}' — single model, no MCO protocol")

        # Pick the fastest available model (prefer Groq Llama 8B)
        _FAST_MODEL_PRIORITY = ["llama31-8b", "llama4-scout", "mixtral-8x7b", "llama33-70b"]
        fast_model = None
        for mk in _FAST_MODEL_PRIORITY:
            spec = COGNITIVE_MODEL_REGISTRY.get(mk)
            if spec and spec.active and spec.enabled:
                fast_model = mk
                break

        if fast_model:
            from metacognitive.schemas import CognitiveGatewayInput, QueryMode
            gateway_input = CognitiveGatewayInput(
                user_query=contextual_query,
                mode=QueryMode.RAW,
            )
            try:
                output = await orch.cognitive_gateway.invoke_model(fast_model, gateway_input)
                if output.success and output.raw_output.strip():
                    answer = output.raw_output.strip()
                    _elapsed = (_time.monotonic() - _fast_start) * 1000
                    omega_metadata = {
                        "version": "8.0.0-cognitive-runtime",
                        "winning_model": fast_model,
                        "winning_score": 0.95,
                        "latency_ms": round(_elapsed, 1),
                        "fast_path": True,
                        "context_builder": context_meta,
                    }
                    if orch_run is not None:
                        try:
                            orch_run.active_agents = [fast_model]
                            orch_run.models_executed = 1
                            orch_run.models_succeeded = 1
                            orch_run.record_provider_call(
                                model_id=fast_model,
                                model_name=output.model_name,
                                provider="mco_fast_path",
                                latency_ms=_elapsed,
                                succeeded=True,
                                input_tokens=int(getattr(output, "input_tokens", 0) or 0),
                                output_tokens=int(getattr(output, "output_tokens", 0) or 0),
                            )
                            orch_run.record_confidence_snapshot("fast_standard", 0.95, method="fast_path")
                            _transition_runtime(orch_run, orch_bus, CognitivePhase.SYNTHESIZE, {"method": "fast_standard"})
                            orch_run.record_synthesis_start("fast_standard", 1)
                            orch_run.record_synthesis_complete(len(answer))
                            _complete_runtime_run(
                                orch_run,
                                orch_bus,
                                reflection="Fast-path routing resolved a trivial query without debate escalation.",
                            )
                            omega_metadata["orchestration_run"] = orch_run.to_frontend_dict()
                        except Exception as runtime_err:
                            logger.debug("[MCO Runtime] Fast-path runtime enrichment failed: %s", runtime_err)
                    await add_message(
                        db,
                        chat.id,
                        user.get("user_id", "00000000-0000-0000-0000-000000000000"),
                        "assistant",
                        answer,
                        reasoning_json=omega_metadata,
                    )
                    logger.info(f"Fast-path complete in {_elapsed:.0f}ms via {fast_model}")
                    fast_result = {
                        "chat_id": str(chat.id),
                        "session_id": str(chat.id),
                        "mode": mode,
                        "sub_mode": sub_mode,
                        "formatted_output": answer,
                        "aggregated_answer": answer,
                        "confidence": 0.95,
                        "data": {"priority_answer": answer},
                        "omega_metadata": omega_metadata,
                    }
                    # Sanitize before returning
                    fast_result = _sanitize_mco_response(fast_result)
                    return fast_result
            except Exception as fast_err:
                logger.warning(f"Fast-path failed ({fast_err}), falling through to MCO")

    # ══════════════════════════════════════════════════════════
    # STANDARD MCO PIPELINE (non-debate modes)
    # ══════════════════════════════════════════════════════════

    # ── Cache check ──────────────────────────────────────────
    from core.cache_engine import reasoning_cache
    cached_result = await reasoning_cache.get_query(query, mode, sub_mode or "")
    if cached_result and not force_retrieval:
        logger.info(f"Returning cached result for [{mode}/{sub_mode}]")
        cached_result = dict(cached_result)
        cached_result["chat_id"] = str(chat.id)
        cached_result["session_id"] = str(chat.id)
        _record_runtime_routing(
            orch_run,
            orch_bus,
            execution_path="cache_hit",
            reason="reasoning_cache",
            query_complexity=query_complexity,
            selected_model=selected_model,
            debate_requested=bool(sub_mode == "debate"),
            cache_hit=True,
        )
        _transition_runtime(orch_run, orch_bus, CognitivePhase.SYNTHESIZE, {"cache_hit": True})
        if orch_run is not None:
            cached_confidence = float(
                cached_result.get("confidence")
                or (cached_result.get("omega_metadata", {}) or {}).get("confidence")
                or 0.5
            )
            orch_run.record_confidence_snapshot("cache_hit", cached_confidence, method="cache_reuse")
            orch_run.record_synthesis_start("cache_reuse", 0)
            cached_output = cached_result.get("formatted_output") or cached_result.get("aggregated_answer") or ""
            orch_run.record_synthesis_complete(len(cached_output))
            _complete_runtime_run(
                orch_run,
                orch_bus,
                reflection="Cached cognitive result reused for this request.",
            )
            _attach_runtime_metadata(cached_result, orch_run)
        return cached_result

    # Build request
    request = OrchestratorRequest(
        session_id=session_id or str(chat.id),
        query=contextual_query,
        mode=op_mode,
        sub_mode=sub_mode,
        chat_id=str(chat.id),
        force_retrieval=force_retrieval,
        selected_model=selected_model,
        query_complexity=query_complexity,
        image_b64=image_b64,
        image_mime=image_mime,
    )

    # Execute 10-step protocol
    if not skip_mco:
        try:
            _transition_runtime(orch_run, orch_bus, CognitivePhase.SPAWN_AGENTS, {"path": predicted_execution_path})
            response = await orch.process(request)
        except RuntimeError as e:
            logger.error(f"MCO execution error: {e}")
            _fail_runtime_run(orch_run, orch_bus, str(e), "MCO_RUNTIME_ERROR")
            raise HTTPException(status_code=502, detail="Model processing failed. Please try again.")
        except Exception as e:
            logger.error(f"MCO execution failed: {e}", exc_info=True)
            _fail_runtime_run(orch_run, orch_bus, str(e), "MCO_EXECUTION_ERROR")
            raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")

    # Guard: no blank responses
    if not response.aggregated_answer or not response.aggregated_answer.strip():
        logger.error(f"Empty output from model '{response.winning_model}'")
        _fail_runtime_run(orch_run, orch_bus, "empty_aggregated_answer", "MCO_EMPTY_OUTPUT")
        raise HTTPException(
            status_code=502,
            detail="No response generated. Please try again.",
        )

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

    if skip_mco:
        omega_metadata = pro_omega_metadata
    else:
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
        "context_builder": context_meta,
        "runtime_preferences": runtime_preferences,
    }

    if orch_run is not None:
        try:
            orch_run.models_executed = len(response.all_results or [])
            orch_run.models_succeeded = sum(1 for result_item in (response.all_results or []) if getattr(result_item.output, "success", False))
            orch_run.models_failed = max(0, orch_run.models_executed - orch_run.models_succeeded)
            orch_run.active_agents = [result_item.output.model_name for result_item in (response.all_results or [])]
            for result_item in (response.all_results or []):
                orch_run.record_provider_call(
                    model_id=result_item.output.model_name,
                    model_name=result_item.output.model_name,
                    provider="mco",
                    latency_ms=float(getattr(result_item.output, "latency_ms", 0.0) or 0.0),
                    succeeded=bool(getattr(result_item.output, "success", False)),
                    error=getattr(result_item.output, "error", None),
                    input_tokens=int(getattr(result_item.output, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(result_item.output, "output_tokens", 0) or 0),
                )
            orch_run.record_confidence_snapshot(
                phase="mco_arbitration",
                value=float(response.winning_score or 0.0),
                method="mco_arbitration",
            )
            if divergence.get("max_divergence"):
                _transition_runtime(
                    orch_run,
                    orch_bus,
                    CognitivePhase.VERIFY,
                    {"max_divergence": round(float(divergence.get("max_divergence") or 0.0), 4)},
                )
            _transition_runtime(orch_run, orch_bus, CognitivePhase.SYNTHESIZE, {"method": "mco_arbitration"})
            orch_run.record_synthesis_start("mco_arbitration", orch_run.models_succeeded)
            orch_run.record_synthesis_complete(len(response.aggregated_answer or ""))
        except Exception as runtime_err:
            logger.debug("[MCO Runtime] Standard runtime enrichment failed: %s", runtime_err)

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

    # Build visualizations if possible
    omega_metadata["visualizations"] = {}
    try:
        from viz.battle_visualization import BattleVisualizationEngine
        from core.ensemble_schemas import StructuredModelOutput
        viz_engine = BattleVisualizationEngine()
        
        # We need StructuredModelOutputs for the viz engine
        structured_outputs = []
        for r in response.all_results:
            if r.output.success and r.output.raw_output:
                structured_outputs.append(StructuredModelOutput(
                    model_id=r.output.model_name,
                    position=r.output.raw_output[:500],
                    reasoning=r.output.raw_output,
                ))
        
        if structured_outputs:
            sim_matrix, model_labels = viz_engine._build_similarity_matrix(structured_outputs)
            
            if sim_matrix and model_labels:
                # 1. Heatmap
                from viz.conflict_visualizer import ConflictVisualizer
                cv = ConflictVisualizer()
                heatmap_b64 = cv.plot_similarity_heatmap(model_labels, sim_matrix)
                if heatmap_b64:
                    omega_metadata["visualizations"]["heatmap_png"] = heatmap_b64
                
                # 2. Conflict Graph
                from viz.conflict_graph import build_conflict_edges, plot_conflict_graph
                similarities = {}
                n = len(structured_outputs)
                for i in range(n):
                    for j in range(i + 1, n):
                        key = (structured_outputs[i].model_id, structured_outputs[j].model_id)
                        similarities[key] = sim_matrix[i][j]
                
                conflict_png = plot_conflict_graph(
                    models=model_labels,
                    similarities=similarities,
                    threshold=0.6,
                    display=False,
                )
                if conflict_png:
                    omega_metadata["visualizations"]["conflict_graph_png"] = conflict_png

                # Also save the raw sim matrix for the frontend to render interactively
                omega_metadata["visualizations"]["similarity_matrix"] = sim_matrix
                omega_metadata["visualizations"]["model_labels"] = model_labels

    except Exception as viz_err:
        logger.warning(f"Failed to generate visualizations for chat {chat.id}: {viz_err}")

    # Build sub-mode-specific structured results for frontend components
    effective_sub_mode = response.sub_mode or sub_mode

    if response.mode == OperatingMode.EXPERIMENTAL or effective_sub_mode:
        # Build debate_result (DebateView/DebateArena consumes this)
        # PHASE 6: Filter out empty/failed model outputs from debate positions
        _valid_results = [
            r for r in response.all_results
            if r.output.success and r.output.raw_output and r.output.raw_output.strip()
        ]
        
        # Only override debate_result if it wasn't already generated by Pro Mode's CognitiveCoreEngine
        if "debate_result" not in omega_metadata:
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

        # Build forensic_result (EvidenceView consumes this)
        if effective_sub_mode in ("evidence", "pro"):
            try:
                from core.evidence_pipeline import build_evidence_result
                omega_metadata["forensic_result"] = await build_evidence_result(
                    query=query,
                    all_results=response.all_results,
                    scoring_breakdown=response.scoring_breakdown,
                    aggregated_answer=response.aggregated_answer,
                    winning_model=response.winning_model,
                )
                omega_metadata["evidence_result"] = omega_metadata["forensic_result"]
            except Exception as ev_err:
                logger.error(f"Evidence pipeline failed, using fallback: {ev_err}")
                omega_metadata["forensic_result"] = {
                    "models_analyzed": len(response.all_results),
                    "scoring_breakdown": scoring_serialized,
                    "divergence": divergence,
                    "winning_model": response.winning_model,
                    "winning_score": round(response.winning_score, 4),
                }
                omega_metadata["evidence_result"] = omega_metadata["forensic_result"]
        else:
            omega_metadata["forensic_result"] = {
                "models_analyzed": len(response.all_results),
                "scoring_breakdown": scoring_serialized,
                "divergence": divergence,
                "winning_model": response.winning_model,
                "winning_score": round(response.winning_score, 4),
            }

        # Build audit_result (GlassView consumes this)
        if effective_sub_mode in ("glass", "pro"):
            try:
                from core.glass_pipeline import build_glass_result
                omega_metadata["audit_result"] = build_glass_result(
                    all_results=response.all_results,
                    scoring_breakdown=response.scoring_breakdown,
                    divergence_metrics=divergence,
                    aggregated_answer=response.aggregated_answer,
                    winning_model=response.winning_model,
                    drift_score=response.drift_score,
                    volatility_score=response.volatility_score,
                )
                omega_metadata["glass_result"] = omega_metadata["audit_result"]
            except Exception as gl_err:
                logger.error(f"Glass pipeline failed, using fallback: {gl_err}")
                omega_metadata["audit_result"] = {
                    "all_outputs": all_outputs_serialized,
                    "scoring_breakdown": scoring_serialized,
                    "divergence_metrics": divergence,
                    "drift_score": round(response.drift_score, 4),
                    "volatility_score": round(response.volatility_score, 4),
                    "refinement_cycles": response.refinement_cycles,
                }
                omega_metadata["glass_result"] = omega_metadata["audit_result"]
        else:
            omega_metadata["audit_result"] = {
                "all_outputs": all_outputs_serialized,
                "scoring_breakdown": scoring_serialized,
                "divergence_metrics": divergence,
                "drift_score": round(response.drift_score, 4),
                "volatility_score": round(response.volatility_score, 4),
                "refinement_cycles": response.refinement_cycles,
            }

        # Build synthesis_result (SynthesisView consumes this)
        if effective_sub_mode in ("synthesis", "pro"):
            try:
                from core.synthesis_engine import build_synthesis_result
                synthesis_result = build_synthesis_result(
                    all_results=response.all_results,
                    scoring_breakdown=response.scoring_breakdown,
                    divergence_metrics=divergence,
                    aggregated_answer=response.aggregated_answer,
                    winning_model=response.winning_model,
                )

                # Call Claude for refined synthesis if toggled ON
                claude_spec = COGNITIVE_MODEL_REGISTRY.get("claude-sonnet-4.6")
                if claude_spec and claude_spec.active and claude_spec.enabled:
                    try:
                        from models.mco_bridge import MCOModelBridge
                        from metacognitive.cognitive_gateway import get_claude_usage
                        bridge = MCOModelBridge()
                        perspectives = []
                        for r in (response.all_results or [])[:6]:
                            if r.output.success and r.output.raw_output:
                                perspectives.append(
                                    f"[{r.output.model_name}]: {r.output.raw_output[:800]}"
                                )
                        if perspectives:
                            synthesis_prompt = (
                                "Multiple AI models analyzed a query. Synthesize their perspectives "
                                "into a clear, comprehensive final answer. Integrate strengths, "
                                "resolve contradictions, and structure the response clearly.\n\n"
                                + "\n\n".join(perspectives)
                            )
                            claude_output = await bridge.call_model(
                                "claude-sonnet-4.6", synthesis_prompt,
                                "You are a synthesis expert combining multiple AI perspectives.",
                                max_tokens=500,
                            )
                            if claude_output and not claude_output.startswith("Error:"):
                                synthesis_result["claude_synthesis"] = claude_output
                                synthesis_result["claude_active"] = True
                                synthesis_result["refined_output"] = claude_output
                                synthesis_result["claude_usage"] = get_claude_usage()
                                logger.info("Claude synthesis completed successfully")
                            else:
                                synthesis_result["claude_active"] = False
                                logger.warning(f"Claude synthesis failed: {claude_output}")
                        else:
                            synthesis_result["claude_active"] = False
                    except Exception as claude_err:
                        logger.warning(f"Claude synthesis skipped: {claude_err}")
                        synthesis_result["claude_active"] = False
                else:
                    synthesis_result["claude_active"] = False

                omega_metadata["synthesis_result"] = synthesis_result
                
                # In Pro Mode, the Unified Response is the output of the synthesis engine
                if effective_sub_mode == "pro":
                    unified_response = synthesis_result.get("refined_output") or synthesis_result.get("final_answer") or response.aggregated_answer
                    response.aggregated_answer = unified_response

            except Exception as syn_err:
                logger.error(f"Synthesis pipeline failed: {syn_err}")
                omega_metadata["synthesis_result"] = None

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

    # Persist assistant response
    await add_message(
        db,
        chat.id,
        user.get("user_id", "00000000-0000-0000-0000-000000000000"),
        "assistant",
        response.aggregated_answer,
        reasoning_json=omega_metadata,
    )
    await update_chat_metadata(
        db, chat.id,
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
            "sub_mode": effective_sub_mode,
            "runtime_preferences": runtime_preferences,
            **omega_metadata,  # Embed all runtime artifacts for history hydration
        }
    )

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

    runtime_artifact = _build_mco_cognitive_artifact(response, orch_run) if orch_run is not None else None
    if runtime_artifact:
        omega_metadata["cognitive_artifact"] = runtime_artifact
    result = _attach_runtime_metadata(result, orch_run, runtime_artifact)
    if orch_run is not None:
        if get_deliberative_memory is not None and divergence.get("max_divergence", 0.0):
            try:
                topic_hash = str(abs(hash(query.lower().strip())) % 10_000_000)
                get_deliberative_memory().record(
                    topic_hash=topic_hash,
                    contradiction_density=float(divergence.get("max_divergence", 0.0) or 0.0),
                    consensus_reached=float(divergence.get("max_divergence", 0.0) or 0.0) < 0.35,
                    drift_index=float(response.drift_score or 0.0),
                    key_conflicts=[item.get("position", "")[:120] for item in runtime_artifact.get("alternative_perspectives", [])[:3]],
                    chat_id=str(chat.id),
                )
            except Exception as memory_err:
                logger.debug("[MCO Runtime] Deliberative memory update failed: %s", memory_err)
        if get_tactical_memory is not None:
            try:
                get_tactical_memory().record(
                    query_complexity=query_complexity,
                    execution_path=predicted_execution_path,
                    model_count=len(response.all_results or []),
                    latency_ms=float(response.latency_ms or 0.0),
                    confidence=float(response.winning_score or 0.0),
                    success=True,
                )
            except Exception as memory_err:
                logger.debug("[MCO Runtime] Tactical memory update failed: %s", memory_err)
        _complete_runtime_run(
            orch_run,
            orch_bus,
            reflection=(runtime_artifact or {}).get("reflective_cognition", ""),
        )

    # ── Cache write ──────────────────────────────────────────
    try:
        await reasoning_cache.set_query(query, mode, result, sub_mode or "")
    except Exception:
        pass

    # Sanitize response before returning
    result = _sanitize_mco_response(result)
    
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
    return await _mco_run_impl(
        query=query,
        mode="experimental",
        sub_mode=sub_mode,
        chat_id=chat_id,
        session_id=session_id,
        force_retrieval=force_retrieval,
        selected_model=None,
        image_b64=None,
        image_mime=None,
        rounds=3,
        response_style=None,
        preferences=None,
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
async def mco_models(user: Optional[Dict] = Depends(get_optional_user)):
    """List available cognitive models safely (never crash)."""
    logger.info("Entered /api/mco/models (authenticated=%s)", bool(user))

    try:
        registry = COGNITIVE_MODEL_REGISTRY
        if not registry or not hasattr(registry, "items"):
            logger.warning(
                "COGNITIVE_MODEL_REGISTRY unavailable or invalid type: %s",
                type(registry).__name__,
            )
            return {"models": []}

        keys = list(registry.keys())
        logger.info("Model registry keys (%d): %s", len(keys), keys)

        models = []
        for key, spec in registry.items():
            logger.debug("Processing model key=%s", key)

            if spec is None:
                logger.warning("Skipping model key=%s because spec is None", key)
                continue

            try:
                role_obj = getattr(spec, "role", None)
                role_value = getattr(role_obj, "value", str(role_obj) if role_obj is not None else "unknown")

                context_window = getattr(spec, "context_window", None)
                if context_window is not None and not isinstance(context_window, (str, int, float, bool)):
                    context_window = str(context_window)

                max_output_tokens = getattr(spec, "max_output_tokens", None)
                if max_output_tokens is not None and not isinstance(max_output_tokens, (str, int, float, bool)):
                    max_output_tokens = str(max_output_tokens)

                disable_reason = getattr(spec, "disable_reason", None)
                if disable_reason is not None and not isinstance(disable_reason, (str, int, float, bool)):
                    disable_reason = str(disable_reason)

                models.append({
                    "key": str(key),
                    "name": str(getattr(spec, "name", key)),
                    "model_id": str(getattr(spec, "model_id", key)),
                    "provider": str(getattr(spec, "provider", "unknown")),
                    "role": str(role_value),
                    "context_window": context_window,
                    "max_output_tokens": max_output_tokens,
                    "active": bool(getattr(spec, "active", False)),
                    "enabled": bool(getattr(spec, "enabled", False)),
                    "disable_reason": disable_reason,
                })
            except Exception:
                logger.exception("Failed processing model key=%s", key)
                continue

        return {"models": models}
    except Exception:
        logger.exception("Unhandled failure in /api/mco/models")
        return {"models": []}

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
