"""
============================================================
Sentinel-E v5.0 — Production API Gateway
============================================================
Layered Architecture:
  L1: API Gateway (this file) — routing, auth, middleware
  L2: Orchestrator           — request coordination
  L3: Model Interface        — provider abstraction
  L4: Cognitive Engine       — Omega kernel
  L5: Memory Engine          — 3-tier memory
  L6: Retrieval Engine       — cognitive RAG
  L7: Presentation Layer     — response formatting

Security:
  - JWT authentication
  - CSP headers
  - Rate limiting
  - Input validation
  - Prompt firewall
  - Centralized error handling
  - No credentials in logs
  - Strict CORS
"""

import sys
import os
import json
import logging
import uuid as uuid_lib
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Gateway Layer Imports ────────────────────────────────────
from gateway.config import get_settings
from gateway.auth import (
    create_access_token, create_refresh_token,
    get_current_user, get_optional_user, decode_token,
)
from gateway.middleware import (
    RateLimitMiddleware, SecurityHeadersMiddleware,
    RequestTrackingMiddleware, ErrorHandlerMiddleware,
    InputValidationMiddleware,
)
from gateway.prompt_firewall import get_firewall

# ── Database ─────────────────────────────────────────────────
from database.connection import get_db, init_db, check_redis, redis_client
from database.crud import (
    create_chat, get_chat, list_chats, update_chat_metadata,
    add_message, get_chat_messages,
)

# ── Core Engine Imports ──────────────────────────────────────
from sentinel.sentinel_sigma_v4 import SentinelSigmaOrchestratorV4
from sentinel.schemas import SentinelRequest
from core.omega_kernel import OmegaCognitiveKernel
from core.mode_config import ModeConfig
from core.knowledge_learner import KnowledgeLearner
from utils.chat_naming import generate_chat_name

# ── New Architecture Layers ──────────────────────────────────
from memory.memory_engine import MemoryEngine
from retrieval.cognitive_rag import CognitiveRAG
from core.dynamic_analytics import DynamicAnalyticsEngine

# ── Cognitive Core Engine v7.0 ────────────────────────────────
from core.cognitive_orchestrator import CognitiveCoreEngine

# ── Optimization Layer ───────────────────────────────────────
from optimization import (
    get_token_optimizer,
    get_response_cache,
    get_fallback_router,
    get_cost_governor,
    get_observability_hub,
)

# ── Meta-Cognitive Orchestrator ──────────────────────────────
from metacognitive.orchestrator import MetaCognitiveOrchestrator
from metacognitive.background_daemon import BackgroundDaemon
from metacognitive.routes import router as mco_router, set_orchestrator as mco_set_orchestrator, set_daemon as mco_set_daemon

# ── Logging ──────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Sentinel-API")

# ── Environment Detection ───────────────────────────────────
ENV = os.getenv("ENV", "development").lower()

# ── Global State ─────────────────────────────────────────────
orchestrator: Optional[SentinelSigmaOrchestratorV4] = None
omega_kernel: Optional[OmegaCognitiveKernel] = None
knowledge_learner: Optional[KnowledgeLearner] = None
cognitive_rag: Optional[CognitiveRAG] = None
analytics_engine: Optional[DynamicAnalyticsEngine] = None
mco_orchestrator: Optional[MetaCognitiveOrchestrator] = None
mco_daemon: Optional[BackgroundDaemon] = None
mco_bridge = None  # MCOModelBridge — unified model client
cognitive_orchestrator_engine: Optional[CognitiveCoreEngine] = None  # Cognitive Engine v7.0
omega_sessions: Dict[str, OmegaCognitiveKernel] = {}
memory_sessions: Dict[str, MemoryEngine] = {}

# Maximum in-memory sessions to prevent memory leak
MAX_SESSIONS = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, omega_kernel, knowledge_learner, cognitive_rag, analytics_engine
    global mco_orchestrator, mco_daemon, mco_bridge
    logger.info("Initializing Sentinel-E v5.0 Production System...")

    # Initialize DB
    await init_db()
    await check_redis()

    # Initialize core components
    orchestrator = SentinelSigmaOrchestratorV4()
    knowledge_learner = KnowledgeLearner()

    # New architecture layers
    cognitive_rag = CognitiveRAG()
    analytics_engine = DynamicAnalyticsEngine()

    # Optimization layer (lightweight singletons)
    get_token_optimizer()
    get_response_cache()
    get_fallback_router()
    get_cost_governor()
    get_observability_hub()

    # ── Meta-Cognitive Orchestrator (MUST init before OmegaKernel) ─
    try:
        mco_orchestrator = MetaCognitiveOrchestrator()
        if redis_client:
            mco_orchestrator.set_redis(redis_client)
        mco_set_orchestrator(mco_orchestrator)

        # Create MCO bridge — unified model client that routes through gateway
        from models.mco_bridge import MCOModelBridge
        mco_bridge = MCOModelBridge(mco_orchestrator.cognitive_gateway)
        logger.info("MCO Model Bridge created — all model calls route through CognitiveModelGateway")

        # ── Cognitive Core Engine v7.0 ────────────────────────
        cognitive_orchestrator_engine = CognitiveCoreEngine(model_bridge=mco_bridge)
        logger.info("Cognitive Core Engine v7.0 initialized — ensemble-only, no mode routing")

        # Background daemon (starts paused — activate via API)
        mco_daemon = BackgroundDaemon(
            cognitive_gateway=mco_orchestrator.cognitive_gateway,
            knowledge_engine=mco_orchestrator.knowledge_engine,
            session_engine=mco_orchestrator.session_engine,
            interval=300,
        )
        mco_set_daemon(mco_daemon)
        logger.info("Meta-Cognitive Orchestrator initialized")
    except Exception as e:
        logger.warning(f"MCO init failed (non-fatal): {e}")

    # ── Omega Kernel (uses MCO bridge if available, else legacy client) ─
    omega_kernel = OmegaCognitiveKernel(
        sigma_orchestrator=orchestrator,
        knowledge_learner=knowledge_learner,
        cloud_client=mco_bridge,
    )
    omega_kernel.knowledge_learner = knowledge_learner

    logger.info("Sentinel-E v5.0 online. All systems initialized (with optimization layer + MCO).")
    yield
    # Cleanup MCO
    if mco_orchestrator:
        await mco_orchestrator.close()
    if mco_daemon and mco_daemon.is_running:
        mco_daemon.stop()
    logger.info("Shutting down Sentinel-E v5.0...")


app = FastAPI(
    title="Sentinel-E API",
    version="5.0.0",
    lifespan=lifespan,
    docs_url=None if ENV == "production" else "/docs",
    redoc_url=None if ENV == "production" else "/redoc",
    openapi_url=None if ENV == "production" else "/openapi.json",
)

# ── Middleware Stack (order matters: outermost first) ────────
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(InputValidationMiddleware)

# Strict CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)

# ── Meta-Cognitive Orchestrator Router ──────────────────────
app.include_router(mco_router)


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def _evict_sessions():
    """Evict oldest sessions when limit reached."""
    global omega_sessions, memory_sessions
    if len(omega_sessions) > MAX_SESSIONS:
        keys = list(omega_sessions.keys())[:MAX_SESSIONS // 4]
        for k in keys:
            omega_sessions.pop(k, None)
            memory_sessions.pop(k, None)
        logger.info(f"Evicted {len(keys)} sessions (limit: {MAX_SESSIONS})")


async def _persist_session(chat_id: str, kernel: OmegaCognitiveKernel, memory: MemoryEngine):
    """Persist session to Redis."""
    try:
        session_data = {
            "omega": kernel.serialize_session(),
            "memory": memory.serialize(),
        }
        await redis_client.setex(
            f"session:{chat_id}",
            settings.REDIS_SESSION_TTL,
            json.dumps(session_data, default=str),
        )
    except Exception as e:
        logger.warning(f"Session persist failed: {e}")


async def _restore_session(chat_id: str, user_id: str = ""):
    """Restore session from Redis."""
    try:
        cached = await redis_client.get(f"session:{chat_id}")
        if cached:
            data = json.loads(cached)
            kernel = OmegaCognitiveKernel.restore_from_session(
                data.get("omega", {}),
                sigma_orchestrator=orchestrator,
                knowledge_learner=knowledge_learner,
                cloud_client=mco_bridge,
            )
            memory = MemoryEngine.deserialize(data.get("memory", {}))
            return kernel, memory
    except Exception as e:
        logger.warning(f"Session restore failed: {e}")
    return None, None


async def _get_session(chat_id: str, user_id: str = ""):
    """Get or create session pair."""
    global omega_sessions, memory_sessions

    if chat_id in omega_sessions:
        return omega_sessions[chat_id], memory_sessions.get(chat_id, MemoryEngine(user_id=user_id))

    # Try Redis
    kernel, memory = await _restore_session(chat_id, user_id)
    if kernel:
        omega_sessions[chat_id] = kernel
        memory_sessions[chat_id] = memory
        return kernel, memory

    # Create new
    _evict_sessions()
    kernel = OmegaCognitiveKernel(
        sigma_orchestrator=orchestrator,
        knowledge_learner=knowledge_learner,
        cloud_client=mco_bridge,
    )
    memory = MemoryEngine(user_id=user_id)
    omega_sessions[chat_id] = kernel
    memory_sessions[chat_id] = memory
    return kernel, memory


# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.post("/api/auth/session")
async def create_session():
    """
    Bootstrap an anonymous session.
    Returns a JWT token for session continuity.
    """
    session_id = f"session-{uuid_lib.uuid4().hex[:16]}"
    access_token = create_access_token(session_id)
    refresh_token = create_refresh_token(session_id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "session_id": session_id,
    }


@app.post("/api/auth/refresh")
async def refresh_session(refresh_token: str = Body(..., embed=True)):
    """Refresh an expired access token."""
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    new_access = create_access_token(payload["sub"])
    return {
        "access_token": new_access,
        "token_type": "bearer",
    }


# ============================================================
# HEALTH & STATUS
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Sentinel-E",
        "version": "5.0.0",
    }


@app.get("/health")
async def health_check():
    """Production health check."""
    health = {
        "status": "healthy",
        "version": "5.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if redis_client:
            await redis_client.ping()
            health["redis"] = "connected"
        else:
            health["redis"] = "not_configured"
    except Exception:
        health["redis"] = "disconnected"
        health["status"] = "degraded"
    return health


# ============================================================
# MAIN EXECUTION ENDPOINT
# ============================================================

@app.post("/api/run")
async def run_sentinel(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
    frontend_context: Optional[str] = None,
):
    """
    Main entry point for Sentinel execution.
    Routes through Omega Cognitive Kernel.
    Protected by JWT auth, rate limiting, and prompt firewall.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System initializing. Please retry.")

    user_id = user["user_id"]
    firewall = get_firewall()

    # ── Input Validation ─────────────────────────────────────
    if len(request.text) > settings.MAX_INPUT_LENGTH:
        raise HTTPException(status_code=400, detail="Input too long.")

    request.rounds = min(request.rounds, settings.MAX_ROUNDS)

    # ── Prompt Firewall ──────────────────────────────────────
    verdict = firewall.analyze(request.text)
    if verdict.blocked:
        logger.warning(f"Firewall blocked input from {user_id}: {verdict.violations}")
        raise HTTPException(
            status_code=400,
            detail="Your input could not be processed. Please rephrase your question.",
        )
    effective_text = verdict.sanitized_text or request.text

    # ── Chat Resolution ──────────────────────────────────────
    chat = None
    if request.chat_id:
        chat = await get_chat(db, request.chat_id)
    if not chat:
        chat_name = generate_chat_name(effective_text, request.mode)
        chat = await create_chat(db, chat_name, request.mode, user_id=user_id)

    await add_message(db, chat.id, "user", effective_text)

    # ── Session & Memory ─────────────────────────────────────
    kernel, memory = await _get_session(str(chat.id), user_id)
    memory.add_message("user", effective_text)

    # ── Conversation History ─────────────────────────────────
    history = []
    try:
        stored = await get_chat_messages(db, chat.id)
        if len(stored) > 1:
            for msg in stored[-settings.SHORT_TERM_MEMORY_SIZE:]:
                history.append({"role": msg.role, "content": msg.content})
    except Exception as e:
        logger.warning(f"History retrieval failed: {e}")

    # ── Frontend Context (Sanitized) ─────────────────────────
    if frontend_context:
        try:
            ctx = json.loads(frontend_context)
            safe_ctx = firewall.validate_context_injection(ctx)
            st = safe_ctx.get("shortTerm", {})

            if st.get("isFollowUp") and st.get("resolvedQuery"):
                effective_text = st["resolvedQuery"]

            active_entity = st.get("activeEntity")
            active_topic = st.get("activeTopic")
            if active_entity or active_topic:
                ctx_hint = "Context: "
                if active_topic:
                    ctx_hint += f"topic is '{active_topic}'"
                if active_entity:
                    ctx_hint += f"{', ' if active_topic else ''}subject is '{active_entity}'"
                history.insert(0, {"role": "system", "content": ctx_hint})
        except Exception as e:
            logger.debug(f"Context injection skipped: {e}")

    # ── Memory Context Injection ─────────────────────────────
    memory_ctx = memory.build_prompt_context()
    if memory_ctx:
        history.insert(0, {"role": "system", "content": memory_ctx})

    # ── Cognitive RAG ────────────────────────────────────────
    rag_result = None
    if cognitive_rag:
        try:
            rag_result = await cognitive_rag.process(effective_text)
            if rag_result and rag_result.retrieval_executed and rag_result.sources:
                rag_context = "External evidence:\n" + "\n".join(
                    [f"- [{s.title}]({s.url}): {s.content[:200]}" for s in rag_result.sources[:3]]
                )
                history.append({"role": "system", "content": rag_context})
        except Exception as e:
            logger.warning(f"RAG failed: {e}")

    # ── Mode Resolution ──────────────────────────────────────
    omega_mode = request.mode
    mode_map = {"conversational": "standard", "forensic": "standard", "experimental": "research"}
    omega_mode = mode_map.get(omega_mode, omega_mode)

    sub_mode = getattr(request, "sub_mode", None) or "debate"
    kill = getattr(request, "kill", False)
    role_map = getattr(request, "role_map", None) or {}

    # ── Optimization: Observability Tracing ───────────────────
    obs_hub = get_observability_hub()
    request_id = str(uuid_lib.uuid4().hex[:12])
    tracer = obs_hub.start_request(session_id=str(chat.id), request_id=request_id)
    tracer.start_span("total")

    # ── Optimization: Response Cache Check ────────────────────
    cache = get_response_cache()
    cache_result = cache.lookup(effective_text, omega_mode)
    if cache_result.hit:
        cached_response = cache_result.response or {}
        cache_latency = tracer.end_span("total")
        tier_name = {1: "exact", 2: "lexical", 3: "semantic"}.get(cache_result.tier, "unknown")
        tracer.record_cache_hit(tier=tier_name, latency_ms=cache_latency)
        summary = tracer.finalize()
        obs_hub.record(summary)

        # Return cached response (preserving full response contract)
        return {
            "chat_id": str(chat.id),
            "chat_name": cached_response.get("chat_name", ""),
            "mode": cached_response.get("mode", omega_mode),
            "sub_mode": cached_response.get("sub_mode", sub_mode),
            "formatted_output": cached_response.get("formatted_output", ""),
            "data": {"priority_answer": cached_response.get("formatted_output", "")},
            "confidence": cached_response.get("confidence", 0.5),
            "session_state": cached_response.get("session_state", {}),
            "boundary_result": cached_response.get("boundary_result", {}),
            "omega_metadata": {**cached_response.get("omega_metadata", {}), "cache_hit": True, "cache_tier": tier_name},
        }
    tracer.record_cache_miss()

    # ── Optimization: Cost Governance ─────────────────────────
    governor = get_cost_governor()
    tier = "premium" if omega_mode in ("research", "experimental") else "standard"
    gov_decision = governor.check_budget(str(chat.id), requested_tier=tier)
    if not gov_decision.allowed:
        from optimization.observability import ObservabilityEvent, EventType
        tracer.record_event(ObservabilityEvent(event_type=EventType.BUDGET_EXCEEDED))
        logger.warning(f"Budget exceeded for chat {chat.id}: {gov_decision.reason}")
        raise HTTPException(status_code=429, detail=f"Session budget exceeded. {gov_decision.reason}")

    # Apply cost governor model recommendation
    if gov_decision.downgraded and gov_decision.recommended_model:
        logger.info(f"Cost governor downgraded model for chat {chat.id}: {gov_decision.recommended_model}")

    # ── Optimization: Token Optimization ──────────────────────
    token_optimizer = get_token_optimizer()

    # Separate system messages and user/assistant history
    system_msgs = [m for m in history if m.get("role") == "system"]
    conv_history = [m for m in history if m.get("role") != "system"]
    system_prompt = "\n".join(m.get("content", "") for m in system_msgs)

    opt_result = token_optimizer.optimize(
        query=effective_text,
        system_prompt=system_prompt,
        history=conv_history,
        context_window=settings.TOKEN_BUDGET_PER_REQUEST,
    )
    depth_assessment = opt_result.get("depth_assessment")
    if opt_result.get("compression_applied") or opt_result.get("deduped_history_count", 0) < len(conv_history):
        original_tokens = sum(len(m.get("content", "")) // 4 for m in history)
        opt_system = opt_result.get("system_prompt", system_prompt)
        opt_history_list = opt_result.get("history", conv_history)
        optimized_tokens = len(opt_system) // 4 + sum(len(m.get("content", "")) // 4 for m in opt_history_list)
        if original_tokens > optimized_tokens:
            tracer.record_token_optimization(
                original_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
            )
        # Rebuild history with optimized system prompt + conversation
        history = []
        if opt_system:
            history.append({"role": "system", "content": opt_system})
        history.extend(opt_history_list)

    # ══════════════════════════════════════════════════════════
    # COGNITIVE ENSEMBLE (v7.0) — Always-on, no mode routing
    # ══════════════════════════════════════════════════════════
    if cognitive_orchestrator_engine is not None:
        tracer.start_span("kernel")
        try:
            from core.ensemble_schemas import EnsembleFailure
            ensemble_response = await cognitive_orchestrator_engine.execute_cognitive_cycle(
                query=effective_text,
                chat_id=str(chat.id),
                rounds=max(request.rounds, 3),
            )
        except EnsembleFailure as ef:
            logger.error(f"Ensemble hard failure: {ef}")
            ensemble_response = ef.to_response()
            cognitive_orchestrator_engine_failed = False  # still return structured error
        except Exception as ens_err:
            logger.error(f"Ensemble engine crashed: {ens_err} — falling back to legacy kernel")
            cognitive_orchestrator_engine_failed = True
        else:
            cognitive_orchestrator_engine_failed = False

        if not cognitive_orchestrator_engine_failed:
            kernel_latency = tracer.end_span("kernel")
            payload = ensemble_response.to_frontend_payload()

            formatted_output = ensemble_response.final_answer
            if rag_result and rag_result.retrieval_executed:
                if rag_result.no_sources_found:
                    formatted_output += "\n\n*No verified external sources found for this query.*"
                elif rag_result.citations_text:
                    formatted_output += "\n\n" + rag_result.citations_text
                payload["formatted_output"] = formatted_output
                payload["final_answer"] = formatted_output

            confidence = ensemble_response.confidence

            omega_metadata = payload.get("omega_metadata", {})
            omega_metadata.update({
                "version": "7.0.0-cognitive",
                "mode": "ensemble",
                "sub_mode": "cognitive",
                "confidence": confidence,
                "entropy": ensemble_response.entropy,
                "fragility": ensemble_response.fragility,
                "ensemble_metrics": payload.get("ensemble_metrics", {}),
                "debate_rounds": payload.get("debate_rounds", []),
                "model_outputs": payload.get("model_outputs", []),
                "agreement_matrix": payload.get("agreement_matrix", {}),
                "drift_metrics": payload.get("drift_metrics", {}),
                "tactical_map": payload.get("tactical_map", {}),
                "confidence_graph": payload.get("calibrated_confidence", {}),
                "session_intelligence": payload.get("session_intelligence", {}),
                "model_status": payload.get("model_status", []),
                "reasoning_trace": {
                    "engine": "CognitiveCoreEngine",
                    "pipeline": "cognitive_v7",
                    "models_executed": ensemble_response.models_executed,
                    "models_succeeded": ensemble_response.models_succeeded,
                    "models_failed": ensemble_response.models_failed,
                    "debate_rounds": ensemble_response.debate_total_rounds,
                },
                "boundary_result": {
                    "risk_level": (
                        "LOW" if confidence > 0.7
                        else "MEDIUM" if confidence > 0.4
                        else "HIGH"
                    ),
                    "severity_score": int((1 - confidence) * 100),
                    "explanation": (
                        f"Ensemble confidence from {ensemble_response.models_executed} models, "
                        f"{ensemble_response.debate_total_rounds} debate rounds"
                    ),
                },
            })

            if rag_result and rag_result.retrieval_executed:
                omega_metadata["rag_result"] = {
                    "executed": True,
                    "source_count": rag_result.source_count,
                    "average_reliability": rag_result.average_reliability,
                    "contradictions": len(rag_result.contradictions),
                    "no_sources": rag_result.no_sources_found,
                }

            await update_chat_metadata(
                db, chat.id,
                priority_answer=formatted_output,
                machine_metadata=omega_metadata,
                rounds=request.rounds,
            )
            memory.add_message("assistant", formatted_output)
            await _persist_session(str(chat.id), kernel, memory)

            try:
                await redis_client.setex(
                    f"chat:{chat.id}:metadata",
                    settings.REDIS_SESSION_TTL,
                    json.dumps(omega_metadata, default=str),
                )
            except Exception:
                pass

            await add_message(db, chat.id, "assistant", formatted_output)

            response_payload = {
                **payload,
                "chat_id": str(chat.id),
                "mode": "ensemble",
                "sub_mode": "cognitive",
                "formatted_output": formatted_output,
                "confidence": confidence,
                "entropy": ensemble_response.entropy,
                "fragility": ensemble_response.fragility,
                "boundary_result": omega_metadata["boundary_result"],
                "omega_metadata": omega_metadata,
            }

            try:
                cache.store(effective_text, "ensemble", response_payload)
            except Exception:
                pass

            try:
                est_input_tokens = sum(len(m.get("content", "")) // 4 for m in history)
                est_output_tokens = len(formatted_output) // 4
                governor.record_usage(
                    session_id=str(chat.id),
                    model_id="ensemble",
                    input_tokens=est_input_tokens,
                    output_tokens=est_output_tokens,
                    latency_ms=kernel_latency,
                )
            except Exception:
                pass

            try:
                total_latency = tracer.end_span("total")
                tracer.record_model_call(
                    model_id="ensemble",
                    input_tokens=sum(len(m.get("content", "")) // 4 for m in history),
                    output_tokens=len(formatted_output) // 4,
                    latency_ms=kernel_latency,
                    cost_estimate=0.0,
                )
                summary = tracer.finalize()
                obs_hub.record(summary)
            except Exception:
                pass

            return response_payload

    # ══════════════════════════════════════════════════════════
    # LEGACY KERNEL PATH (fallback if ensemble unavailable)
    # ══════════════════════════════════════════════════════════
    config = ModeConfig.from_legacy(
        text=effective_text,
        mode=omega_mode,
        sub_mode=sub_mode,
        kill_switch=kill,
        enable_shadow=request.enable_shadow,
        rounds=request.rounds,
        chat_id=str(chat.id),
        history=history,
        role_map=role_map,
    )

    # ── Execute (Legacy) ─────────────────────────────────────
    tracer.start_span("kernel")
    result = await kernel.process(config)
    kernel_latency = tracer.end_span("kernel")

    # ── Extract & Build Response (Legacy) ────────────────────
    formatted_output = result.get("formatted_output", "")
    confidence = result.get("confidence", 0.5)
    session_state = result.get("session_state", {})
    reasoning_trace = result.get("reasoning_trace", {})
    boundary_result = result.get("boundary_result", {})

    # ── Inject RAG Citations ─────────────────────────────────
    if rag_result and rag_result.retrieval_executed:
        if rag_result.no_sources_found:
            formatted_output += "\n\n*No verified external sources found for this query.*"
        elif rag_result.citations_text:
            formatted_output += "\n\n" + rag_result.citations_text

    # ── Dynamic Analytics ────────────────────────────────────
    analytics = None
    if analytics_engine:
        model_outputs = []
        if result.get("omega_metadata", {}).get("aggregation_result"):
            agg = result["omega_metadata"]["aggregation_result"]
            if isinstance(agg, dict):
                for m in agg.get("model_outputs", []):
                    output_text = m.get("output", "") if isinstance(m, dict) else str(m)
                    if output_text and (not isinstance(m, dict) or not m.get("error")):
                        model_outputs.append(output_text)

        if model_outputs:
            analytics = analytics_engine.compute(
                model_outputs=model_outputs,
                evidence_sources=rag_result.source_count if rag_result else 0,
                contradiction_count=rag_result.contradiction_count if rag_result else 0,
                evidence_reliability=rag_result.average_reliability if rag_result else 0,
            )
            confidence = analytics.confidence  # Use dynamic confidence

    # ── Build Metadata ───────────────────────────────────────
    omega_metadata = {
        "version": "5.0.0",
        "mode": result.get("mode", omega_mode),
        "sub_mode": result.get("sub_mode", sub_mode),
        "confidence": confidence,
        "session_state": session_state,
        "reasoning_trace": reasoning_trace,
        "boundary_result": boundary_result,
    }

    # Mode-specific data
    for key in ["confidence_evolution", "fragility_index", "behavioral_risk",
                 "evidence_result", "stress_result", "confidence_components",
                 "debate_result"]:
        if result.get(key) is not None:
            omega_metadata[key] = result[key]

    # Dynamic analytics override
    if analytics:
        omega_metadata["confidence_components"] = analytics.confidence_components
        omega_metadata["boundary_result"] = {
            "risk_level": analytics.risk_level,
            "severity_score": int(analytics.boundary_risk * 100),
            "explanation": analytics.explanation,
            "risk_dimensions": analytics.boundary_components,
        }

    # Engine metadata passthrough
    if result.get("omega_metadata"):
        engine_meta = result["omega_metadata"]
        for key in ["aggregation_result", "forensic_result", "audit_result", "pipeline_steps"]:
            if engine_meta.get(key):
                omega_metadata[key] = engine_meta[key]

    # RAG metadata
    if rag_result and rag_result.retrieval_executed:
        omega_metadata["rag_result"] = {
            "executed": True,
            "source_count": rag_result.source_count,
            "average_reliability": rag_result.average_reliability,
            "contradictions": len(rag_result.contradictions),
            "no_sources": rag_result.no_sources_found,
        }

    # ── Knowledge Learning ───────────────────────────────────
    if knowledge_learner and boundary_result:
        severity = boundary_result.get("severity_score", 0)
        if severity > 40:
            try:
                knowledge_learner.record_boundary_violation(
                    model_name=omega_metadata.get("mode", "unknown"),
                    severity_score=severity,
                    severity_level=boundary_result.get("risk_level", "LOW"),
                    claim_type=boundary_result.get("claim_type", "unknown"),
                    run_id=str(chat.id),
                )
            except Exception as e:
                logger.warning(f"Knowledge learning failed: {e}")

    # ── Persist ──────────────────────────────────────────────
    await update_chat_metadata(
        db, chat.id,
        priority_answer=formatted_output,
        machine_metadata=omega_metadata,
        rounds=request.rounds,
    )
    memory.add_message("assistant", formatted_output)
    await _persist_session(str(chat.id), kernel, memory)

    try:
        await redis_client.setex(
            f"chat:{chat.id}:metadata",
            settings.REDIS_SESSION_TTL,
            json.dumps(omega_metadata, default=str),
        )
    except Exception:
        pass

    await add_message(db, chat.id, "assistant", formatted_output)

    # ── Rolling Summary Check ────────────────────────────────
    if memory.needs_summarization():
        try:
            summary_prompt = memory.generate_summary_prompt()
            if summary_prompt and mco_orchestrator and mco_orchestrator.cognitive_gateway:
                from metacognitive.schemas import CognitiveGatewayInput
                gw_input = CognitiveGatewayInput(
                    user_query=summary_prompt,
                    stabilized_context={},
                    knowledge_bundle=[],
                    session_summary={},
                )
                gw_output = await mco_orchestrator.cognitive_gateway.invoke_model(
                    "llama-3.3", gw_input
                )
                summary = gw_output.raw_output if gw_output.success else None
                if summary and not summary.startswith("Error"):
                    memory.rolling_summary.add_summary(summary, settings.ROLLING_SUMMARY_INTERVAL)
                    logger.info(f"Rolling summary generated for chat {chat.id}")
        except Exception as e:
            logger.debug(f"Summary generation skipped: {e}")

    # ── Optimization: Cache Store & Observability ─────────────
    response_payload = {
        "chat_id": str(chat.id),
        "chat_name": result.get("chat_name", ""),
        "mode": result.get("mode", omega_mode),
        "sub_mode": result.get("sub_mode", sub_mode),
        "formatted_output": formatted_output,
        "data": {"priority_answer": formatted_output},
        "confidence": confidence,
        "session_state": session_state,
        "boundary_result": omega_metadata.get("boundary_result", boundary_result),
        "omega_metadata": omega_metadata,
    }

    # Store in response cache (non-blocking, best-effort)
    try:
        cache.store(effective_text, omega_mode, response_payload)
    except Exception:
        pass

    # Record usage in cost governor
    try:
        # Estimate tokens from history length + output length
        est_input_tokens = sum(len(m.get("content", "")) // 4 for m in history)
        est_output_tokens = len(formatted_output) // 4
        governor.record_usage(
            session_id=str(chat.id),
            model_id=result.get("omega_metadata", {}).get("primary_model", "groq-small"),
            input_tokens=est_input_tokens,
            output_tokens=est_output_tokens,
            latency_ms=kernel_latency,
        )
    except Exception:
        pass

    # Finalize observability
    try:
        total_latency = tracer.end_span("total")
        tracer.record_model_call(
            model_id=result.get("omega_metadata", {}).get("primary_model", "groq-small"),
            input_tokens=est_input_tokens,
            output_tokens=est_output_tokens,
            latency_ms=kernel_latency,
            cost_estimate=0.0,
        )
        summary = tracer.finalize()
        obs_hub.record(summary)
    except Exception:
        pass

    # ── Response ─────────────────────────────────────────────
    return response_payload


# ============================================================
# FORM-DATA ENDPOINTS (Frontend Compatibility)
# ============================================================

@app.post("/run/standard")
async def run_standard_form(
    text: str = Form(...),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Standard mode via FormData."""
    request = SentinelRequest(text=text, mode="standard", chat_id=chat_id)
    return await run_sentinel(request, db, user, frontend_context=context)


@app.post("/run/experimental")
async def run_experimental_form(
    text: str = Form(...),
    mode: str = Form("experimental"),
    rounds: int = Form(6),
    kill_switch: bool = Form(False),
    sub_mode: str = Form("debate"),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Experimental mode via FormData."""
    if kill_switch:
        request = SentinelRequest(text=text or "kill", mode="kill", rounds=min(rounds, settings.MAX_ROUNDS), chat_id=chat_id, sub_mode="glass")
        return await run_sentinel(request, db, user)

    valid_sub_modes = {"debate", "glass", "evidence"}
    sub_mode = sub_mode if sub_mode in valid_sub_modes else "debate"
    valid_modes = {"conversational", "experimental", "forensic", "standard"}
    mode = mode if mode in valid_modes else "experimental"

    request = SentinelRequest(
        text=text, mode=mode, sub_mode=sub_mode,
        rounds=min(rounds, settings.MAX_ROUNDS), chat_id=chat_id,
    )
    return await run_sentinel(request, db, user, frontend_context=context)


@app.post("/run/omega/standard")
async def omega_standard_form(
    text: str = Form(...),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    request = SentinelRequest(text=text, mode="standard", chat_id=chat_id)
    return await run_sentinel(request, db, user)


@app.post("/run/omega/experimental")
async def omega_experimental_form(
    text: str = Form(...),
    rounds: int = Form(3),
    sub_mode: str = Form("debate"),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    valid_sub_modes = {"debate", "glass", "evidence"}
    sub_mode = sub_mode if sub_mode in valid_sub_modes else "debate"
    request = SentinelRequest(text=text, mode="experimental", sub_mode=sub_mode, rounds=min(rounds, settings.MAX_ROUNDS), chat_id=chat_id)
    return await run_sentinel(request, db, user)


@app.post("/run/omega/kill")
async def omega_kill_form(
    text: str = Form(""),
    chat_id: Optional[UUID] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    request = SentinelRequest(text=text or "kill", mode="kill", sub_mode="glass", chat_id=chat_id)
    return await run_sentinel(request, db, user)


@app.post("/run/ensemble")
async def run_ensemble_form(
    text: str = Form(...),
    rounds: int = Form(3),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """
    Ensemble endpoint — always-on multi-model reasoning.
    All requests route through CognitiveOrchestrator.
    Minimum 3 debate rounds enforced. No single-model fallback.
    """
    request = SentinelRequest(
        text=text,
        mode="ensemble",
        sub_mode="full_debate",
        rounds=max(min(rounds, settings.MAX_ROUNDS), 3),
        chat_id=chat_id,
    )
    return await run_sentinel(request, db, user, frontend_context=context)


# ============================================================
# FEEDBACK
# ============================================================

@app.post("/feedback")
async def feedback_endpoint(
    run_id: str = Form(...),
    feedback: str = Form(...),
    rating: Optional[int] = Form(None),
    reason: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    sub_mode: Optional[str] = Form(None),
    boundary_severity: Optional[float] = Form(None),
    fragility_index: Optional[float] = Form(None),
    disagreement_score: Optional[float] = Form(None),
    confidence: Optional[float] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Enhanced feedback with memory learning."""
    user_id = user["user_id"]

    try:
        chat_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    chat = await get_chat(db, chat_uuid)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Update chat metadata
    metadata = chat.machine_metadata or {}
    if "feedback" not in metadata:
        metadata["feedback"] = []

    feedback_entry = {
        "vote": feedback,
        "rating": rating,
        "reason": reason,
        "mode": mode,
        "sub_mode": sub_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(metadata.get("feedback"), list):
        metadata["feedback"].append(feedback_entry)
    else:
        metadata["feedback"] = [feedback_entry]

    await update_chat_metadata(db, chat.id, priority_answer=chat.priority_answer, machine_metadata=metadata, rounds=chat.rounds)

    # Memory learning
    memory = memory_sessions.get(str(chat.id))
    if memory:
        memory.record_feedback(
            vote=feedback,
            rating=rating,
            reason=reason,
            mode=mode,
        )

    # Knowledge learner
    if knowledge_learner:
        try:
            knowledge_learner.record_feedback(
                run_id=run_id, vote=feedback, rating=rating,
                mode=mode or "unknown", sub_mode=sub_mode, reason=reason,
            )
        except Exception as e:
            logger.warning(f"Knowledge learner feedback failed: {e}")

    return {"status": "success", "feedback_id": run_id}


# ============================================================
# CHAT HISTORY
# ============================================================

@app.get("/api/chats")
async def get_chats_list(
    limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    chats = await list_chats(db, limit, offset)
    return chats


@app.get("/api/history")
async def history_alias(
    limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    return await list_chats(db, limit, offset)


@app.get("/api/chat/{chat_id}")
async def get_chat_detail(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    chat = await get_chat(db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = await get_chat_messages(db, chat_id)
    return {"chat": chat, "messages": messages}


@app.get("/api/chat/{chat_id}/messages")
async def get_messages(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    msgs = await get_chat_messages(db, chat_id)
    return [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]


@app.get("/api/history/{chat_id}")
async def history_detail(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    return await get_chat_detail(chat_id, db, user)


# ============================================================
# SESSION & ANALYTICS ENDPOINTS
# ============================================================

@app.get("/api/omega/session/{chat_id}")
async def omega_session_state(
    chat_id: str,
    user: Dict = Depends(get_current_user),
):
    if chat_id in omega_sessions:
        kernel = omega_sessions[chat_id]
        return {
            "chat_id": chat_id,
            "session_state": kernel.get_session_state(),
            "initialized": kernel.is_initialized(),
        }
    kernel, _ = await _restore_session(chat_id)
    if kernel:
        return {
            "chat_id": chat_id,
            "session_state": kernel.get_session_state(),
            "initialized": kernel.is_initialized(),
        }
    return {"chat_id": chat_id, "session_state": None, "initialized": False}


@app.get("/api/session/{chat_id}/descriptive")
async def session_descriptive(
    chat_id: str,
    user: Dict = Depends(get_current_user),
):
    kernel = omega_sessions.get(chat_id)
    if not kernel:
        kernel, _ = await _restore_session(chat_id)
    if not kernel:
        return {"error": "Session not found", "chat_id": chat_id}
    try:
        return kernel.session.get_descriptive_summary()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/cross-analysis")
async def cross_model_analysis(
    chat_id: Optional[str] = Body(None),
    query: str = Body(""),
    llm_response: str = Body(""),
    user: Dict = Depends(get_current_user),
):
    if not mco_orchestrator or not mco_orchestrator.cognitive_gateway:
        raise HTTPException(status_code=503, detail="System not ready")

    try:
        if not llm_response and chat_id and chat_id in omega_sessions:
            kernel = omega_sessions[chat_id]
            session_state = kernel.get_session_state()
            llm_response = session_state.get("last_response", "")

        if not llm_response:
            llm_response = "No response available for analysis."

        from metacognitive.schemas import CognitiveGatewayInput
        analysis_prompt = (
            f"Analyze the following AI response for accuracy, completeness, and potential issues.\n\n"
            f"Original query: {query}\n\nAI Response:\n{llm_response}\n\n"
            f"Provide a structured analysis with: factual accuracy, completeness, potential biases, "
            f"confidence assessment, and suggested improvements."
        )
        gw_input = CognitiveGatewayInput(
            user_query=analysis_prompt,
            stabilized_context={},
            knowledge_bundle=[],
            session_summary={},
        )
        # Run cross-analysis through MCO cognitive gateway in parallel
        outputs = await mco_orchestrator.cognitive_gateway.invoke_parallel(gw_input)
        result = {
            "analyses": {
                out.model_name: {
                    "analysis": out.raw_output,
                    "success": out.success,
                    "latency_ms": round(out.latency_ms, 1),
                }
                for out in outputs
                if out.success
            },
            "models_used": [out.model_name for out in outputs if out.success],
            "total_models": len(outputs),
        }
        return result
    except Exception as e:
        logger.error(f"Cross-analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.get("/api/learning")
async def learning_summary(user: Dict = Depends(get_current_user)):
    if not knowledge_learner:
        return {"status": "disabled"}
    try:
        return {
            "status": "active",
            "summary": knowledge_learner.get_learning_summary(),
            "threshold_suggestions": knowledge_learner.suggest_threshold_adjustments(),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# PROVIDER STATUS (Admin Only)
# ============================================================

@app.get("/api/providers/status")
async def provider_status(user: Dict = Depends(get_current_user)):
    """Provider usage stats from unified cognitive registry."""
    from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
    return {
        "active_models": [
            {
                "id": key,
                "name": spec.name,
                "provider": spec.provider,
                "enabled": spec.enabled,
                "active": spec.active,
            }
            for key, spec in COGNITIVE_MODEL_REGISTRY.items()
        ],
    }


# ============================================================
# OPTIMIZATION STATS
# ============================================================

@app.get("/api/optimization/stats")
async def optimization_stats(user: Dict = Depends(get_current_user)):
    """Optimization layer metrics (non-sensitive)."""
    cache = get_response_cache()
    governor = get_cost_governor()
    obs_hub = get_observability_hub()

    return {
        "cache": cache.stats,
        "cost": governor.get_global_stats(),
        "observability": obs_hub.get_metrics(),
    }


@app.get("/api/optimization/session/{chat_id}")
async def optimization_session_stats(
    chat_id: str,
    user: Dict = Depends(get_current_user),
):
    """Per-session budget status."""
    governor = get_cost_governor()
    return governor.get_session_budget(chat_id)


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
