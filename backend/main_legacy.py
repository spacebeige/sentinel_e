import sys
import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, APIRouter, Body, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import get_db, init_db, check_redis, redis_client
from backend.database.crud import create_chat, get_chat, list_chats, update_chat_metadata, add_message, get_chat_messages
import json
import glob
from pathlib import Path
from datetime import datetime
from backend.sentinel.sentinel_sigma_v4 import SentinelSigmaOrchestratorV4, SigmaV4Config
from backend.sentinel.schemas import SentinelRequest, SentinelResponse, OmegaResponse
from backend.core.omega_kernel import OmegaCognitiveKernel, OmegaConfig
from backend.core.mode_config import ModeConfig, Mode, SubMode
from backend.core.knowledge_learner import KnowledgeLearner
from backend.utils.chat_naming import generate_chat_name
from backend.core.cross_model_analyzer import CrossModelAnalyzer, run_cross_analysis_on_response




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sentinel-API")

# Global State
orchestrator: Optional[SentinelSigmaOrchestratorV4] = None
omega_kernel: Optional[OmegaCognitiveKernel] = None
knowledge_learner: Optional[KnowledgeLearner] = None
# Per-session omega kernels (keyed by chat_id)
omega_sessions: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, omega_kernel, knowledge_learner
    logger.info("Initializing Sentinel-E Omega Cognitive Kernel v4.5...")
    
    # Initialize DB (Tables)
    await init_db()
    
    # Check Redis
    await check_redis()
    
    # Initialize Sigma V4 Orchestrator (LLM layer)
    orchestrator = SentinelSigmaOrchestratorV4()
    
    # Initialize Omega Cognitive Kernel (wraps Sigma V4)
    omega_kernel = OmegaCognitiveKernel(
        sigma_orchestrator=orchestrator,
        knowledge_learner=knowledge_learner,
    )
    
    # Initialize Knowledge Learner
    knowledge_learner = KnowledgeLearner()
    # Attach knowledge_learner to kernel after init
    omega_kernel.knowledge_learner = knowledge_learner
    logger.info("Omega Cognitive Kernel v3.X online. Knowledge Learner active.")
    
    yield
    
    logger.info("Shutting down Sentinel-E Omega System...")



app = FastAPI(title="Sentinel-E Omega Cognitive Kernel API", lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SESSION PERSISTENCE HELPERS
# ============================================================

async def _persist_omega_session(chat_id: str, kernel: OmegaCognitiveKernel):
    """Persist Omega kernel session state to Redis."""
    try:
        session_data = kernel.serialize_session()
        await redis_client.setex(
            f"omega:session:{chat_id}",
            3600,  # 1 hour TTL
            json.dumps(session_data, default=str)
        )
    except Exception as e:
        logger.warning(f"Failed to persist Omega session to Redis: {e}")


async def _restore_omega_session(chat_id: str) -> Optional[OmegaCognitiveKernel]:
    """Restore Omega kernel session from Redis."""
    try:
        cached = await redis_client.get(f"omega:session:{chat_id}")
        if cached:
            data = json.loads(cached)
            kernel = OmegaCognitiveKernel.restore_from_session(
                data, sigma_orchestrator=orchestrator, knowledge_learner=knowledge_learner
            )
            logger.info(f"Restored Omega session from Redis for chat {chat_id}")
            return kernel
    except Exception as e:
        logger.warning(f"Failed to restore Omega session from Redis: {e}")
    return None


def _get_omega_session_sync(chat_id: str) -> OmegaCognitiveKernel:
    """Get existing in-memory Omega session (sync, for immediate access)."""
    global omega_sessions, orchestrator, knowledge_learner
    if chat_id not in omega_sessions:
        omega_sessions[chat_id] = OmegaCognitiveKernel(
            sigma_orchestrator=orchestrator, knowledge_learner=knowledge_learner
        )
    return omega_sessions[chat_id]


async def _get_omega_session(chat_id: str) -> OmegaCognitiveKernel:
    """Get or create an Omega kernel session for a chat, with Redis restore."""
    global omega_sessions, orchestrator, knowledge_learner
    if chat_id in omega_sessions:
        return omega_sessions[chat_id]
    # Try to restore from Redis
    kernel = await _restore_omega_session(chat_id)
    if kernel:
        omega_sessions[chat_id] = kernel
        return kernel
    # Create new session
    omega_sessions[chat_id] = OmegaCognitiveKernel(
        sigma_orchestrator=orchestrator, knowledge_learner=knowledge_learner
    )
    return omega_sessions[chat_id]

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Sentinel-E Omega Cognitive Kernel",
        "version": "4.5.0",
        "modes": ["standard", "experimental"],
        "sub_modes": ["debate", "glass", "evidence"],
        "omega_active": omega_kernel is not None,
        "learning_active": knowledge_learner is not None,
    }

router = APIRouter(prefix="/api")

# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/run", response_model=None)
async def run_sentinel(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db),
    frontend_context: Optional[str] = None,
):
    """
    Main entry point for Sentinel execution.
    All modes route through Omega Cognitive Kernel for multipass reasoning,
    boundary evaluation, and session intelligence.
    SigmaV4 remains the LLM execution layer inside Omega.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        chat = None
        if request.chat_id:
            chat = await get_chat(db, request.chat_id)
        
        if not chat:
            # 1. Create Chat Record
            chat_name = generate_chat_name(request.text, request.mode)
            chat = await create_chat(db, chat_name, request.mode)
        
        # 2. Log User Message
        await add_message(db, chat.id, "user", request.text)

        # [Feature] Conversational Continuity
        history = []
        try:
            stored_messages = await get_chat_messages(db, chat.id)
            if len(stored_messages) > 1:
                context_messages = stored_messages[:-1] 
                for msg in context_messages[-20:]: 
                    history.append({"role": msg.role, "content": msg.content})
        except Exception as e:
            logger.warning(f"Failed to retrieve chat history: {e}")

        # [Feature] Frontend Context Injection (vNext memory layer)
        resolved_text = request.text
        if frontend_context:
            try:
                ctx = json.loads(frontend_context)
                stm = ctx.get("shortTerm", {})
                prefs = ctx.get("preferences", {})

                # Pronoun resolution: use resolved query if it's a follow-up
                if stm.get("isFollowUp") and stm.get("resolvedQuery"):
                    resolved_text = stm["resolvedQuery"]
                    logger.info(f"Context Injection: resolved follow-up query → {resolved_text[:80]}")

                # Inject active entity/topic as a subtle system context
                active_entity = stm.get("activeEntity")
                active_topic = stm.get("activeTopic")
                if active_entity or active_topic:
                    context_hint = "Context: "
                    if active_topic:
                        context_hint += f"topic is '{active_topic}'"
                    if active_entity:
                        context_hint += f"{', ' if active_topic else ''}subject is '{active_entity}'"
                    # Prepend as system context (won't displace user messages)
                    history.insert(0, {"role": "system", "content": context_hint})

                # Log preference signals for future backend tuning
                if prefs:
                    logger.debug(f"User prefs: verbosity={prefs.get('verbosity')}, analytics={prefs.get('analyticsVisibility')}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse frontend context: {e}")

        # Use resolved text for Omega processing
        effective_text = resolved_text if resolved_text != request.text else request.text
        
        # 3. Route through Omega Cognitive Kernel
        # Map legacy modes to new mode system
        omega_mode = request.mode
        if omega_mode == "conversational":
            omega_mode = "standard"
        elif omega_mode == "forensic":
            omega_mode = "standard"  # Forensic treated as standard with shadow
        elif omega_mode == "experimental":
            omega_mode = "research"  # Experimental → Research

        kernel = await _get_omega_session(str(chat.id))

        # Resolve sub_mode
        sub_mode = getattr(request, 'sub_mode', None) or "debate"
        kill = getattr(request, 'kill', False)

        # Build ModeConfig (v3.X) directly for new system
        role_map = getattr(request, 'role_map', None) or {}
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

        result = await kernel.process(config)

        # 4. Extract response fields
        formatted_output = result.get("formatted_output", "")
        confidence = result.get("confidence", 0.5)
        session_state = result.get("session_state", {})
        reasoning_trace = result.get("reasoning_trace", {})
        boundary_result = result.get("boundary_result", {})
        # 5. Build unified metadata
        omega_metadata = {
            "omega_version": "4.5.0",
            "mode": result.get("mode", omega_mode),
            "sub_mode": result.get("sub_mode", sub_mode),
            "original_mode": request.mode,
            "confidence": confidence,
            "session_state": session_state,
            "reasoning_trace": reasoning_trace,
            "boundary_result": boundary_result,
        }

        # Include sub-mode specific data
        if result.get("confidence_evolution"):
            omega_metadata["confidence_evolution"] = result["confidence_evolution"]
        if result.get("fragility_index") is not None:
            omega_metadata["fragility_index"] = result["fragility_index"]
        if result.get("behavioral_risk"):
            omega_metadata["behavioral_risk"] = result["behavioral_risk"]
        if result.get("evidence_result"):
            omega_metadata["evidence_result"] = result["evidence_result"]
        if result.get("kill_active") is not None:
            omega_metadata["kill_active"] = result["kill_active"]
        if result.get("stress_result"):
            omega_metadata["stress_result"] = result["stress_result"]
        if result.get("confidence_components"):
            omega_metadata["confidence_components"] = result["confidence_components"]
        if result.get("debate_result"):
            omega_metadata["debate_result"] = result["debate_result"]

        # v4 Engine metadata (aggregation, forensic, audit pipeline data)
        if result.get("omega_metadata"):
            engine_meta = result["omega_metadata"]
            if engine_meta.get("aggregation_result"):
                omega_metadata["aggregation_result"] = engine_meta["aggregation_result"]
            if engine_meta.get("forensic_result"):
                omega_metadata["forensic_result"] = engine_meta["forensic_result"]
            if engine_meta.get("audit_result"):
                omega_metadata["audit_result"] = engine_meta["audit_result"]
            if engine_meta.get("pipeline_steps"):
                omega_metadata["pipeline_steps"] = engine_meta["pipeline_steps"]

        # Wire KnowledgeLearner — record boundary violations
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
                    logger.warning(f"KnowledgeLearner violation recording failed: {e}")

        # 6. Persist to DB
        await update_chat_metadata(
            db,
            chat.id,
            priority_answer=formatted_output,
            machine_metadata=omega_metadata,
            rounds=request.rounds,
        )

        # 7. Persist Omega session to Redis
        await _persist_omega_session(str(chat.id), kernel)

        # 8. Cache metadata in Redis
        try:
            await redis_client.setex(
                f"chat:{chat.id}:metadata",
                3600,
                json.dumps(omega_metadata, default=str)
            )
        except Exception as e:
            logger.warning(f"Redis Cache Error: {e}")
        
        # 9. Log System Message
        await add_message(db, chat.id, "assistant", formatted_output)
        
        # 10. Return unified Omega response
        return {
            "chat_id": str(chat.id),
            "chat_name": result.get("chat_name", ""),
            "mode": result.get("mode", omega_mode),
            "sub_mode": result.get("sub_mode", sub_mode),
            "original_mode": request.mode,
            "formatted_output": formatted_output,
            "data": {
                "priority_answer": formatted_output,
            },
            "confidence": confidence,
            "session_state": session_state,
            "reasoning_trace": reasoning_trace,
            "boundary_result": boundary_result,
            "omega_metadata": omega_metadata,
        }

    except Exception as e:
        logger.error(f"Execution Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run/experimental", response_model=SentinelResponse)
async def run_experimental(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db)
):
    """Alias for experimental mode"""
    request.mode = "experimental"
    return await run_sentinel(request, db)

@router.post("/run/forensic", response_model=SentinelResponse)
async def run_forensic(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db)
):
    """Alias for forensic mode"""
    request.mode = "forensic"
    return await run_sentinel(request, db)

@router.post("/run/shadow", response_model=SentinelResponse)
async def run_shadow_route(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Explicit Shadow Execution.
    Forces enable_shadow=True.
    Mode defaults to forensic if not specified, or keeps user generic mode.
    """
    request.enable_shadow = True
    # "Shadow Mode" isn't a primary mode, it's an overlay. But if user asks for /run/shadow
    # we treat it as forensic audit primarily unless specified otherwise.
    if request.mode == "conversational":
        request.mode = "forensic"
        
    return await run_sentinel(request, db)

# ============================================================
# OMEGA COGNITIVE KERNEL ENDPOINTS
# ============================================================

@router.post("/omega/run")
async def omega_run(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Omega Cognitive Kernel direct entry point.
    Delegates to the unified run_sentinel which routes through Omega.
    """
    return await run_sentinel(request, db)


@router.post("/omega/standard")
async def omega_standard(request: SentinelRequest, db: AsyncSession = Depends(get_db)):
    """Omega STANDARD mode alias."""
    request.mode = "standard"
    return await omega_run(request, db)


@router.post("/omega/experimental")
async def omega_experimental(request: SentinelRequest, db: AsyncSession = Depends(get_db)):
    """Omega EXPERIMENTAL mode alias."""
    request.mode = "experimental"
    return await omega_run(request, db)


@router.post("/omega/kill")
async def omega_kill(request: SentinelRequest, db: AsyncSession = Depends(get_db)):
    """
    Omega KILL mode — diagnostic only.
    Returns session cognitive state snapshot without generating new reasoning.
    """
    request.mode = "kill"
    return await omega_run(request, db)


@router.get("/omega/session/{chat_id}")
async def omega_session_state(chat_id: str):
    """Get Omega session intelligence state for a chat."""
    # Try in-memory first, then Redis
    if chat_id in omega_sessions:
        kernel = omega_sessions[chat_id]
        return {
            "chat_id": chat_id,
            "session_state": kernel.get_session_state(),
            "boundary_trend": kernel.get_boundary_trend(),
            "initialized": kernel.is_initialized(),
        }
    # Try Redis restore
    kernel = await _restore_omega_session(chat_id)
    if kernel:
        return {
            "chat_id": chat_id,
            "session_state": kernel.get_session_state(),
            "boundary_trend": kernel.get_boundary_trend(),
            "initialized": kernel.is_initialized(),
        }
    return {"chat_id": chat_id, "session_state": None, "initialized": False}


@router.post("/cross-analysis")
async def cross_model_analysis(
    chat_id: Optional[str] = Body(None),
    query: str = Body(""),
    llm_response: str = Body(""),
):
    """
    Run the 8-step cross-model behavioral analysis pipeline.
    Auto-triggered in glass mode — no user input required.
    
    If llm_response is provided, all models analyze it.
    If chat_id is provided, fetches the latest response from that chat.
    """
    if not orchestrator or not hasattr(orchestrator, 'client'):
        raise HTTPException(status_code=503, detail="Cloud model client not initialized")

    try:
        # If no llm_response but chat_id, try to get latest response
        if not llm_response and chat_id:
            if chat_id in omega_sessions:
                kernel = omega_sessions[chat_id]
                session_state = kernel.get_session_state()
                llm_response = session_state.get("last_response", "")

        if not llm_response:
            llm_response = "No response available for analysis."

        result = await run_cross_analysis_on_response(
            cloud_client=orchestrator.client,
            llm_response=llm_response,
            query=query,
        )
        return result

    except Exception as e:
        logger.error(f"Cross-analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-analysis/models")
async def get_analyzed_models():
    """Return the list of models being analyzed in the cross-analysis pipeline."""
    from backend.core.cross_model_analyzer import ANALYZED_MODELS, ANALYSIS_STEPS
    return {
        "analyzed_models": ANALYZED_MODELS,
        "analysis_steps": ANALYSIS_STEPS,
        "total_steps": len(ANALYSIS_STEPS),
    }


@router.get("/chats")
async def get_chats_history(
    limit: int = 50, 
    offset: int = 0, 
    db: AsyncSession = Depends(get_db)
):
    """List recent chats"""
    chats = await list_chats(db, limit, offset)
    return chats

@router.get("/chat/{chat_id}")
async def get_chat_details(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get full chat details including messages"""
    chat = await get_chat(db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    # Get messages
    # Assuming crud function exists or we add it to response
    from backend.database.crud import get_chat_messages
    messages = await get_chat_messages(db, chat_id)
    
    return {
        "chat": chat,
        "messages": messages
    }

@router.get("/chat/{chat_id}/messages")
async def get_messages_for_chat(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Return all messages for a chat as clean JSON, ordered oldest first."""
    msgs = await get_chat_messages(db, chat_id)
    return [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]

@router.post("/chat/share")
async def share_chat(
    chat_id: UUID = Body(..., embed=True),
    db: AsyncSession = Depends(get_db)
):
    """Generate a share token (Dummy implementation for contract)"""
    # Verify chat exists
    chat = await get_chat(db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return {"share_token": f"share_{chat_id}_token_12345"}

# ============================================================
# COMPATIBILITY ROUTES (Frontend Legacy Support)
# ============================================================
@router.get("/history")
async def history_alias(
    limit: int = 50, 
    offset: int = 0, 
    db: AsyncSession = Depends(get_db)
):
    """Alias for /api/chats to match frontend"""
    return await list_chats(db, limit, offset)

# Note: This route is on 'app' directly because frontend calls /run/standard (no /api prefix)
@app.post("/run/standard")
async def run_standard_alias(
    text: str = Form(...),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Standard mode via FormData — routes through Omega pipeline."""
    request = SentinelRequest(text=text, mode="standard", chat_id=chat_id)
    return await run_sentinel(request, db, frontend_context=context)

@app.get("/api/history/{chat_id}")
async def history_detail_alias(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Alias for /api/chat/{chat_id}"""
    return await get_chat_details(chat_id, db)

@app.post("/run/experimental")
async def run_experimental_alias_form(
    text: str = Form(...),
    mode: str = Form("experimental"), 
    rounds: int = Form(6),
    kill_switch: bool = Form(False),
    sub_mode: str = Form("debate"),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Experimental mode via FormData — routes through Omega pipeline.
    Sub-modes: debate, glass, evidence.
    If kill_switch=true, routes to glass+kill.
    """
    # Handle kill switch — route to glass + kill
    if kill_switch:
        request = SentinelRequest(text=text or "kill", mode="kill", rounds=rounds, chat_id=chat_id, sub_mode="glass")
        return await run_sentinel(request, db)

    # Validate sub_mode
    valid_sub_modes = ["debate", "glass", "evidence"]
    if sub_mode not in valid_sub_modes:
        sub_mode = "debate"

    # Validate Mode
    valid_modes = ["conversational", "experimental", "forensic", "standard"]
    if mode not in valid_modes:
        mode = "experimental"

    request = SentinelRequest(
        text=text, 
        mode=mode,
        sub_mode=sub_mode,
        rounds=rounds,
        chat_id=chat_id
    )
    return await run_sentinel(request, db, frontend_context=context)

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
    db: AsyncSession = Depends(get_db)
):
    """
    Enhanced feedback — 1-5 rating, full context, wired to KnowledgeLearner.
    PATCH: Stores feedback in Chat.machine_metadata + KnowledgeLearner.
    """
    logger.info(f"Feedback received for {run_id}: {feedback} rating={rating} - {reason}")
    
    try:
        # Validate run_id is UUID
        try:
            chat_uuid = UUID(run_id)
        except ValueError:
            logger.warning(f"Invalid UUID for feedback: {run_id}")
            return {"status": "error", "message": "Invalid Run ID"}

        chat = await get_chat(db, chat_uuid)
        if not chat:
            logger.warning(f"Chat not found for feedback: {run_id}")
            return {"status": "error", "message": "Chat not found"}
        
        # Merge feedback into machine_metadata
        metadata = chat.machine_metadata or {}
        if "feedback" not in metadata:
            metadata["feedback"] = []
            
        feedback_entry = {
            "vote": feedback,
            "rating": rating,
            "reason": reason,
            "mode": mode,
            "sub_mode": sub_mode,
            "boundary_severity": boundary_severity,
            "fragility_index": fragility_index,
            "disagreement_score": disagreement_score,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Ensure it's a list (handle legacy data if any)
        if isinstance(metadata.get("feedback"), list):
            metadata["feedback"].append(feedback_entry)
        else:
             metadata["feedback"] = [feedback_entry]
             
        chat.machine_metadata = metadata
        
        from backend.database.crud import update_chat_metadata
        await update_chat_metadata(
            db, 
            chat.id, 
            priority_answer=chat.priority_answer,
            machine_metadata=metadata,
            rounds=chat.rounds
        )

        # Wire to KnowledgeLearner
        if knowledge_learner:
            try:
                knowledge_learner.record_feedback(
                    run_id=run_id,
                    vote=feedback,
                    rating=rating,
                    mode=mode or "unknown",
                    sub_mode=sub_mode,
                    reason=reason,
                )
            except Exception as e:
                logger.warning(f"KnowledgeLearner feedback recording failed: {e}")
        
        return {"status": "success", "feedback_id": run_id, "storage": "postgres", "learning": knowledge_learner is not None}

    except Exception as e:
        logger.error(f"Feedback DB Error: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}

app.include_router(router)

# ============================================================
# OMEGA FORM-DATA ENDPOINTS (Frontend Integration)
# ============================================================

@app.post("/run/omega/standard")
async def omega_standard_form(
    text: str = Form(...),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    """Omega STANDARD mode via FormData."""
    request = SentinelRequest(text=text, mode="standard", chat_id=chat_id)
    return await run_sentinel(request, db)

@app.post("/run/omega/experimental")
async def omega_experimental_form(
    text: str = Form(...),
    rounds: int = Form(3),
    sub_mode: str = Form("debate"),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    """Omega EXPERIMENTAL mode via FormData. Sub-modes: debate, glass, evidence."""
    valid_sub_modes = ["debate", "glass", "evidence"]
    if sub_mode not in valid_sub_modes:
        sub_mode = "debate"
    request = SentinelRequest(text=text, mode="experimental", sub_mode=sub_mode, rounds=rounds, chat_id=chat_id)
    return await run_sentinel(request, db)

@app.post("/run/omega/kill")
async def omega_kill_form(
    text: str = Form(""),
    chat_id: Optional[UUID] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Omega KILL mode — routes to Glass + Kill for backward compat."""
    request = SentinelRequest(text=text or "kill", mode="kill", sub_mode="glass", chat_id=chat_id)
    return await run_sentinel(request, db)


# ============================================================
# SENTINEL-E v4.5 API ROUTES
# ============================================================

@app.get("/health")
async def health_check():
    """Production health check — DB, Redis, models."""
    health = {
        "status": "healthy",
        "version": "4.5.0",
        "omega_kernel": omega_kernel is not None,
        "knowledge_learner": knowledge_learner is not None,
        "orchestrator": orchestrator is not None,
    }
    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
            health["redis"] = "connected"
        else:
            health["redis"] = "not_configured"
    except Exception:
        health["redis"] = "disconnected"
        health["status"] = "degraded"
    # Check DB
    try:
        from backend.database.connection import get_db
        health["database"] = "connected"
    except Exception:
        health["database"] = "disconnected"
        health["status"] = "degraded"
    return health


@app.get("/kernel-status")
async def kernel_status():
    """Omega kernel introspection — active sessions, boundary trends."""
    if not omega_kernel:
        return {"status": "offline", "message": "Omega kernel not initialized"}
    
    status = {
        "status": "online",
        "version": "4.5.0",
        "active_sessions": len(omega_sessions),
        "session_ids": list(omega_sessions.keys())[:20],
        "sub_modes": ["debate", "glass", "evidence"],
        "behavioral_analyzer": hasattr(omega_kernel, 'behavioral') and omega_kernel.behavioral is not None,
        "evidence_engine": hasattr(omega_kernel, 'evidence') and omega_kernel.evidence is not None,
    }
    return status


@app.get("/session-stats")
async def session_stats(db: AsyncSession = Depends(get_db)):
    """Aggregate session analytics."""
    try:
        chats = await list_chats(db)
        total = len(chats)
        modes = {}
        for c in chats:
            m = c.mode or "unknown"
            modes[m] = modes.get(m, 0) + 1
        
        return {
            "total_sessions": total,
            "mode_distribution": modes,
            "active_omega_sessions": len(omega_sessions),
        }
    except Exception as e:
        logger.error(f"Session stats error: {e}")
        return {"error": str(e)}


@app.get("/api/learning")
async def learning_summary():
    """KnowledgeLearner summary — boundary violations, feedback trends, threshold suggestions."""
    if not knowledge_learner:
        return {"status": "disabled", "message": "KnowledgeLearner not initialized"}
    
    try:
        summary = knowledge_learner.get_learning_summary()
        suggestions = knowledge_learner.suggest_threshold_adjustments()
        risk_profiles = knowledge_learner.get_all_risk_profiles()
        claim_risks = knowledge_learner.get_claim_type_risk_summary()
        
        return {
            "status": "active",
            "summary": summary,
            "threshold_suggestions": suggestions,
            "risk_profiles": risk_profiles,
            "claim_type_risks": claim_risks,
        }
    except Exception as e:
        logger.error(f"Learning summary error: {e}")
        return {"status": "error", "detail": str(e)}


@app.get("/api/learning/risk-profiles")
async def learning_risk_profiles():
    """Model risk profiles from KnowledgeLearner."""
    if not knowledge_learner:
        return {"status": "disabled"}
    try:
        return knowledge_learner.get_all_risk_profiles()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/session/{chat_id}/descriptive")
async def session_descriptive(chat_id: str):
    """
    Descriptive session summary for right panel display.
    v4.5: All metrics have human-readable labels, no raw floats.
    """
    kernel = omega_sessions.get(chat_id)
    if not kernel:
        # Try Redis restore
        kernel = await _restore_omega_session(chat_id)
    if not kernel:
        return {"error": "Session not found", "chat_id": chat_id}
    
    try:
        return kernel.session.get_descriptive_summary()
    except Exception as e:
        logger.error(f"Descriptive summary error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("\nChoose backend mode:")
    print("1. Offline (mock responses)")
    print("2. Normal (live AI)")
    choice = input("Enter 1 for offline, 2 for normal: ").strip()
    if choice == "1":
        os.environ["SENTINEL_MOCK_MODE"] = "true"
        print("Backend will run in OFFLINE MOCK mode.")
    else:
        os.environ["SENTINEL_MOCK_MODE"] = "false"
        print("Backend will run in NORMAL (live AI) mode.")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
