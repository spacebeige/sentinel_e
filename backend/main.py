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
from backend.sentinel.schemas import SentinelRequest, SentinelResponse
from backend.utils.chat_naming import generate_chat_name




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sentinel-API")

# Global State
orchestrator: Optional[SentinelSigmaOrchestratorV4] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    logger.info("Initializing Sentinel System...")
    
    # Initialize DB (Tables)
    await init_db()
    
    # Check Redis
    await check_redis()
    
    # Initialize Orchestrator
    orchestrator = SentinelSigmaOrchestratorV4()
    
    yield
    
    logger.info("Shutting down Sentinel System...")



app = FastAPI(title="Sentinel-Σ v4 API", lifespan=lifespan)

# Added for Cloudflare Tunnel deployment: Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "service": "Sentinel-Σ v4 API"}

router = APIRouter(prefix="/api")

# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/run", response_model=SentinelResponse)
async def run_sentinel(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Main entry point for Sentinel execution.
    Supports all modes via 'mode' parameter.
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
        # Fetch previous messages to maintain context like ChatGPT
        history = []
        try:
            stored_messages = await get_chat_messages(db, chat.id)
            # Exclude the message we just added (the last one) to avoid duplication in prompt + history context
            # taking up to 20 previous messages
            if len(stored_messages) > 1:
                context_messages = stored_messages[:-1] 
                for msg in context_messages[-20:]: 
                    history.append({"role": msg.role, "content": msg.content})
        except Exception as e:
            logger.warning(f"Failed to retrieve chat history: {e}")
        
        # 3. Execute Orchestrator
        config = SigmaV4Config(
            text=request.text,
            mode=request.mode,
            enable_shadow=request.enable_shadow,
            rounds=request.rounds,
            chat_id=str(chat.id),
            history=history
        )
        
        result = await orchestrator.run_sentinel(config)
        
        # 4. Update Chat Metadata
        machine_metadata = result.metadata.model_dump()
        
        # [PERSISTENCE FIX] - Inject debate data for experimental mode analysis
        if request.mode == "experimental" or request.mode == "forensic":
             machine_metadata["debate_data"] = {
                "model_positions": result.data.get("model_positions", []),
                "agreements": result.data.get("agreements", []),
                "disagreements": result.data.get("disagreements", [])
             }

        # Redis Cache for Hot Retrieval
        try:
            await redis_client.setex(
                f"chat:{chat.id}:metadata",
                3600, # 1 hour TTL
                json.dumps(machine_metadata, default=str)
            )
        except Exception as e:
            logger.warning(f"Redis Cache Error: {e}")
        
        # Access data fields from result.data per strict contract
        priority_answer = result.data.get("priority_answer")
        shadow_metadata = result.data.get("shadow_analysis", None)
        # Convert shadow_metadata (dict or obj) to dict if needed
        if hasattr(shadow_metadata, "model_dump"):
            shadow_metadata = shadow_metadata.model_dump()

        await update_chat_metadata(
            db,
            chat.id,
            priority_answer=priority_answer,
            machine_metadata=machine_metadata,
            shadow_metadata=shadow_metadata,
            rounds=result.metadata.rounds_executed,
            models_used=result.metadata.models_used
        )
        
        # 5. Log System Message
        await add_message(db, chat.id, "assistant", priority_answer)
        
        return result

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
@app.post("/run/standard", response_model=SentinelResponse)
async def run_standard_alias(
    text: str = Form(...),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    """Alias for /api/run with mode='standard' to match frontend FormData"""
    request = SentinelRequest(text=text, mode="standard", chat_id=chat_id)
    # We call the logic of run_sentinel directly
    return await run_sentinel(request, db)

@app.get("/api/history/{chat_id}")
async def history_detail_alias(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Alias for /api/chat/{chat_id}"""
    return await get_chat_details(chat_id, db)

@app.post("/run/experimental", response_model=SentinelResponse)
async def run_experimental_alias_form(
    text: str = Form(...),
    mode: str = Form("experimental"), 
    rounds: int = Form(6),
    kill_switch: bool = Form(False),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Alias for /api/run/experimental allowing FormData.
    Frontend sends FormData, so we need to receive Form fields and construct the request.
    PATCH: Respect incoming mode (forensic/experimental) instead of hardcoding.
    """
    # 1. Validate Mode
    valid_modes = ["conversational", "experimental", "forensic", "standard"]
    if mode not in valid_modes:
        mode = "experimental" # Fallback

    # 2. Construct Request with dynamic mode
    request = SentinelRequest(
        text=text, 
        mode=mode,
        rounds=rounds,
        chat_id=chat_id
    )
    return await run_sentinel(request, db)

@app.post("/feedback")
async def feedback_endpoint(
    run_id: str = Form(...),
    feedback: str = Form(...),
    reason: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Receives feedback for a run.
    PATCH: Stores feedback in Chat.machine_metadata instead of filesystem to fix persistence.
    """
    logger.info(f"Feedback received for {run_id}: {feedback} - {reason}")
    
    try:
        # Patch: Store in DB directly
        # Validate run_id is UUID
        try:
            chat_uuid = UUID(run_id)
        except ValueError:
            # If not valid UUID, fallback or fail. For integrity, we fail cleanly.
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
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Ensure it's a list (handle legacy data if any)
        if isinstance(metadata.get("feedback"), list):
            metadata["feedback"].append(feedback_entry)
        else:
             metadata["feedback"] = [feedback_entry] # Overwrite/Reset if malformed
             
        # Commit update
        # We need to explicitly re-assign to trigger dirty flag for JSONB sometimes, 
        # or use update_chat_metadata
        chat.machine_metadata = metadata
        # Also update priority_answer if needed? No.
        
        # Using CRUD update ensures timestamps are refreshed
        from backend.database.crud import update_chat_metadata
        await update_chat_metadata(
            db, 
            chat.id, 
            priority_answer=chat.priority_answer,
            machine_metadata=metadata,
            # Pass existing values for others
            rounds=chat.rounds
        )
        
        return {"status": "success", "feedback_id": run_id, "storage": "postgres"}

    except Exception as e:
        logger.error(f"Feedback DB Error: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
