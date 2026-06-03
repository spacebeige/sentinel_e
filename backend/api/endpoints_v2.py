"""
============================================================
API Endpoints v2 — Deterministic, Persistent, Safe
============================================================

Implements PHASE 2-10 API contracts:

Endpoints:
  • POST /api/session — Create session
  • GET /api/history — Load chat history (with all messages)
  • POST /api/chat — Create new chat
  • GET /api/chat/{id} — Get chat (for page refresh)
  • POST /api/chat/{id}/message — Add message to chat
  • GET /api/chat/{id}/messages — Get chat messages
  • GET /api/memory — Get user memory
  • POST /api/memory — Upsert memory
  • GET /api/user/settings — Get user settings
  • PUT /api/user/settings — Update settings
  • GET /api/context — Get context window for LLM
  • GET /api/user — Get current user

Principles:
  • Every endpoint requires auth
  • All responses: {success, data, error}
  • Never return null values
  • Update session on every request
  • Persist all writes to Neon

Usage in main.py:
    from api.endpoints_v2 import router as api_v2
    app.include_router(api_v2, prefix="/api")
"""

import logging
from typing import Optional, Dict, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from database.connection_v2 import get_db
from database.crud_v2 import (
    upsert_user, get_user_by_id,
    create_session, update_session_activity, get_session, list_user_sessions,
    create_chat, get_chat, list_user_chats, update_chat_title, update_chat_metadata, archive_chat,
    add_message, get_chat_messages, get_message, soft_delete_message,
    upsert_memory, get_user_memory, get_memory_by_key,
    upsert_user_setting, get_user_settings, get_user_stats,
)
from gateway.auth_v2 import get_current_user, ensure_user_exists as ensure_user
from database.crud_v2 import insert_analytics_event
from utils.safe_responses import (
    success, error, 
    user_to_dict, session_to_dict, chat_to_dict, message_to_dict, memory_to_dict,
    chat_history_response, context_window_response,
    empty_history_structure, empty_memory_structure,
)

logger = logging.getLogger("API-v2")

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# DEPENDENCY: Get current user (with DB session)
# ─────────────────────────────────────────────────────────────

async def get_current_user_with_db(
    user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[Dict[str, Any], str, AsyncSession]:
    """
    Dependency that provides: (user_dict, user_id, db_session)
    
    Also ensures user exists in database.
    """
    user_id = await ensure_user(user, db)
    # Instrumentation: log user_id here for request-level visibility
    try:
        logger.info(f"USER_ID (dependency): {user_id}")
        print("USER_ID:", user_id)
    except Exception:
        pass

    return user, user_id, db


# ─────────────────────────────────────────────────────────────
# SESSION ENDPOINTS
# ─────────────────────────────────────────────────────────────

@router.post("/session")
async def create_user_session(
    request: Request,
    client: str = "web",
    payload: Dict[str, Any] = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Create new user session.
    
    Called on app load/login.
    
    Request:
        POST /api/session
        {client: "web" | "mobile" | "api"}
    
    Response:
        {
            success: true,
            data: {
                session_id: UUID,
                user_id: string,
                created_at: ISO8601,
                last_active_at: ISO8601
            },
            error: null
        }
    """
    try:
        _, user_id, db = payload
        
        # PATCH 6: Log debug header if present
        try:
            debug_user = request.headers.get("x-debug-user") if hasattr(request, "headers") else None
            if debug_user:
                logger.info(f"HEADER_DEBUG_USER: {debug_user} vs EXTRACTED_USER_ID: {user_id}")
        except Exception:
            pass
        
        session = await create_session(
            db,
            user_id=user_id,
            client=client or "web",
        )
        
        return success({
            "session_id": str(session.id),
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_active_at": session.last_active_at.isoformat(),
        })
    
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return error("Failed to create session", 500)


# ─────────────────────────────────────────────────────────────
# CHAT HISTORY ENDPOINT (CRITICAL FOR PERSISTENCE)
# ─────────────────────────────────────────────────────────────

@router.get("/history")
async def get_chat_history(
    payload: tuple = Depends(get_current_user_with_db),
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get chat history for logged-in user.
    
    CRITICAL: Called on app load to restore state.
    
    Returns:
        ALL chats with ALL messages for each chat.
        Empty list if no chats.
    
    Response:
        {
            success: true,
            data: {
                chats: [
                    {
                        id: UUID,
                        title: string,
                        messages: [{...}, {...}],
                        message_count: int,
                        created_at: ISO8601,
                        ...
                    }
                ],
                chat_count: int,
                total_messages: int
            },
            error: null
        }
    
    GUARANTEE:
        • Never returns null
        • Empty chats list if no chats (not null)
        • All messages included for each chat
        • Last update shows most recent first
    """
    try:
        _, user_id, db = payload
        
        # PATCH 5: History validation logging
        logger.info(f"HISTORY REQUEST USER_ID: {user_id}")
        print(f"BACKEND HISTORY USER_ID: {user_id}")
        
        # Get all chats (non-archived, filtered by user_id)
        chats = await list_user_chats(db, user_id, limit=limit, archived=False)
        
        # Get messages for each chat
        messages_by_chat = {}
        for chat in chats:
            messages = await get_chat_messages(db, chat.id)
            messages_by_chat[str(chat.id)] = [message_to_dict(m) for m in messages]
        
        # Build response
        data = chat_history_response(chats, messages_by_chat)
        
        logger.info(f"History loaded: user={user_id} chats={data['chat_count']}")
        return success(data)
    
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        # Return empty history, not error (so frontend doesn't crash)
        return success(empty_history_structure())


# ─────────────────────────────────────────────────────────────
# CHAT ENDPOINTS
# ─────────────────────────────────────────────────────────────

@router.post("/chat")
async def create_new_chat(
    title: Optional[str] = None,
    mode: Optional[str] = None,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Create new chat for user.
    
    Request:
        POST /api/chat
        {
            title?: string,
            mode?: "conversational" | "forensic" | "experimental"
        }
    
    Response:
        {
            success: true,
            data: {
                id: UUID,
                user_id: string,
                title: string,
                mode: string,
                messages: [],
                created_at: ISO8601
            },
            error: null
        }
    """
    try:
        _, user_id, db = payload
        
        chat = await create_chat(
            db,
            user_id=user_id,
            title=title or "New Chat",
            mode=mode or "conversational",
        )
        
        data = chat_to_dict(chat, messages=[])
        logger.info(f"Chat created: {chat.id} for user {user_id}")
        return success(data)
    
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
        return error("Failed to create chat", 500)


@router.get("/chat/{chat_id}")
async def get_chat_detail(
    chat_id: str,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Get chat with all messages.
    
    CRITICAL: Called on page refresh to restore chat state.
    
    Response:
        {
            success: true,
            data: {
                id: UUID,
                title: string,
                messages: [
                    {id, role, content, created_at, ...},
                    ...
                ],
                created_at: ISO8601
            },
            error: null
        }
    
    GUARANTEE:
        • Returns chat even if no messages yet
        • Messages array is always present (never null)
        • User can only access their own chats
    """
    try:
        _, user_id, db = payload
        
        # Convert string to UUID
        try:
            chat_uuid = UUID(chat_id)
        except ValueError:
            return error("Invalid chat ID format", 400)
        
        chat = await get_chat(db, chat_uuid)
        if not chat:
            return error("Chat not found", 404)
        
        # Verify ownership
        if str(chat.user_id) != str(user_id):
            return error("Access denied", 403)
        
        # Get messages
        messages = await get_chat_messages(db, chat_uuid)
        message_dicts = [message_to_dict(m) for m in messages]
        
        data = chat_to_dict(chat, messages=message_dicts)
        return success(data)
    
    except Exception as e:
        logger.error(f"Error loading chat: {e}")
        return error("Failed to load chat", 500)


class ChatUpdatePayload(BaseModel):
    title: Optional[str] = None
    mode: Optional[str] = None
    machine_metadata: Optional[Dict[str, Any]] = None

@router.put("/chat/{chat_id}")
async def update_chat_detail(
    chat_id: str,
    payload: ChatUpdatePayload,
    auth: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """Update chat metadata."""
    try:
        _, user_id, db = auth
        
        chat_uuid = UUID(chat_id)
        chat = await get_chat(db, chat_uuid)
        if not chat or str(chat.user_id) != str(user_id):
            return error("Chat not found or access denied", 404)
        
        await update_chat_metadata(
            db, 
            chat_uuid, 
            title=payload.title, 
            mode=payload.mode, 
            machine_metadata=payload.machine_metadata
        )
        
        chat = await get_chat(db, chat_uuid) # refresh
        messages = await get_chat_messages(db, chat_uuid)
        data = chat_to_dict(chat, messages=[message_to_dict(m) for m in messages])
        return success(data)
    
    except Exception as e:
        logger.error(f"Error updating chat: {e}")
        return error("Failed to update chat", 500)


# ─────────────────────────────────────────────────────────────
# MESSAGE ENDPOINTS
# ─────────────────────────────────────────────────────────────

@router.post("/chat/{chat_id}/message")
async def add_chat_message(
    chat_id: str,
    role: str,
    content: str,
    reasoning_json: Optional[Dict[str, Any]] = None,
    image_url: Optional[str] = None,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Add message to chat.
    
    Request:
        POST /api/chat/{id}/message
        {
            role: "user" | "assistant",
            content: string,
            reasoning_json?: object,
            image_url?: string
        }
    
    Response:
        {
            success: true,
            data: {
                id: UUID,
                chat_id: UUID,
                role: string,
                content: string,
                created_at: ISO8601
            },
            error: null
        }
    """
    try:
        _, user_id, db = payload
        
        # Verify chat exists and belongs to user
        chat_uuid = UUID(chat_id)
        chat = await get_chat(db, chat_uuid)
        if not chat or str(chat.user_id) != str(user_id):
            return error("Chat not found or access denied", 404)
        
        # Add message (transactionally)
        message = await add_message(
            db,
            chat_id=chat_uuid,
            user_id=user_id,
            role=role,
            content=content,
            reasoning_json=reasoning_json,
            image_url=image_url,
        )
        
        data = message_to_dict(message)
        logger.info(f"Message added: {message.id} to chat {chat_id}")
        return success(data)
    
    except ValueError as e:
        return error(str(e), 400)
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        return error("Failed to add message", 500)


@router.get("/chat/{chat_id}/messages")
async def get_chat_messages_endpoint(
    chat_id: str,
    limit: int = 100,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """Get messages for chat."""
    import time
    start_time = time.time()
    logger.info(f"START get_chat_messages_endpoint {chat_id}")
    try:
        _, user_id, db = payload
        
        chat_uuid = UUID(chat_id)
        chat = await get_chat(db, chat_uuid)
        if not chat or str(chat.user_id) != str(user_id):
            return error("Chat not found or access denied", 404)
        
        logger.info("QUERY START")
        messages = await get_chat_messages(db, chat_uuid, limit=limit)
        logger.info(f"QUERY END - elapsed: {time.time() - start_time:.3f}s")
        
        logger.info("SERIALIZE START")
        message_dicts = [message_to_dict(m) for m in messages]
        logger.info(f"SERIALIZE END - elapsed: {time.time() - start_time:.3f}s")
        
        logger.info(f"REQUEST COMPLETE - elapsed: {time.time() - start_time:.3f}s")
        return success({
            "messages": message_dicts,
            "count": len(message_dicts),
        })
    
    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        return error("Failed to load messages", 500)


# ─────────────────────────────────────────────────────────────
# MEMORY ENDPOINTS
# ─────────────────────────────────────────────────────────────

@router.get("/memory")
async def get_user_memory_endpoint(
    limit: int = 100,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Get all learned facts for user.
    
    Response:
        {
            success: true,
            data: {
                entries: [
                    {key, value, weight, confidence, ...},
                    ...
                ],
                entry_count: int
            },
            error: null
        }
    """
    try:
        _, user_id, db = payload
        
        memory_entries = await get_user_memory(db, user_id, limit=limit)
        memory_dicts = [memory_to_dict(m) for m in memory_entries]
        
        return success({
            "entries": memory_dicts,
            "entry_count": len(memory_dicts),
        })
    
    except Exception as e:
        logger.error(f"Error loading memory: {e}")
        return success(empty_memory_structure())  # Return empty, not error


@router.post("/memory")
async def upsert_user_memory(
    key: str,
    value: Dict[str, Any],
    weight: Optional[float] = None,
    tag: Optional[str] = None,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Upsert learned fact.
    
    Idempotent: same (key, value) → updates weight and timestamp.
    
    Request:
        POST /api/memory
        {
            key: "preferred_model" | "writing_style" | etc,
            value: {...any JSON...},
            weight?: float (optional, incremented on upsert),
            tag?: string
        }
    """
    try:
        _, user_id, db = payload
        
        memory = await upsert_memory(
            db,
            user_id=user_id,
            key=key,
            value=value,
            weight=weight or 1.0,
            tag=tag,
        )
        
        return success(memory_to_dict(memory))
    
    except Exception as e:
        logger.error(f"Error upserting memory: {e}")
        return error("Failed to save memory", 500)


# ─────────────────────────────────────────────────────────────
# USER SETTINGS ENDPOINTS
# ─────────────────────────────────────────────────────────────
# SETTINGS & PREFERENCES
# ─────────────────────────────────────────────────────────────

SETTINGS_SCHEMA: Dict[str, Any] = {
    "theme": {"type": str, "allowed": ["dark", "light", "system"], "default": "dark"},
    "theme_preference": {"type": str, "allowed": ["dark", "light", "system"], "default": "dark"},
    "language": {"type": str, "allowed": ["en", "es", "fr", "de", "zh"], "default": "en"},
    "response_style": {"type": str, "allowed": ["concise", "balanced", "detailed"], "default": "balanced"},
    "default_mode": {"type": str, "allowed": ["standard", "debate", "evidence", "glass", "synthesis"], "default": "standard"},
    "default_model": {"type": str, "default": "llama-3-3-70b"},
    "runtime_preference": {"type": str, "allowed": ["standard", "pro"], "default": "standard"},
    "favorite_model": {"type": str, "default": "llama-3-3-70b"},
    "favorite_mode": {"type": str, "default": "standard"},
    "notifications_enabled": {"type": bool, "default": True},
    "telemetry_opt_in": {"type": bool, "default": True},
    "analytics_opt_in": {"type": bool, "default": True},
    "feedback_opt_in": {"type": bool, "default": True},
    "debate_rounds": {"type": int, "min": 1, "max": 10, "default": 3},
    "debate_depth": {"type": int, "min": 1, "max": 10, "default": 6},
    "auto_save": {"type": bool, "default": True},
    "display_name": {"type": str, "default": ""},
    "avatar_url": {"type": str, "default": ""},
}

@router.get("/user/settings")
async def get_user_settings_endpoint(
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Get user interface preferences. Merges with defaults.
    """
    try:
        _, user_id, db = payload
        
        db_settings = await get_user_settings(db, user_id)
        
        # Merge DB settings over schema defaults
        settings = {k: v.get("default") for k, v in SETTINGS_SCHEMA.items()}
        for k, v in db_settings.items():
            if k in settings:
                settings[k] = v
                
        return success({
            "settings": settings,
            "count": len(settings),
        })
    
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        # Return defaults on error
        default_settings = {k: v.get("default") for k, v in SETTINGS_SCHEMA.items()}
        return success({"settings": default_settings, "count": len(default_settings)})


@router.put("/user/settings")
async def update_user_settings_endpoint(
    settings: Dict[str, Any],
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Update user settings with schema validation.
    """
    try:
        _, user_id, db = payload
        
        validated = {}
        # Validate each incoming setting against schema
        for key, value in settings.items():
            if key not in SETTINGS_SCHEMA:
                return error(f"Invalid setting key: {key}", 400)
                
            schema = SETTINGS_SCHEMA[key]
            if value is None:
                value = schema.get("default")
                
            try:
                # Attempt type coercion if it's not the exact type
                if not isinstance(value, schema["type"]):
                    if schema["type"] is bool and isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes")
                    else:
                        value = schema["type"](value)
            except (ValueError, TypeError):
                return error(f"Invalid type for {key}. Expected {schema['type'].__name__}", 400)
                
            # Allowed values check
            if "allowed" in schema and value not in schema["allowed"]:
                return error(f"Invalid value for {key}. Allowed: {schema['allowed']}", 400)
                
            # Range check for ints
            if schema["type"] is int:
                if "min" in schema and value < schema["min"]:
                    return error(f"Value for {key} below minimum {schema['min']}", 400)
                if "max" in schema and value > schema["max"]:
                    return error(f"Value for {key} above maximum {schema['max']}", 400)
                    
            validated[key] = value
        
        # Upsert validated settings
        for key, value in validated.items():
            await upsert_user_setting(db, user_id=user_id, key=key, value=value)
        
        # Fetch updated and merged settings
        db_settings = await get_user_settings(db, user_id)
        merged_settings = {k: v.get("default") for k, v in SETTINGS_SCHEMA.items()}
        for k, v in db_settings.items():
            if k in merged_settings:
                merged_settings[k] = v
                
        return success({
            "settings": merged_settings,
            "count": len(merged_settings),
        })
    
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return error("Failed to update settings", 500)


# ─────────────────────────────────────────────────────────────
# ANALYTICS ENDPOINTS
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel
class AnalyticsEventSchema(BaseModel):
    event_type: str
    event_data: Optional[Dict[str, Any]] = None

@router.post("/analytics/events")
async def log_analytics_event(
    event: AnalyticsEventSchema,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Log an analytics event.
    """
    import time
    start_time = time.time()
    logger.info(f"START log_analytics_event {event.event_type}")
    try:
        _, user_id, db = payload
        logger.info("QUERY START")
        await insert_analytics_event(db, user_id=user_id, event_type=event.event_type, event_data=event.event_data)
        logger.info(f"QUERY END - elapsed: {time.time() - start_time:.3f}s")
        logger.info(f"REQUEST COMPLETE - elapsed: {time.time() - start_time:.3f}s")
        return success({"status": "logged"})
    except Exception as e:
        logger.error(f"Error logging analytics event: {e}")
        return error("Failed to log event", 500)


# ─────────────────────────────────────────────────────────────
# CONTEXT WINDOW (FOR LLM)
# ─────────────────────────────────────────────────────────────

@router.get("/context")
async def get_context_window(
    chat_id: str,
    max_messages: int = 10,
    max_tokens: int = 2048,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Build deterministic context window for LLM.
    
    Combines:
      • Recent messages (recency)
      • User memory (relevance)
      • Enforce token limit
    
    Response:
        {
            success: true,
            data: {
                messages: [{role, content, ...}],
                memory: [{key, value, weight, ...}],
                token_estimate: int
            },
            error: null
        }
    """
    try:
        _, user_id, db = payload
        
        chat_uuid = UUID(chat_id)
        chat = await get_chat(db, chat_uuid)
        if not chat or str(chat.user_id) != str(user_id):
            return error("Chat not found", 404)
        
        # Get recent messages
        messages = await get_chat_messages(db, chat_uuid, limit=max_messages)
        message_dicts = [message_to_dict(m) for m in messages]
        
        # Get user memory
        memory_entries = await get_user_memory(db, user_id, limit=20)
        memory_dicts = [memory_to_dict(m) for m in memory_entries]
        
        # Build context (enforce token limit in client or LLM)
        data = context_window_response(message_dicts, memory_dicts)
        
        return success(data)
    
    except Exception as e:
        logger.error(f"Error building context: {e}")
        return error("Failed to build context", 500)


# ─────────────────────────────────────────────────────────────
# USER ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.get("/user")
async def get_current_user_info(
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Get current user info.
    
    Response:
        {
            success: true,
            data: {
                id: string,
                email: string,
                name: string,
                stats: {chat_count, message_count, memory_count}
            },
            error: null
        }
    """
    try:
        user, user_id, db = payload
        
        user_obj = await get_user_by_id(db, user_id)
        stats = await get_user_stats(db, user_id)
        
        return success({
            "user": user_to_dict(user_obj),
            "stats": stats,
        })
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return error("Failed to get user info", 500)


@router.get("/user/search")
async def search_history(
    q: str,
    limit: int = 50,
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Search user's history for messages, artifacts, modes, and models.
    """
    try:
        from database.crud import search_user_history
        _, user_id, db = payload
        
        if not q or len(q.strip()) < 2:
            return error("Query too short", 400)
            
        results = await search_user_history(db, user_id, q.strip(), limit)
        return success({"results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Error searching history: {e}")
        return error("Failed to search history", 500)


@router.get("/user/analytics")
async def get_analytics(
    payload: tuple = Depends(get_current_user_with_db),
) -> Dict[str, Any]:
    """
    Aggregate session, message, and mode usage statistics for the user.
    """
    try:
        from database.models import Chat, Message
        from sqlalchemy.future import select
        from collections import Counter
        
        _, user_id, db = payload
        
        # Fetch all user chats to aggregate
        chats_result = await db.execute(select(Chat).where(Chat.user_id == user_id))
        chats = chats_result.scalars().all()
        
        # Fetch all messages to get total count
        msgs_result = await db.execute(select(Message).where(Message.user_id == user_id))
        messages = msgs_result.scalars().all()
        
        mode_usage = Counter()
        model_usage = Counter()
        
        for chat in chats:
            mode = chat.mode or "conversational"
            mode_usage[mode] += 1
            
            if chat.machine_metadata:
                model = chat.machine_metadata.get("winning_model")
                if model:
                    model_usage[model] += 1
                
                # Also capture sub_mode usage as part of mode usage if desired, but mode is enough
                sub_mode = chat.machine_metadata.get("sub_mode")
                if sub_mode:
                    mode_usage[f"{mode}:{sub_mode}"] += 1
        
        return success({
            "total_sessions": len(chats),
            "total_messages": len(messages),
            "mode_usage": dict(mode_usage),
            "model_usage": dict(model_usage),
        })
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return error("Failed to fetch analytics", 500)

