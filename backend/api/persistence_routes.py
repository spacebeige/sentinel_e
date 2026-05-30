import logging
from typing import Optional, Dict, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from database.connection_v2 import get_db
from database.crud_v2 import (
    create_session, list_user_sessions,
    create_chat, list_user_chats, update_chat_title, archive_chat,
    add_message, get_chat_messages
)
from database.models_v2 import Chat
from gateway.auth_v2 import get_current_user, ensure_user_exists as ensure_user
from utils.api_response import api_success, api_error
from utils.safe_responses import (
    session_to_dict, chat_to_dict, message_to_dict
)

logger = logging.getLogger("Persistence-v2")

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# DEPENDENCY
# ─────────────────────────────────────────────────────────────
async def get_current_user_with_db(
    user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[Dict[str, Any], str, AsyncSession]:
    user_id = await ensure_user(user, db)
    return user, user_id, db

# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class SessionCreate(BaseModel):
    client: str = "web"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationCreate(BaseModel):
    title: str = "Untitled Chat"
    mode: Optional[str] = "conversational"
    engine: Optional[str] = None
    machine_metadata: Optional[Dict[str, Any]] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    mode: Optional[str] = None
    engine: Optional[str] = None
    archived: Optional[bool] = None

class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str
    reasoning_json: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None

# ─────────────────────────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────────────────────────
@router.post("/sessions")
async def api_create_session(
    payload: SessionCreate,
    req: Request,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        new_session = await create_session(
            db,
            user_id=user_id,
            client=payload.client,
            ip_address=payload.ip_address or req.client.host if req.client else None,
            user_agent=payload.user_agent or req.headers.get("user-agent"),
            metadata=payload.metadata
        )
        return api_success(session_to_dict(new_session), status_code=201)
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return api_error(str(e), status_code=500)

@router.get("/sessions")
async def api_get_sessions(
    limit: int = 10,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        sessions = await list_user_sessions(db, user_id, limit)
        return api_success([session_to_dict(s) for s in sessions])
    except Exception as e:
        return api_error(str(e), status_code=500)

# ─────────────────────────────────────────────────────────────
# CONVERSATIONS
# ─────────────────────────────────────────────────────────────
@router.post("/conversations")
async def api_create_conversation(
    payload: ConversationCreate,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        new_chat = await create_chat(
            db,
            user_id=user_id,
            title=payload.title,
            mode=payload.mode,
            machine_metadata=payload.machine_metadata
        )
        if payload.engine:
            new_chat.engine = payload.engine
            await db.commit()
        return api_success(chat_to_dict(new_chat), status_code=201)
    except Exception as e:
        return api_error(str(e), status_code=500)

@router.get("/conversations")
async def api_get_conversations(
    limit: int = 50,
    offset: int = 0,
    archived: bool = False,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        chats = await list_user_chats(db, user_id, limit, offset, archived)
        return api_success([chat_to_dict(c) for c in chats])
    except Exception as e:
        return api_error(str(e), status_code=500)

@router.get("/conversations/{conversation_id}")
async def api_get_conversation(
    conversation_id: str,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        cid = UUID(conversation_id)
        from sqlalchemy.future import select
        result = await db.execute(select(Chat).where(Chat.id == cid, Chat.user_id == user_id))
        chat = result.scalars().first()
        if not chat:
            return api_error("Conversation not found", status_code=404)
        return api_success(chat_to_dict(chat))
    except Exception as e:
        return api_error(str(e), status_code=500)

@router.patch("/conversations/{conversation_id}")
async def api_update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        cid = UUID(conversation_id)
        from sqlalchemy.future import select
        result = await db.execute(select(Chat).where(Chat.id == cid, Chat.user_id == user_id))
        chat = result.scalars().first()
        if not chat:
            return api_error("Conversation not found", status_code=404)

        if payload.title is not None:
            chat.title = payload.title
        if payload.mode is not None:
            chat.mode = payload.mode
        if payload.engine is not None:
            chat.engine = payload.engine
        if payload.archived is not None:
            chat.is_archived = payload.archived
            
        from datetime import datetime
        chat.updated_at = datetime.utcnow()
        await db.commit()
        return api_success(chat_to_dict(chat))
    except Exception as e:
        return api_error(str(e), status_code=500)

@router.delete("/conversations/{conversation_id}")
async def api_delete_conversation(
    conversation_id: str,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        cid = UUID(conversation_id)
        from sqlalchemy.future import select
        result = await db.execute(select(Chat).where(Chat.id == cid, Chat.user_id == user_id))
        if not result.scalars().first():
            return api_error("Conversation not found", status_code=404)
        
        await archive_chat(db, cid)
        return api_success(None)
    except Exception as e:
        return api_error(str(e), status_code=500)

# ─────────────────────────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────────────────────────
@router.post("/messages")
async def api_add_message(
    payload: MessageCreate,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        cid = UUID(payload.conversation_id)
        from sqlalchemy.future import select
        result = await db.execute(select(Chat).where(Chat.id == cid, Chat.user_id == user_id))
        if not result.scalars().first():
            return api_error("Conversation not found", status_code=404)

        new_msg = await add_message(
            db,
            chat_id=cid,
            user_id=user_id,
            role=payload.role,
            content=payload.content,
            reasoning_json=payload.reasoning_json,
            metadata=payload.metadata,
            image_url=payload.image_url
        )
        return api_success(message_to_dict(new_msg), status_code=201)
    except Exception as e:
        return api_error(str(e), status_code=500)

@router.get("/messages")
async def api_get_messages(
    conversation_id: str,
    limit: int = 100,
    auth_data: tuple = Depends(get_current_user_with_db)
):
    _, user_id, db = auth_data
    try:
        cid = UUID(conversation_id)
        from sqlalchemy.future import select
        result = await db.execute(select(Chat).where(Chat.id == cid, Chat.user_id == user_id))
        if not result.scalars().first():
            return api_error("Conversation not found", status_code=404)

        messages = await get_chat_messages(db, cid, limit)
        return api_success([message_to_dict(m) for m in messages])
    except Exception as e:
        return api_error(str(e), status_code=500)
