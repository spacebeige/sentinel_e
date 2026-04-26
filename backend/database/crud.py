from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, text, delete
from .models import Chat, Message, User, UserMemory, UserPreference, ContextWindow, UserSession
import json
import logging
import uuid
import asyncio

logger = logging.getLogger("CRUD")


# ── User CRUD ────────────────────────────────────────────────

async def get_user_by_user_id(db: AsyncSession, user_id: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.user_id == user_id))
    return result.scalars().first()


async def get_user_by_email(db: AsyncSession, email: Optional[str]) -> Optional[User]:
    if not email:
        return None
    result = await db.execute(select(User).where(User.email == email))
    return result.scalars().first()


async def upsert_authenticated_user(
    db: AsyncSession,
    *,
    user_id: str,
    email: Optional[str] = None,
    name: Optional[str] = None,
    provider: Optional[str] = None,
) -> User:
    normalized_email = email.strip().lower() if email else None
    
    # Try finding existing user
    result = await db.execute(select(User).where(User.user_id == user_id))
    existing = result.scalars().first()
    
    if not existing and normalized_email:
        existing = await get_user_by_email(db, normalized_email)
    
    if existing:
        existing.user_id = user_id
        if normalized_email:
            existing.email = normalized_email
        if name:
            existing.name = name
        if provider:
            existing.provider = provider
        existing.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing)
        return existing

    new_user = User(
        user_id=user_id,
        email=normalized_email,
        name=name or (normalized_email.split("@")[0] if normalized_email else None),
        provider=provider,
        role="user",
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user


# ── Chat CRUD ────────────────────────────────────────────────

async def create_chat(
    db: AsyncSession, 
    chat_name: str, 
    mode: str, 
    user_id: str,
    session_id: Optional[UUID] = None
) -> Chat:
    new_chat = Chat(
        chat_name=chat_name,
        mode=mode,
        user_id=user_id,
        session_id=session_id,
        rounds=0,
        models_used=[]
    )
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)
    return new_chat


async def get_chat(db: AsyncSession, chat_id: UUID, user_id: Optional[str] = None) -> Optional[Chat]:
    query = select(Chat).where(Chat.id == chat_id)
    if user_id:
        query = query.where(Chat.user_id == user_id)
    
    result = await db.execute(query)
    return result.scalars().first()


async def list_chats(db: AsyncSession, user_id: str, limit: int = 50, offset: int = 0) -> List[Chat]:
    result = await db.execute(
        select(Chat)
        .where(Chat.user_id == user_id)
        .order_by(Chat.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


# ── Message CRUD ─────────────────────────────────────────────

async def add_message(
    db: AsyncSession,
    chat_id: UUID,
    user_id: str,
    role: str,
    content: str,
    image_b64: Optional[str] = None,
    reasoning_json: Optional[dict] = None,
    metadata_json: Optional[dict] = None
) -> Message:
    new_message = Message(
        chat_id=chat_id,
        user_id=user_id,
        role=role,
        content=content,
        image_b64=image_b64,
        reasoning_json=reasoning_json,
        metadata_json=metadata_json
    )
    db.add(new_message)
    
    # Update chat timestamp
    await db.execute(
        update(Chat).where(Chat.id == chat_id).values(updated_at=datetime.utcnow())
    )
    
    await db.commit()
    await db.refresh(new_message)
    return new_message


async def get_chat_messages(db: AsyncSession, chat_id: UUID, user_id: Optional[str] = None) -> List[Message]:
    query = select(Message).where(Message.chat_id == chat_id)
    if user_id:
        query = query.where(Message.user_id == user_id)
    
    result = await db.execute(query.order_by(Message.created_at.asc()))
    return result.scalars().all()


# ── User Memory CRUD ─────────────────────────────────────────

async def add_user_memory(
    db: AsyncSession,
    user_id: str,
    key: str,
    value: str,
    confidence: int = 75,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> UserMemory:
    # Upsert logic
    result = await db.execute(
        select(UserMemory).where(UserMemory.user_id == user_id, UserMemory.key == key)
    )
    existing = result.scalars().first()
    
    if existing:
        existing.value = value
        existing.confidence = confidence
        existing.metadata_json = metadata_json
        existing.updated_at = datetime.utcnow()
    else:
        existing = UserMemory(
            user_id=user_id,
            key=key,
            value=value,
            confidence=confidence,
            metadata_json=metadata_json
        )
        db.add(existing)
    
    await db.commit()
    await db.refresh(existing)
    return existing


async def get_user_memory(db: AsyncSession, user_id: str) -> List[UserMemory]:
    result = await db.execute(
        select(UserMemory).where(UserMemory.user_id == user_id).order_by(UserMemory.updated_at.desc())
    )
    return result.scalars().all()


# ── User Preference CRUD ─────────────────────────────────────

async def upsert_user_preference(
    db: AsyncSession,
    user_id: str,
    key: str,
    value: str
) -> UserPreference:
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == user_id, UserPreference.key == key)
    )
    existing = result.scalars().first()
    
    if existing:
        existing.value = value
        existing.updated_at = datetime.utcnow()
    else:
        existing = UserPreference(
            user_id=user_id,
            key=key,
            value=value
        )
        db.add(existing)
    
    await db.commit()
    await db.refresh(existing)
    return existing


async def get_user_preferences(db: AsyncSession, user_id: str) -> Dict[str, str]:
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == user_id)
    )
    prefs = result.scalars().all()
    return {p.key: p.value for p in prefs}


# ── Context Window CRUD ──────────────────────────────────────

async def upsert_context_window(
    db: AsyncSession,
    user_id: str,
    chat_id: UUID,
    context_json: Dict[str, Any]
) -> ContextWindow:
    result = await db.execute(
        select(ContextWindow).where(ContextWindow.user_id == user_id, ContextWindow.chat_id == chat_id)
    )
    existing = result.scalars().first()
    
    if existing:
        existing.context_json = context_json
        existing.updated_at = datetime.utcnow()
    else:
        existing = ContextWindow(
            user_id=user_id,
            chat_id=chat_id,
            context_json=context_json
        )
        db.add(existing)
    
    await db.commit()
    await db.refresh(existing)
    return existing


async def get_context_window(db: AsyncSession, user_id: str, chat_id: UUID) -> Optional[ContextWindow]:
    result = await db.execute(
        select(ContextWindow).where(ContextWindow.user_id == user_id, ContextWindow.chat_id == chat_id)
    )
    return result.scalars().first()
