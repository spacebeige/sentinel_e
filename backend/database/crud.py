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
MAX_MESSAGE_CONTENT_LENGTH = 16000


async def _best_effort_index_message(message: Message):
    """Best-effort semantic indexing to Pinecone. Never raises."""
    try:
        from utils.vector_service import get_vector_service

        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            logger.info("vector status pinecone=skip op=upsert reason=empty_content")
            return

        vector_service = get_vector_service()
        embedding = await vector_service.get_embedding(content)
        if not embedding:
            logger.info("vector status pinecone=skip op=upsert reason=no_embedding")
            return

        metadata = {
            "chat_id": str(getattr(message, "chat_id", "")),
            "user_id": str(getattr(message, "user_id", "") or ""),
            "role": str(getattr(message, "role", "") or ""),
            "content": content[:2000],
            "created_at": getattr(message, "created_at", None).isoformat() if getattr(message, "created_at", None) else None,
        }

        await vector_service.upsert(
            namespace="chat_messages",
            items=[
                {
                    "id": str(getattr(message, "id", uuid.uuid4())),
                    "values": embedding,
                    "metadata": metadata,
                }
            ],
        )
        logger.info("vector status pinecone=success op=upsert chat_id=%s", metadata["chat_id"])
    except Exception as e:
        logger.warning("vector status pinecone=fail op=upsert error=%s", e)


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
    session_id: Optional[UUID] = None,
    chat_id: Optional[UUID] = None
) -> Chat:
    kwargs = {
        "chat_name": chat_name,
        "mode": mode,
        "user_id": user_id,
        "session_id": session_id,
        "rounds": 0,
        "models_used": []
    }
    if chat_id:
        kwargs["id"] = chat_id
    new_chat = Chat(**kwargs)
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


async def update_chat_metadata(
    db: AsyncSession,
    chat_id: UUID,
    **kwargs
) -> Optional[Chat]:
    """Generic chat metadata updater."""
    query = update(Chat).where(Chat.id == chat_id).values(**kwargs, updated_at=datetime.utcnow())
    await db.execute(query)
    await db.commit()
    return await get_chat(db, chat_id)


# ── Message CRUD ─────────────────────────────────────────────

async def add_message(
    db: AsyncSession,
    chat_id: UUID,
    user_id: str,
    role: str,
    content: str,
    image_b64: Optional[str] = None,
    image_mime: Optional[str] = None,
    reasoning_json: Optional[dict] = None,
    metadata_json: Optional[dict] = None
) -> Optional[Message]:
    """Insert message with safe fallback: retry minimal write on JSON-field failure."""
    # Ensure content is always a clean string
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    content = content[:MAX_MESSAGE_CONTENT_LENGTH]

    try:
        new_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=content,
            image_b64=image_b64,
            reasoning_json=reasoning_json,
            metadata_json=metadata_json,
        )
        db.add(new_message)
        await db.execute(
            update(Chat).where(Chat.id == chat_id).values(updated_at=datetime.utcnow())
        )
        await db.commit()
        await db.refresh(new_message)

        try:
            asyncio.create_task(_best_effort_index_message(new_message))
        except Exception as vector_err:
            logger.warning(f"vector status pinecone=fail op=schedule_upsert error={vector_err}")

        return new_message
    except Exception as full_err:
        logger.warning(f"add_message full write failed ({full_err}), retrying minimal write")
        try:
            await db.rollback()
        except Exception:
            pass
        # Minimal fallback: no JSON fields that could be malformed
        minimal_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=content[:4000],  # stricter fallback truncation
            image_b64=None,
            reasoning_json=None,
            metadata_json=None,
        )
        db.add(minimal_message)
        try:
            await db.execute(
                update(Chat).where(Chat.id == chat_id).values(updated_at=datetime.utcnow())
            )
        except Exception:
            pass
        try:
            await db.commit()
            await db.refresh(minimal_message)
            logger.info(f"add_message minimal fallback succeeded for chat {chat_id}")

            try:
                asyncio.create_task(_best_effort_index_message(minimal_message))
            except Exception as vector_err:
                logger.warning(f"vector status pinecone=fail op=schedule_upsert error={vector_err}")

            return minimal_message
        except Exception as minimal_err:
            logger.error(f"add_message minimal fallback failed for chat {chat_id}: {minimal_err}")
            try:
                await db.rollback()
            except Exception:
                pass
            return None


async def get_chat_messages(db: AsyncSession, chat_id: UUID, user_id: Optional[str] = None) -> List[Message]:
    query = select(Message).where(Message.chat_id == chat_id)
    if user_id:
        query = query.where(Message.user_id == user_id)
    
    result = await db.execute(query.order_by(Message.created_at.asc()))
    return result.scalars().all()


async def update_message(db: AsyncSession, message_id: UUID, new_content: str) -> Optional[Message]:
    await db.execute(
        update(Message).where(Message.id == message_id).values(content=new_content)
    )
    await db.commit()
    result = await db.execute(select(Message).where(Message.id == message_id))
    return result.scalars().first()


async def delete_messages_after(db: AsyncSession, chat_id: UUID, message_id: UUID) -> int:
    # Find the message first to get its created_at
    result = await db.execute(select(Message).where(Message.id == message_id))
    target = result.scalars().first()
    if not target:
        return 0
        
    delete_query = delete(Message).where(
        Message.chat_id == chat_id,
        Message.created_at > target.created_at
    )
    res = await db.execute(delete_query)
    await db.commit()
    return res.rowcount


# ── User Memory CRUD ─────────────────────────────────────────

async def add_user_memory(
    db: AsyncSession,
    user_id: str,
    key: str,
    value: str,
    confidence: int = 75,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> UserMemory:
    # Lightweight decay to keep memory adaptive over time.
    try:
        stale_result = await db.execute(
            select(UserMemory).where(UserMemory.user_id == user_id)
        )
        stale_rows = stale_result.scalars().all()
        for row in stale_rows:
            if row.key != key and row.confidence and row.confidence > 5:
                row.confidence = max(int(row.confidence) - 1, 1)
    except Exception:
        pass

    # Upsert logic
    result = await db.execute(
        select(UserMemory).where(UserMemory.user_id == user_id, UserMemory.key == key)
    )
    existing = result.scalars().first()
    
    if existing:
        existing.value = value
        # Reinforce repeated patterns while avoiding runaway growth.
        existing.confidence = min(max(int(existing.confidence or 0) + 1, confidence), 100)
        existing.weight = min(float(existing.weight or 1.0) + 1.0, 20.0)
        existing.last_used = datetime.utcnow()
        existing.recency_score = 1.0
        merged_meta = dict(existing.metadata_json or {})
        merged_meta.update(metadata_json or {})
        merged_meta["weight"] = existing.weight
        merged_meta["last_used"] = existing.last_used.isoformat()
        merged_meta["recency_score"] = existing.recency_score
        existing.metadata_json = merged_meta
        existing.updated_at = datetime.utcnow()
    else:
        meta = dict(metadata_json or {})
        meta["weight"] = 1.0
        meta["last_used"] = datetime.utcnow().isoformat()
        meta["recency_score"] = 1.0
        existing = UserMemory(
            user_id=user_id,
            key=key,
            value=value,
            confidence=confidence,
            weight=1.0,
            last_used=datetime.utcnow(),
            recency_score=1.0,
            metadata_json=meta
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


async def get_user_preference(db: AsyncSession, user_id: str) -> Dict[str, str]:
    """Alias for get_user_preferences to match main.py import."""
    return await get_user_preferences(db, user_id)


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
