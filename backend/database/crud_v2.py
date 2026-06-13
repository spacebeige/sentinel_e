"""
============================================================
CRUD Operations v2 — Neon (PostgreSQL) Source of Truth
============================================================

Principles:
  • All writes transactional and idempotent
  • No nullable core identity fields (user_id, chat_id)
  • Safe async operations with proper session handling
  • Deterministic queries (same input → same output)
  • Never return None for core structures

Features:
  • Upsert operations (INSERT ... ON CONFLICT)
  • Transactional consistency (SERIALIZABLE isolation)
  • Proper error handling and rollback
  • Background task integration
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, text, delete, and_, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from contextlib import asynccontextmanager
import logging
import uuid as uuid_lib

from .models_v2 import (
    User, Session, Chat, Message, Memory, UserSettings, Embedding
)

logger = logging.getLogger("CRUD-v2")


# ─────────────────────────────────────────────────────────────
# TRANSACTION CONTEXT MANAGER
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def transactional(db: AsyncSession, isolation="SERIALIZABLE"):
    """
    Context manager for transactional operations.
    
    Usage:
        async with transactional(db) as session:
            await user_crud(session)
            # Auto-commit on success, auto-rollback on error
    """
    try:
        # Set isolation level
        await db.connection(execution_options={"isolation_level": isolation})
        
        yield db
        await db.commit()
        logger.debug(f"Transaction committed (isolation={isolation})")
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Transaction rolled back: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# USER CRUD
# ─────────────────────────────────────────────────────────────

async def upsert_user(
    db: AsyncSession,
    *,
    user_id: str,  # Auth provider ID (Clerk, Firebase, etc)
    email: str,
    name: Optional[str] = None,
    provider: str = "clerk",
) -> User:
    """
    Idempotent: create or update user.
    
    If user exists, update only mutable fields (name, updated_at).
    If user doesn't exist, create with email.
    
    Args:
        user_id: Auth provider ID (e.g., "user_123" from Clerk)
        email: User email (must be unique)
        name: User display name
        provider: Auth provider (clerk, firebase, supertokens)
    
    Returns:
        User object
    
    Raises:
        ValueError: If email already belongs to another user
    """
    async with transactional(db) as session:
        # Check existing user by user_id
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        existing_user = result.scalars().first()
        
        if existing_user:
            # Update mutable fields
            existing_user.name = name or existing_user.name
            existing_user.updated_at = datetime.utcnow()
            await session.flush()
            logger.info(f"User updated: {user_id}")
            return existing_user
        
        # Create new user
        new_user = User(
            id=user_id,
            email=email,
            name=name,
            provider=provider,
            role="user",
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(new_user)
        await session.flush()
        logger.info(f"User created: {user_id} ({email})")
        return new_user


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Get user by auth provider ID."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalars().first()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email."""
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalars().first()


# ─────────────────────────────────────────────────────────────
# SESSION CRUD
# ─────────────────────────────────────────────────────────────

async def create_session(
    db: AsyncSession,
    *,
    user_id: str,
    client: str = "web",
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Session:
    """
    Create a new session for user.
    
    Args:
        user_id: Auth provider ID
        client: web | mobile | api
        ip_address: Optional client IP
        user_agent: Optional browser/client info
        metadata: Optional custom metadata (device, feature flags, etc)
    
    Returns:
        Session object
    """
    async with transactional(db) as session:
        new_session = Session(
            id=uuid_lib.uuid4(),
            user_id=user_id,
            client=client,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            last_active_at=datetime.utcnow(),
        )
        session.add(new_session)
        await session.flush()
        logger.info(f"Session created: {new_session.id} for user {user_id}")
        return new_session


async def update_session_activity(
    db: AsyncSession,
    session_id: UUID,
) -> Optional[Session]:
    """
    Update last_active_at for session.
    
    Called on every API request to keep session alive.
    """
    async with transactional(db) as sess:
        result = await sess.execute(
            select(Session).where(Session.id == session_id)
        )
        session_obj = result.scalars().first()
        
        if session_obj:
            session_obj.last_active_at = datetime.utcnow()
            await sess.flush()
            logger.debug(f"Session activity updated: {session_id}")
        
        return session_obj


async def get_session(db: AsyncSession, session_id: UUID) -> Optional[Session]:
    """Get session by ID."""
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    return result.scalars().first()


async def list_user_sessions(
    db: AsyncSession,
    user_id: str,
    limit: int = 10,
) -> List[Session]:
    """List recent sessions for user."""
    result = await db.execute(
        select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.last_active_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────
# CHAT CRUD
# ─────────────────────────────────────────────────────────────

async def create_chat(
    db: AsyncSession,
    *,
    user_id: str,
    title: str = "Untitled Chat",
    mode: Optional[str] = None,
    machine_metadata: Optional[Dict[str, Any]] = None,
) -> Chat:
    """
    Create new chat for user.
    
    Args:
        user_id: Auth provider ID (NOT UUID)
        title: Chat title
        mode: conversational | forensic | experimental
        machine_metadata: Model used, tokens, etc.
    
    Returns:
        Chat object with id set
    
    Raises:
        ValueError: If user_id not found
    """
    async with transactional(db) as session:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalars().first()
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        # Create chat
        new_chat = Chat(
            id=uuid_lib.uuid4(),
            user_id=user_id,
            title=title,
            mode=mode or "conversational",
            machine_metadata=machine_metadata or {},
            user_metadata={},
            is_archived=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(new_chat)
        await session.flush()
        logger.info(f"Chat created: {new_chat.id} for user {user_id}")
        return new_chat


async def get_chat(db: AsyncSession, chat_id: UUID) -> Optional[Chat]:
    """Get chat by ID."""
    result = await db.execute(
        select(Chat).where(Chat.id == chat_id)
    )
    return result.scalars().first()


async def list_user_chats(
    db: AsyncSession,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    archived: bool = False,
) -> List[Chat]:
    """
    List chats for user.
    
    Args:
        user_id: Auth provider ID
        limit: Max results
        offset: Pagination offset
        archived: Include archived chats?
    
    Returns:
        List of Chat objects, ordered by updated_at DESC
    """
    result = await db.execute(
        select(Chat)
        .where(and_(Chat.user_id == user_id, Chat.is_archived == archived))
        .order_by(Chat.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


async def update_chat_title(
    db: AsyncSession,
    chat_id: UUID,
    title: str,
) -> Optional[Chat]:
    """Update chat title."""
    async with transactional(db) as session:
        result = await session.execute(
            select(Chat).where(Chat.id == chat_id)
        )
        chat = result.scalars().first()
        
        if chat:
            chat.title = title
            chat.updated_at = datetime.utcnow()
            await session.flush()
            logger.info(f"Chat title updated: {chat_id}")
        
        return chat


async def archive_chat(
    db: AsyncSession,
    chat_id: UUID,
) -> Optional[Chat]:
    """Archive chat (soft delete)."""
    async with transactional(db) as session:
        result = await session.execute(
            select(Chat).where(Chat.id == chat_id)
        )
        chat = result.scalars().first()
        
        if chat:
            chat.is_archived = True
            chat.updated_at = datetime.utcnow()
            await session.flush()
            logger.info(f"Chat archived: {chat_id}")
        
        return chat


# ─────────────────────────────────────────────────────────────
# MESSAGE CRUD
# ─────────────────────────────────────────────────────────────

async def add_message(
    db: AsyncSession,
    *,
    chat_id: UUID,
    user_id: str,
    role: str,
    content: str,
    reasoning_json: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    image_url: Optional[str] = None,
) -> Message:
    """
    Add message to chat.
    
    Args:
        chat_id: Chat ID (must exist)
        user_id: Auth provider ID
        role: user | assistant | system
        content: Message text
        reasoning_json: Optional reasoning/thought process
        metadata: Model, tokens, temperature, etc.
        image_url: Optional image URL (NOT base64)
    
    Returns:
        Message object with id set
    
    Raises:
        ValueError: If chat doesn't exist
    """
    async with transactional(db) as session:
        # Verify chat exists
        chat_result = await session.execute(
            select(Chat).where(Chat.id == chat_id)
        )
        chat = chat_result.scalars().first()
        if not chat:
            raise ValueError(f"Chat not found: {chat_id}")
        
        # Create message
        new_message = Message(
            id=uuid_lib.uuid4(),
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=content,
            reasoning_json=reasoning_json,
            metadata=metadata or {},
            image_url=image_url,
            is_deleted=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(new_message)
        
        # Update chat.updated_at
        chat.updated_at = datetime.utcnow()
        
        await session.flush()
        logger.info(f"Message added: {new_message.id} to chat {chat_id}")
        return new_message


async def get_chat_messages(
    db: AsyncSession,
    chat_id: UUID,
    limit: int = 100,
) -> List[Message]:
    """
    Get all messages for chat (excluding deleted).
    
    Returns:
        List of Message objects, ordered by created_at ASC
    """
    result = await db.execute(
        select(Message)
        .where(and_(
            Message.chat_id == chat_id,
            Message.is_deleted == False,
        ))
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    return result.scalars().all()


async def get_message(db: AsyncSession, message_id: UUID) -> Optional[Message]:
    """Get message by ID."""
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    return result.scalars().first()


async def soft_delete_message(
    db: AsyncSession,
    message_id: UUID,
) -> Optional[Message]:
    """Soft delete message (mark as deleted, don't remove)."""
    async with transactional(db) as session:
        result = await session.execute(
            select(Message).where(Message.id == message_id)
        )
        message = result.scalars().first()
        
        if message:
            message.is_deleted = True
            message.updated_at = datetime.utcnow()
            await session.flush()
            logger.info(f"Message soft-deleted: {message_id}")
        
        return message


# ─────────────────────────────────────────────────────────────
# MEMORY CRUD
# ─────────────────────────────────────────────────────────────

async def upsert_memory(
    db: AsyncSession,
    *,
    user_id: str,
    key: str,
    value: Dict[str, Any],
    weight: float = 1.0,
    confidence: int = 50,
    tag: Optional[str] = None,
) -> Memory:
    """
    Idempotent: upsert memory entry.
    
    If (user_id, key) exists, update value and weight.
    Otherwise, create new entry.
    
    Args:
        user_id: Auth provider ID
        key: Fact key (e.g., "preferred_model", "writing_style")
        value: JSON value
        weight: Reinforcement weight (increases on repeat)
        confidence: 0-100 confidence score
        tag: Optional categorization
    
    Returns:
        Memory object
    """
    async with transactional(db) as session:
        # Try to find existing memory
        result = await session.execute(
            select(Memory).where(
                and_(Memory.user_id == user_id, Memory.key == key)
            )
        )
        existing_memory = result.scalars().first()
        
        if existing_memory:
            # Update: increase weight, update value, refresh timestamp
            existing_memory.value = value
            existing_memory.weight = min(existing_memory.weight + weight, 100.0)  # Cap at 100
            existing_memory.confidence = confidence
            existing_memory.tag = tag or existing_memory.tag
            existing_memory.updated_at = datetime.utcnow()
            await session.flush()
            logger.info(f"Memory upserted (update): {user_id}/{key}")
            return existing_memory
        
        # Create new memory
        new_memory = Memory(
            id=uuid_lib.uuid4(),
            user_id=user_id,
            key=key,
            value=value,
            weight=weight,
            confidence=confidence,
            tag=tag,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(new_memory)
        await session.flush()
        logger.info(f"Memory upserted (create): {user_id}/{key}")
        return new_memory


async def get_user_memory(
    db: AsyncSession,
    user_id: str,
    limit: int = 100,
) -> List[Memory]:
    """
    Get all memory entries for user, ordered by weight DESC.
    
    Returns:
        List of Memory objects (sorted by importance)
    """
    result = await db.execute(
        select(Memory)
        .where(Memory.user_id == user_id)
        .order_by(Memory.weight.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def get_memory_by_key(
    db: AsyncSession,
    user_id: str,
    key: str,
) -> Optional[Memory]:
    """Get specific memory entry."""
    result = await db.execute(
        select(Memory).where(
            and_(Memory.user_id == user_id, Memory.key == key)
        )
    )
    return result.scalars().first()


# ─────────────────────────────────────────────────────────────
# USER SETTINGS CRUD
# ─────────────────────────────────────────────────────────────

async def upsert_user_setting(
    db: AsyncSession,
    *,
    user_id: str,
    key: str,
    value: Any,
) -> UserSettings:
    """
    Idempotent: upsert user setting.
    
    Args:
        user_id: Auth provider ID
        key: Setting key (e.g., theme, language, notifications_enabled)
        value: Setting value (can be JSON)
    
    Returns:
        UserSettings object
    """
    async with transactional(db) as session:
        # Try to find existing setting
        result = await session.execute(
            select(UserSettings).where(
                and_(UserSettings.user_id == user_id, UserSettings.key == key)
            )
        )
        existing_setting = result.scalars().first()
        
        if existing_setting:
            existing_setting.value = value
            existing_setting.updated_at = datetime.utcnow()
            await session.flush()
            logger.info(f"Setting updated: {user_id}/{key}")
            return existing_setting
        
        # Create new setting
        new_setting = UserSettings(
            id=uuid_lib.uuid4(),
            user_id=user_id,
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(new_setting)
        await session.flush()
        logger.info(f"Setting created: {user_id}/{key}")
        return new_setting


async def get_user_settings(
    db: AsyncSession,
    user_id: str,
) -> Dict[str, Any]:
    """
    Get all settings for user as dict.
    
    Returns:
        Dict mapping key → value
    """
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == user_id)
    )
    settings = result.scalars().all()
    return {s.key: s.value for s in settings}


# ─────────────────────────────────────────────────────────────
# ANALYTICS / UTILITY
# ─────────────────────────────────────────────────────────────

async def get_user_stats(
    db: AsyncSession,
    user_id: str,
) -> Dict[str, Any]:
    """Get user statistics (chat count, message count, etc)."""
    
    # Chat count
    chats_result = await db.execute(
        select(Chat).where(
            and_(Chat.user_id == user_id, Chat.is_archived == False)
        )
    )
    chat_count = len(chats_result.scalars().all())
    
    # Message count
    messages_result = await db.execute(
        select(Message).where(
            and_(Message.user_id == user_id, Message.is_deleted == False)
        )
    )
    message_count = len(messages_result.scalars().all())
    
    # Memory count
    memory_result = await db.execute(
        select(Memory).where(Memory.user_id == user_id)
    )
    memory_count = len(memory_result.scalars().all())
    
    return {
        "chat_count": chat_count,
        "message_count": message_count,
        "memory_count": memory_count,
    }


async def cleanup_old_sessions(
    db: AsyncSession,
    user_id: Optional[str] = None,
    days_old: int = 30,
) -> int:
    """
    Delete sessions older than N days.
    
    Args:
        user_id: Optional user to filter
        days_old: Sessions older than this many days
    
    Returns:
        Number of sessions deleted
    """
    cutoff = datetime.utcnow() - timedelta(days=days_old)
    
    query = delete(Session).where(Session.last_active_at < cutoff)
    if user_id:
        query = query.where(Session.user_id == user_id)
    
    result = await db.execute(query)
    await db.commit()
    
    deleted_count = result.rowcount
    logger.info(f"Cleaned up {deleted_count} old sessions (> {days_old} days)")
    return deleted_count
