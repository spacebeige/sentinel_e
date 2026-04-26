from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, text
from .models import Chat, Message, User
import json
import logging
import uuid

logger = logging.getLogger("CRUD")


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
    """
    Upsert an authenticated user without duplicating records.
    """
    normalized_email = email.strip().lower() if email else None
    normalized_name = name.strip() if isinstance(name, str) and name.strip() else None
    normalized_provider = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None

    # Check by clerk_user_id if provider is clerk, otherwise user_id
    existing = None
    if normalized_provider == "clerk":
        result = await db.execute(select(User).where(User.clerk_user_id == user_id))
        existing = result.scalars().first()
        
    if not existing:
        existing = await get_user_by_user_id(db, user_id)
        
    if not existing and normalized_email:
        existing = await get_user_by_email(db, normalized_email)
        
    if existing:
        if normalized_provider == "clerk":
            existing.clerk_user_id = user_id
        else:
            existing.user_id = user_id
            
        if normalized_email:
            existing.email = normalized_email
        if normalized_name:
            existing.name = normalized_name
        elif not existing.name and normalized_email:
            existing.name = normalized_email.split("@")[0]
        if normalized_provider:
            existing.provider = normalized_provider
        existing.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing)
        return existing

    new_user = User(
        user_id=user_id if normalized_provider != "clerk" else f"legacy_{user_id}",
        clerk_user_id=user_id if normalized_provider == "clerk" else None,
        email=normalized_email,
        name=normalized_name or (normalized_email.split("@")[0] if normalized_email else None),
        provider=normalized_provider,
        role="user",
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def create_chat(
    db: AsyncSession, 
    chat_name: str, 
    mode: str, 
    user_id: Optional[str] = None
) -> Chat:
    new_chat = Chat(
        chat_name=chat_name,
        mode=mode,
        user_id=user_id,
        rounds=0,
        models_used=[]
    )
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)
    return new_chat

async def get_chat(db: AsyncSession, chat_id: UUID, user_id: Optional[str] = None) -> Optional[Chat]:
    """Get a chat by ID. If user_id provided, verify ownership."""
    query = select(Chat).where(Chat.id == chat_id)
    if user_id:
        # ✅ Verify user owns this chat
        query = query.where(Chat.user_id == user_id)
        logger.debug(f"Query: get_chat for {chat_id} by user {user_id}")
    result = await db.execute(query)
    return result.scalars().first()

async def list_chats(db: AsyncSession, user_id: str, limit: int = 50, offset: int = 0) -> List[Chat]:
    """List chats for a specific user."""
    # ✅ REQUIRED: user_id filter to prevent cross-user data exposure
    logger.debug(f"Query: list_chats for user {user_id}")
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
    priority_answer: str,
    machine_metadata: Dict[str, Any],
    shadow_metadata: Optional[Dict[str, Any]] = None,
    rounds: int = 0,
    models_used: List[str] = []
):
    stmt = (
        update(Chat)
        .where(Chat.id == chat_id)
        .values(
            priority_answer=priority_answer,
            machine_metadata=machine_metadata,
            shadow_metadata=shadow_metadata if shadow_metadata else Chat.shadow_metadata,
            rounds=rounds,
            models_used=models_used,
            updated_at=datetime.utcnow()
        )
    )
    # Re-fetch is safer for updating ORM objects but update stmt is faster
    # Let's perform direct update
    chat = await get_chat(db, chat_id)
    if chat:
        chat.priority_answer = priority_answer
        chat.machine_metadata = machine_metadata
        if shadow_metadata:
            chat.shadow_metadata = shadow_metadata
        chat.rounds = rounds
        chat.models_used = models_used
        await db.commit()
        await db.refresh(chat)
        return chat
    return None

async def add_message(
    db: AsyncSession,
    chat_id: UUID,
    role: str,
    content: str,
    image_b64: str = None,
    image_mime: str = None,
    reasoning_json: dict = None
) -> Message:
    message_id = uuid.uuid4()
    created_at = datetime.utcnow()

    try:
        new_message = Message(
            id=message_id,
            chat_id=chat_id,
            role=role,
            content=content,
            image_b64=image_b64,
            image_mime=image_mime,
            reasoning_json=reasoning_json,
            created_at=created_at,
        )
        db.add(new_message)
        await db.commit()
        await db.refresh(new_message)
        return new_message
    except Exception as exc:
        await db.rollback()
        err = str(exc).lower()
        missing_reasoning_col = (
            "reasoning_json" in err and (
                "does not exist" in err or "undefinedcolumn" in err
            )
        )

        if not missing_reasoning_col:
            logger.error("add_message failed: %s", exc, exc_info=True)
            raise

        logger.warning(
            "messages.reasoning_json missing in DB; retrying insert without reasoning_json column"
        )

        await db.execute(
            text(
                """
                INSERT INTO messages (id, chat_id, role, content, image_b64, image_mime, created_at)
                VALUES (:id, :chat_id, :role, :content, :image_b64, :image_mime, :created_at)
                """
            ),
            {
                "id": message_id,
                "chat_id": chat_id,
                "role": role,
                "content": content,
                "image_b64": image_b64,
                "image_mime": image_mime,
                "created_at": created_at,
            },
        )
        await db.commit()

        result = await db.execute(select(Message).where(Message.id == message_id))
        retried_message = result.scalars().first()
        if retried_message:
            return retried_message

        # Ultra-safe fallback if ORM model refresh fails for any reason
        return Message(
            id=message_id,
            chat_id=chat_id,
            role=role,
            content=content,
            image_b64=image_b64,
            image_mime=image_mime,
            created_at=created_at,
        )

async def get_chat_messages(db: AsyncSession, chat_id: UUID, user_id: Optional[str] = None) -> List[Message]:
    """Get messages for a chat. If user_id provided, verify ownership."""
    from sqlalchemy import and_
    
    # ✅ Verify chat ownership before returning messages
    if user_id:
        result = await db.execute(
            select(Message)
            .join(Chat, Message.chat_id == Chat.id)
            .where(
                and_(
                    Message.chat_id == chat_id,
                    Chat.user_id == user_id  # ✅ Ownership check
                )
            )
            .order_by(Message.created_at.asc())
        )
        logger.debug(f"Query: get_chat_messages for {chat_id} by user {user_id}")
    else:
        result = await db.execute(
            select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())
        )
        logger.debug(f"Query: get_chat_messages for {chat_id} (no user verification)")
    
    return result.scalars().all()


async def create_asset(db, session_id: str, file_type: str, base64_data: str = None,
                       file_path: str = None, summary: str = None,
                       original_filename: str = None, file_size_bytes: int = None):
    """Store an uploaded asset for a session."""
    from .models import UploadedAsset
    asset = UploadedAsset(
        session_id=session_id,
        file_type=file_type,
        base64_data=base64_data,
        file_path=file_path,
        summary=summary,
        original_filename=original_filename,
        file_size_bytes=file_size_bytes,
    )
    db.add(asset)
    await db.commit()
    await db.refresh(asset)
    return asset


async def get_session_assets(db, session_id: str):
    """Get all assets for a session."""
    from .models import UploadedAsset
    result = await db.execute(
        select(UploadedAsset)
        .where(UploadedAsset.session_id == session_id)
        .order_by(UploadedAsset.created_at)
    )
    return result.scalars().all()


async def update_asset_summary(db, asset_id: str, summary: str):
    """Update the vision summary for an asset."""
    from .models import UploadedAsset
    result = await db.execute(
        select(UploadedAsset).where(UploadedAsset.id == asset_id)
    )
    asset = result.scalar_one_or_none()
    if asset:
        asset.summary = summary
        await db.commit()
    return asset


async def update_message(db, message_id, new_content: str):
    """Edit a message's content."""
    from .models import Message
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    msg = result.scalar_one_or_none()
    if msg:
        msg.content = new_content
        await db.commit()
    return msg


async def delete_messages_after(db, chat_id, message_id):
    """Delete all messages after a given message (for regeneration).
    Returns the count of deleted messages."""
    from .models import Message
    # Get the target message to find its created_at
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    target = result.scalar_one_or_none()
    if not target:
        return 0

    # Delete all messages after this one
    from sqlalchemy import delete as sql_delete
    stmt = sql_delete(Message).where(
        Message.chat_id == chat_id,
        Message.created_at > target.created_at
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount
