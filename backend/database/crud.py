from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from .models import Chat, Message
import json

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

async def get_chat(db: AsyncSession, chat_id: UUID) -> Optional[Chat]:
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    return result.scalars().first()

async def list_chats(db: AsyncSession, limit: int = 50, offset: int = 0) -> List[Chat]:
    result = await db.execute(
        select(Chat).order_by(Chat.updated_at.desc()).limit(limit).offset(offset)
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
    image_mime: str = None
) -> Message:
    new_message = Message(
        chat_id=chat_id,
        role=role,
        content=content,
        image_b64=image_b64,
        image_mime=image_mime
    )
    db.add(new_message)
    await db.commit()
    await db.refresh(new_message)
    return new_message

async def get_chat_messages(db: AsyncSession, chat_id: UUID) -> List[Message]:
    result = await db.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())
    )
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
