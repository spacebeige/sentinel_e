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
    content: str
) -> Message:
    new_message = Message(
        chat_id=chat_id,
        role=role,
        content=content
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
