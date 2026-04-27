from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Any, Dict
from datetime import datetime
from uuid import UUID

class MessageSchema(BaseModel):
    id: UUID
    chat_id: UUID
    user_id: Optional[str] = None
    role: str
    content: str
    image_b64: Optional[str] = None
    image_mime: Optional[str] = None
    reasoning_json: Optional[Dict[str, Any]] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ChatSchema(BaseModel):
    id: UUID
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None
    chat_name: str
    mode: str
    created_at: datetime
    updated_at: datetime
    priority_answer: Optional[str] = None
    machine_metadata: Optional[Dict[str, Any]] = None
    shadow_metadata: Optional[Dict[str, Any]] = None
    rounds: int
    models_used: Optional[List[str]] = None

    model_config = ConfigDict(from_attributes=True)

class SessionMetaSchema(BaseModel):
    id: UUID
    user_id: str
    session_token: str
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    is_active: bool
    last_activity_at: datetime
    created_at: datetime
    expires_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class HistoryResponseSchema(BaseModel):
    chats: List[ChatSchema]
    messages: List[MessageSchema]
    metadata: List[SessionMetaSchema]

class UserMemorySchema(BaseModel):
    id: UUID
    user_id: str
    key: str
    value: str
    confidence: float
    weight: Optional[float] = 1.0
    last_used: Optional[datetime] = None
    recency_score: Optional[float] = 1.0
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime

    model_config = ConfigDict(from_attributes=True)

class UserPreferenceSchema(BaseModel):
    id: UUID
    user_id: str
    response_style: str
    tone: str
    default_chat_mode: str
    preferred_model: Optional[str] = None
    preferred_provider: Optional[str] = None
    dark_mode: bool
    show_reasoning: bool
    auto_save_chats: bool
    chat_retention_days: int
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ContextWindowSchema(BaseModel):
    id: UUID
    user_id: str
    chat_id: UUID
    context_json: Dict[str, Any]
    token_count: int
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
