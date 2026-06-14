import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, ARRAY, JSON, Boolean, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """User profiles with role management."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, unique=True, index=True, nullable=False)  # JWT sub claim
    clerk_user_id = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    name = Column(String, nullable=True)
    provider = Column(String, nullable=True)
    role = Column(String, default="user", nullable=False)  # "user" | "admin" | "moderator"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserSession(Base):
    """Session tracking."""
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Chat(Base):
    __tablename__ = "chats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False) 
    chat_name = Column(String, nullable=False)
    mode = Column(String, nullable=False) 
    
    is_archived = Column(Boolean, default=False, nullable=False)
    # Runtime complexity & priority metadata
    priority_answer = Column(Text, nullable=True)
    rounds = Column(Integer, server_default="1", nullable=True)

    machine_metadata = Column(JSONB, default=dict) 

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


    @property
    def session_id(self):
        return None

    @property
    def shadow_metadata(self):
        return None

    @property
    def models_used(self):
        return []

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    role = Column(String, nullable=False) # user | assistant | etc
    content = Column(Text, nullable=False)
    reasoning_json = Column(JSONB, nullable=True)
    image_b64 = Column(Text, nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserMemory(Base):
    """User-specific learned facts and knowledge."""
    __tablename__ = "user_memory"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Integer, default=50)  # 0-100 scale
    weight = Column(Float, default=1.0)
    last_used = Column(DateTime, default=datetime.utcnow)
    recency_score = Column(Float, default=1.0)
    metadata_json = Column(JSONB, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserPreference(Base):
    """User configuration & interface preferences."""
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class ContextWindow(Base):
    __tablename__ = "context_windows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    chat_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    context_json = Column(JSONB, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UploadedAsset(Base):
    __tablename__ = "uploaded_assets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    file_type = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    base64_data = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    original_filename = Column(String, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
