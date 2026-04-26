import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, ARRAY, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base

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

class Chat(Base):
    __tablename__ = "chats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=True) # Optional for now
    chat_name = Column(String, nullable=False)
    mode = Column(String, nullable=False) # conversational | forensic | experimental
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    priority_answer = Column(Text, nullable=True)
    machine_metadata = Column(JSONB, nullable=True) # Full structured metadata
    shadow_metadata = Column(JSONB, nullable=True) # Shadow mode specific data
    
    rounds = Column(Integer, default=1)
    models_used = Column(ARRAY(String), nullable=True)

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    role = Column(String, nullable=False) # user | assistant | model_groq | model_qwen | model_llama70b
    content = Column(Text, nullable=False)
    image_b64 = Column(Text, nullable=True)  # Base64 image data
    image_mime = Column(String, nullable=True)  # MIME type (e.g. image/png)
    reasoning_json = Column(JSONB, nullable=True)  # Structured reasoning artifacts per assistant turn
    created_at = Column(DateTime, default=datetime.utcnow)

class UploadedAsset(Base):
    __tablename__ = "uploaded_assets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    file_type = Column(String, nullable=False)  # "image/png", "application/pdf", etc.
    file_path = Column(String, nullable=True)   # local/cloud path if stored on disk
    base64_data = Column(Text, nullable=True)   # base64-encoded file content
    summary = Column(Text, nullable=True)       # vision model's text description
    original_filename = Column(String, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────
# USER MEMORY GRAPH — Persistent Knowledge & Preferences
# ──────────────────────────────────────────────────────────────

class UserMemory(Base):
    """
    User-specific learned facts and knowledge.
    
    Stores contextual information learned from conversations:
      - Key facts about user preferences
      - Domain knowledge specific to user's needs
      - Learned behavioral patterns
      - Session insights
    
    Used by context window builder to personalize responses.
    """
    __tablename__ = "user_memory"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    
    # Knowledge key (e.g., "preferred_response_length", "domain_interest", "technical_level")
    key = Column(String, nullable=False)
    
    # Knowledge value (e.g., "concise", "machine_learning", "expert")
    value = Column(Text, nullable=False)
    
    # Confidence score (0.0 to 1.0) — higher = more reliable
    confidence = Column(Integer, default=50)  # 0-100 scale
    
    # Metadata (e.g., source_chat_id, source_query, reasoning)
    metadata_json = Column(JSONB, nullable=True)
    
    # Lifecycle tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserPreference(Base):
    """
    User configuration & interface preferences.
    
    Stores user-specific settings that persist across sessions:
      - Response style (concise, detailed, narrative)
      - Tone preference (formal, casual, technical)
      - Default chat mode
      - UI preferences
    """
    __tablename__ = "user_preference"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, unique=True, index=True, nullable=False)
    
    # Response preferences
    response_style = Column(String, default="balanced", nullable=False)  # concise|balanced|detailed
    tone = Column(String, default="professional", nullable=False)  # formal|casual|technical|friendly
    
    # Default chat mode
    default_chat_mode = Column(String, default="standard", nullable=False)  # standard|experimental|debate
    
    # Model preferences
    preferred_model = Column(String, nullable=True)
    preferred_provider = Column(String, nullable=True)
    
    # UI & behavior
    dark_mode = Column(Boolean, default=True)
    show_reasoning = Column(Boolean, default=False)  # Show debug/reasoning data
    auto_save_chats = Column(Boolean, default=True)
    chat_retention_days = Column(Integer, default=90)  # Auto-delete old chats after N days
    
    # Metadata
    metadata_json = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserSession(Base):
    """
    Session tracking for multi-tab/device support.
    
    Ensures chat continuity even when:
      - User logs out and logs back in
      - User uses multiple devices/tabs
      - Network interruption occurs
    """
    __tablename__ = "user_session"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    
    # Session identifier (e.g., device ID, browser fingerprint)
    session_token = Column(String, unique=True, index=True, nullable=False)
    device_id = Column(String, nullable=True)
    device_name = Column(String, nullable=True)
    
    # Browser/app info
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    
    # Session state
    is_active = Column(Boolean, default=True)
    last_activity_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Session expiration time
