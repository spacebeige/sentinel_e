"""
============================================================
Sentinel-E v5.0 — Normalized Data Model (Neon PostgreSQL)
============================================================

SCHEMA: Deterministic, single source of truth

Principles:
  • No nullable core identity fields (user_id, chat_id, session_id)
  • All writes idempotent + transactional
  • Proper foreign key relationships
  • Semantic indexes for fast queries
  • JSONB for structured metadata
  • Audit trail via created_at, updated_at

Tables:
  1. users          — Auth provider identity
  2. sessions       — User session tracking
  3. chats          — Conversation containers
  4. messages       — Conversation content
  5. memory         — User learned facts
  6. embeddings     — Optional semantic vectors
  7. user_settings  — User preferences & configuration
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, DateTime, Integer, Text, ARRAY, JSON, Boolean,
    ForeignKey, Float, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ─────────────────────────────────────────────────────────────
# TABLE: users (single source of truth for auth)
# ─────────────────────────────────────────────────────────────
class User(Base):
    """
    Auth provider identity.
    
    Constraints:
      • id (auth provider ID, e.g. from Clerk) is PRIMARY KEY
      • email is unique, indexed
      • created_at is immutable (first login timestamp)
    """
    __tablename__ = "users"

    # Auth provider ID (Clerk, Firebase, etc) — Since Supabase is used, it's a UUID
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    
    # Email — always required, always unique
    email = Column(String, unique=True, nullable=False)
    
    # Human-readable name
    name = Column(String, nullable=True)
    
    # Auth provider (clerk, firebase, supertokens)
    provider = Column(String, nullable=False, default="clerk")
    
    # Role-based access control
    role = Column(String, nullable=False, default="user")  # user | admin | moderator
    
    # Soft delete flag
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Immutable: when user first authenticated
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Updated: profile changes
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    memory_entries = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("UserSettings", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_users_id", "id"),
        Index("ix_users_email", "email"),
        Index("ix_users_created_at", "created_at"),
    )


# ─────────────────────────────────────────────────────────────
# TABLE: sessions (user session tracking)
# ─────────────────────────────────────────────────────────────
class Session(Base):
    """
    User session tracking.
    
    One session per login.
    last_active_at updates on every API call.
    """
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Foreign key to users.id (auth provider ID)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Client type: web, mobile, api
    client = Column(String, nullable=False, default="web")
    
    # IP address (optional, for audit)
    ip_address = Column(String, nullable=True)
    
    # User agent (optional, for audit)
    user_agent = Column(String, nullable=True)
    
    # Session metadata: device info, feature flags, etc.
    metadata_json = Column(JSONB, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_active_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Optional: session expiry
    expires_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index("ix_sessions_user_id", "user_id"),
        Index("ix_sessions_created_at", "created_at"),
        Index("ix_sessions_last_active_at", "last_active_at"),
    )


# ─────────────────────────────────────────────────────────────
# TABLE: chats (conversation containers)
# ─────────────────────────────────────────────────────────────
class Chat(Base):
    """
    Conversation container.
    
    Immutable: user_id, created_at
    Mutable: title, updated_at, metadata
    """
    __tablename__ = "chats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Foreign key to users.id (auth provider ID)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Conversation title
    title = Column(String, nullable=False, default="Untitled Chat")
    
    # Chat mode (optional): conversational, forensic, experimental, etc.
    mode = Column(String, nullable=True, default="conversational")
    
    # Engine/Model selection (for Pro Mode capability)
    engine = Column(String, nullable=True)
    
    # Semantic search preparedness
    search_text = Column(Text, nullable=True)
    
    # Machine metadata: model used, tokens spent, priority_answer
    machine_metadata = Column(JSONB, nullable=True, default={})
    
    # User metadata: tags, color, custom fields
    user_metadata = Column(JSONB, nullable=True, default={})
    
    # Soft delete
    is_archived = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_chats_user_id", "user_id"),
        Index("ix_chats_created_at", "created_at"),
        Index("ix_chats_updated_at", "updated_at"),
        Index("ix_chats_is_archived", "is_archived"),
    )


# ─────────────────────────────────────────────────────────────
# TABLE: messages (conversation content)
# ─────────────────────────────────────────────────────────────
class Message(Base):
    """
    Message in a chat.
    
    Immutable: chat_id, user_id, role, content, created_at
    Mutable: reasoning_json, metadata (for regenerations)
    """
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Foreign key to chats.id
    chat_id = Column(UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    
    # Foreign key to users.id (auth provider ID)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Role: user | assistant | system
    role = Column(String, nullable=False)
    
    # Message content (text, always required)
    content = Column(Text, nullable=False)
    
    # Assistant reasoning (optional, JSON structure)
    reasoning_json = Column(JSONB, nullable=True)
    
    # Metadata: model, tokens, temperature, etc.
    metadata_json = Column(JSONB, nullable=True, default={})
    
    # Image storage pointer (URL or base64 if small)
    # Preference: store URL, not base64
    image_url = Column(String, nullable=True)
    
    # Soft delete
    is_deleted = Column(Boolean, nullable=False, default=False)
    
    # Timestamp
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    chat = relationship("Chat", back_populates="messages")
    user = relationship("User", back_populates="messages")
    
    __table_args__ = (
        Index("ix_messages_chat_id", "chat_id"),
        Index("ix_messages_user_id", "user_id"),
        Index("ix_messages_created_at", "created_at"),
        Index("ix_messages_is_deleted", "is_deleted"),
    )


# ─────────────────────────────────────────────────────────────
# TABLE: memory (user-learned facts)
# ─────────────────────────────────────────────────────────────
class Memory(Base):
    """
    User-learned facts and preferences.
    
    Upserted on conflict (user_id, key).
    Weight increases with repeated signals.
    """
    __tablename__ = "memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Key: unique per user, type of fact (e.g., "preferred_model", "writing_style")
    key = Column(String, nullable=False)
    
    # Value: any JSON structure
    value = Column(JSONB, nullable=False)
    
    # Weight: reinforced by recency and frequency (0.0 to 1.0+)
    weight = Column(Float, nullable=False, default=1.0)
    
    # Confidence score: 0-100
    confidence = Column(Integer, nullable=False, default=50)
    
    # Tag for categorization
    tag = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="memory_entries")
    
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_memory_user_key"),
        Index("ix_memory_user_id", "user_id"),
        Index("ix_memory_user_id_key", "user_id", "key"),
        Index("ix_memory_weight", "weight"),
        Index("ix_memory_updated_at", "updated_at"),
    )


# ─────────────────────────────────────────────────────────────
# TABLE: embeddings (optional semantic vectors)
# ─────────────────────────────────────────────────────────────
class Embedding(Base):
    """
    Semantic embeddings for messages or memory.
    
    If using external store (Pinecone, Milvus), keep pointer.
    If using pgvector, store vector directly.
    """
    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Foreign key to users.id
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Reference type: message | memory | custom
    ref_type = Column(String, nullable=False)
    
    # Reference ID (message.id or memory.id)
    ref_id = Column(UUID(as_uuid=True), nullable=False)
    
    # Embedding vector (if using pgvector extension)
    # Otherwise, store external pointer in metadata
    vector_metadata = Column(JSONB, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_embeddings_user_id", "user_id"),
        Index("ix_embeddings_ref_type_ref_id", "ref_type", "ref_id"),
    )


# ─────────────────────────────────────────────────────────────
# TABLE: user_settings (user preferences & configuration)
# ─────────────────────────────────────────────────────────────
class UserSettings(Base):
    """
    User interface preferences and feature flags.
    
    Examples:
      • theme: dark | light
      • notifications_enabled: bool
      • preferred_model: string
      • language: en | es | etc
    """
    __tablename__ = "user_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Foreign key to users.id
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Setting key
    key = Column(String, nullable=False)
    
    # Setting value (JSON for flexibility)
    value = Column(JSONB, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="settings")
    
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_settings_user_key"),
        Index("ix_user_settings_user_id", "user_id"),
    )


# ─────────────────────────────────────────────────────────────
# VIEW: user_activity (materialized view for analytics)
# ─────────────────────────────────────────────────────────────
# Optional: Create a materialized view for fast analytics queries
# SELECT
#   u.id,
#   COUNT(DISTINCT c.id) as chat_count,
#   COUNT(DISTINCT m.id) as message_count,
#   MAX(m.created_at) as last_message_at,
#   MAX(s.last_active_at) as last_active_at
# FROM users u
# LEFT JOIN chats c ON u.id = c.user_id
# LEFT JOIN messages m ON u.id = m.user_id
# LEFT JOIN sessions s ON u.id = s.user_id
# GROUP BY u.id;
