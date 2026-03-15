import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, ARRAY, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()

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
