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
    role = Column(String, nullable=False) # user | assistant | model_groq | model_qwen | model_mistral
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
