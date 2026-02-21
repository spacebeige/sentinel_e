from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime

class SentinelRequest(BaseModel):
    text: str
    mode: str = "conversational" # conversational | forensic | experimental
    enable_shadow: bool = False
    rounds: int = 1
    chat_id: Optional[UUID] = None  # Added for continuation

class ModelPosition(BaseModel):
    model: str
    position: str
    confidence: float
    key_points: List[str]

class MachineMetadata(BaseModel):
    models_used: List[str]
    rounds_executed: int
    debate_depth: int
    shadow_enabled: bool
    parse_success_rate: float
    variance_score: float
    historical_instability_score: float

class ShadowAnalysis(BaseModel):
    is_safe: bool
    triggers: List[str]
    risk_score: float
    raw_analysis: Optional[str] = None

class SentinelResponse(BaseModel):
    chat_id: UUID  # Strict UUID enforcement
    chat_name: str
    mode: str
    data: Dict[str, Any]  # Encapsulated payload
    metadata: MachineMetadata
    # No top-level duplicates per strict contract

