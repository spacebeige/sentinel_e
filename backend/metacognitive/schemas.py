"""
============================================================
Meta-Cognitive Orchestrator — Data Contracts & Schemas
============================================================
Strict Pydantic models enforcing the 3-API contract.
No cross-contamination. No merged responsibilities.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# Enums
# ============================================================

class OperatingMode(str, Enum):
    STANDARD = "standard"
    EXPERIMENTAL = "experimental"


class QueryMode(str, Enum):
    RAW = "raw"
    REFINED = "refined"
    RESEARCH = "research"


class ModelRole(str, Enum):
    """Routing hint for model selection."""
    CODE = "code"
    VISION = "vision"
    CONCEPTUAL = "conceptual"
    BASELINE = "baseline"
    LONGCTX = "longctx"
    FAST = "fast"
    GENERAL = "general"


# ============================================================
# Shared Primitives
# ============================================================

class EmbeddingVector(BaseModel):
    """Wrapper for embedding arrays with metadata."""
    vector: List[float] = Field(default_factory=list)
    model: str = "sentence-transformers"
    dimensions: int = 384


# ============================================================
# API 2 — Knowledge & Retrieval Engine Schemas
# (Defined first because API 1 depends on KnowledgeBlock)
# ============================================================

class KnowledgeBlock(BaseModel):
    """Single normalized factual block. No summarization before storage."""
    source: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content: str = ""
    embedding: List[float] = Field(default_factory=list)
    confidence: float = 0.0
    domain: str = ""


class KnowledgeRetrievalInput(BaseModel):
    """Input contract for Knowledge & Retrieval API."""
    query_embedding: List[float] = Field(default_factory=list)
    query_text: str = ""
    volatility_score: float = 0.0
    domain: str = ""
    concept_expansion_depth: int = 1


class KnowledgeRetrievalOutput(BaseModel):
    """Output contract from Knowledge & Retrieval API."""
    knowledge_bundle: List[KnowledgeBlock] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    sources_queried: int = 0
    expansion_applied: bool = False


# ============================================================
# API 1 — Cognitive Model Gateway Schemas
# ============================================================

class CognitiveGatewayInput(BaseModel):
    """Input contract for Cognitive Model Gateway."""
    stabilized_context: Dict[str, Any] = Field(default_factory=dict)
    knowledge_bundle: List[KnowledgeBlock] = Field(default_factory=list)
    session_summary: Dict[str, Any] = Field(default_factory=dict)
    user_query: str
    mode: QueryMode = QueryMode.RAW


class CognitiveGatewayOutput(BaseModel):
    """Output contract from a single model invocation."""
    model_name: str
    raw_output: str
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    confidence_estimate: Optional[float] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    success: bool = True


# ============================================================
# API 3 — Session & Persistence Engine Schemas
# ============================================================

class StructuredGoal(BaseModel):
    """A user-facing goal tracked across the session."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: str = "active"  # active | resolved | abandoned
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: Optional[str] = None


class MemoryBlock(BaseModel):
    """Session memory block with relevance scoring."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: List[float] = Field(default_factory=list)
    relevance_score: float = 0.0
    turn_created: int = 0
    source: str = "user"  # user | model | retrieval | daemon


class BehavioralRecord(BaseModel):
    """Single behavioral observation."""
    model_name: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    final_score: float = 0.0
    topic_alignment: float = 0.0
    grounding_score: float = 0.0
    specificity: float = 0.0
    confidence_calibration: float = 0.0
    drift_penalty: float = 0.0


class UnresolvedQuestion(BaseModel):
    """Question that requires async daemon refinement."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    priority: float = 0.5
    attempts: int = 0
    last_attempt: Optional[str] = None


class SessionState(BaseModel):
    """Full session object for API 3."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic_centroid_embedding: List[float] = Field(default_factory=list)
    structured_goals: List[StructuredGoal] = Field(default_factory=list)
    unresolved_questions: List[UnresolvedQuestion] = Field(default_factory=list)
    memory_blocks: List[MemoryBlock] = Field(default_factory=list)
    behavioral_history: List[BehavioralRecord] = Field(default_factory=list)
    drift_score: float = 0.0
    volatility_score: float = 0.0
    refinement_cycles: int = 0
    mode: OperatingMode = OperatingMode.STANDARD
    turn_count: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ============================================================
# Arbitration Schemas
# ============================================================

class ArbitrationScore(BaseModel):
    """Scoring breakdown for a single model output."""
    model_name: str
    topic_alignment: float = 0.0       # T
    knowledge_grounding: float = 0.0   # K
    specificity: float = 0.0           # S
    confidence_calibration: float = 0.0  # C
    drift_penalty: float = 0.0         # D
    final_score: float = 0.0

    def compute_final(self) -> float:
        """
        FinalScore = 0.30*T + 0.25*K + 0.15*S + 0.15*C - 0.15*D
        """
        self.final_score = (
            0.30 * self.topic_alignment
            + 0.25 * self.knowledge_grounding
            + 0.15 * self.specificity
            + 0.15 * self.confidence_calibration
            - 0.15 * self.drift_penalty
        )
        return self.final_score


# ============================================================
# Orchestrator I/O
# ============================================================

class OrchestratorRequest(BaseModel):
    """Top-level request to the Meta-Cognitive Orchestrator."""
    session_id: Optional[str] = None
    query: str
    mode: OperatingMode = OperatingMode.STANDARD
    sub_mode: Optional[str] = None  # debate | evidence | glass (experimental only)
    chat_id: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    force_retrieval: bool = False
    selected_model: Optional[str] = None  # Single Model Focus Mode


class ModelResult(BaseModel):
    """Result from a single model with arbitration score."""
    output: CognitiveGatewayOutput
    score: ArbitrationScore


class OrchestratorResponse(BaseModel):
    """Top-level response from the Meta-Cognitive Orchestrator."""
    session_id: str
    chat_id: str = ""
    mode: OperatingMode
    sub_mode: Optional[str] = None  # debate | evidence | glass
    aggregated_answer: str = ""
    winning_model: str = ""
    winning_score: float = 0.0
    all_results: List[ModelResult] = Field(default_factory=list)
    knowledge_bundle: List[KnowledgeBlock] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    session_state: Optional[SessionState] = None
    drift_score: float = 0.0
    volatility_score: float = 0.0
    refinement_cycles: int = 0
    latency_ms: float = 0.0
    # Experimental-mode observability
    divergence_metrics: Optional[Dict[str, Any]] = None
    scoring_breakdown: Optional[List[ArbitrationScore]] = None
