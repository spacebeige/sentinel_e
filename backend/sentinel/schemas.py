from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime

class SentinelRequest(BaseModel):
    text: str = Field(..., max_length=50000, description="User query text")
    mode: str = Field(default="conversational", description="conversational | standard | research | experimental | kill")
    sub_mode: Optional[str] = Field(default=None, description="debate | glass | evidence | stress")
    enable_shadow: bool = False
    rounds: int = Field(default=1, ge=1, le=10, description="Max debate/analysis rounds")
    chat_id: Optional[UUID] = None
    role_map: Optional[Dict[str, str]] = None
    kill: bool = False

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        valid = {"conversational", "standard", "research", "experimental", "kill", "forensic", "ensemble"}
        if v not in valid:
            return "standard"
        return v

    @field_validator("sub_mode")
    @classmethod
    def validate_sub_mode(cls, v):
        if v is None:
            return v
        valid = {"debate", "glass", "evidence", "stress"}
        if v not in valid:
            return "debate"
        return v

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


# ============================================================
# OMEGA COGNITIVE KERNEL SCHEMAS
# ============================================================

class OmegaSessionState(BaseModel):
    """Persistent session intelligence state."""
    session_id: str
    chat_name: Optional[str] = None
    primary_goal: Optional[str] = None
    inferred_domain: str = "general"
    user_expertise_score: float = 0.5
    message_count: int = 0
    error_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    boundary_history_count: int = 0
    latest_boundary_severity: int = 0
    boundary_trend: str = "insufficient_data"
    disagreement_score: float = 0.0
    fragility_index: float = 0.0
    session_confidence: float = 0.5
    reasoning_depth: str = "standard"

class OmegaReasoningTrace(BaseModel):
    """Structured summary of multi-pass reasoning (never raw)."""
    passes_executed: int = 0
    initial_confidence: float = 0.5
    final_confidence: float = 0.5
    assumptions_extracted: int = 0
    logical_gaps_detected: int = 0
    boundary_severity: int = 0
    self_critique_applied: bool = False
    refinement_applied: bool = False

class OmegaBoundaryResult(BaseModel):
    """Boundary evaluation result."""
    risk_level: str = "LOW"
    severity_score: int = 0
    explanation: str = ""
    risk_dimensions: Dict[str, float] = Field(default_factory=dict)
    human_review_required: bool = False

class OmegaConfidenceEvolution(BaseModel):
    """Confidence tracking across reasoning stages."""
    initial: float = 0.5
    post_debate: Optional[float] = None
    post_boundary: Optional[float] = None
    post_evidence: Optional[float] = None
    post_stress: Optional[float] = None
    final: float = 0.5


class StressVectorResult(BaseModel):
    """Result from a single stress vector."""
    vector: str = ""
    severity: float = 0.0
    stable: bool = True
    finding: str = ""


class StressResultSchema(BaseModel):
    """Full stress test result."""
    stability_after_stress: float = 0.5
    contradictions_found: int = 0
    revised_confidence: float = 0.5
    overall_stability: float = 0.5
    vector_results: Dict[str, Any] = Field(default_factory=dict)
    breakdown_points: List[str] = Field(default_factory=list)

class BehavioralRiskProfileSchema(BaseModel):
    """Behavioral analytics risk profile for a single response."""
    self_preservation_score: float = 0.0
    manipulation_probability: float = 0.0
    evasion_index: float = 0.0
    confidence_inflation: float = 0.0
    overall_risk: float = 0.0
    risk_level: str = "LOW"
    signals_detected: int = 0
    signal_breakdown: Dict[str, int] = Field(default_factory=dict)
    explanation: str = ""


class EvidenceSourceSchema(BaseModel):
    """A single evidence source."""
    url: str = ""
    title: str = ""
    content_snippet: str = ""
    reliability_score: float = 0.5
    domain: str = ""


class EvidenceResultSchema(BaseModel):
    """Evidence engine result."""
    query: str = ""
    sources: List[EvidenceSourceSchema] = Field(default_factory=list)
    source_count: int = 0
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    contradiction_count: int = 0
    evidence_confidence: float = 0.5
    source_agreement: float = 0.0
    lineage: List[Dict[str, str]] = Field(default_factory=list)
    search_executed: bool = False


class FeedbackRequest(BaseModel):
    """Enhanced feedback model for Sentinel-E 3.0."""
    run_id: str
    session_id: Optional[str] = None
    mode: Optional[str] = None
    sub_mode: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)      # 1-5 star rating
    vote: Optional[str] = None                             # "up" | "down"
    comment: Optional[str] = None
    # Context for learning
    boundary_severity: Optional[float] = None
    fragility_index: Optional[float] = None
    disagreement_score: Optional[float] = None
    confidence: Optional[float] = None


class OmegaMetadata(BaseModel):
    """Extended metadata for Omega Kernel responses."""
    models_used: List[str] = Field(default_factory=list)
    rounds_executed: int = 0
    debate_depth: int = 0
    shadow_enabled: bool = False
    parse_success_rate: float = 1.0
    variance_score: float = 0.0
    historical_instability_score: float = 0.0
    # Omega extensions
    omega_version: str = "3.0.0"
    passes_executed: int = 9
    sub_mode: Optional[str] = None                                 # debate | glass | evidence | stress
    session_state: Optional[OmegaSessionState] = None
    reasoning_trace: Optional[OmegaReasoningTrace] = None
    boundary_result: Optional[OmegaBoundaryResult] = None
    confidence_evolution: Optional[OmegaConfidenceEvolution] = None
    fragility_index: float = 0.0
    behavioral_risk: Optional[BehavioralRiskProfileSchema] = None  # Glass mode
    evidence_result: Optional[EvidenceResultSchema] = None         # Evidence mode
    stress_result: Optional[StressResultSchema] = None             # Stress mode
    confidence_components: Optional[Dict[str, Any]] = None         # Detailed confidence breakdown


class SentinelResponse(BaseModel):
    chat_id: UUID  # Strict UUID enforcement
    chat_name: str
    mode: str
    data: Dict[str, Any]  # Encapsulated payload
    metadata: MachineMetadata
    # No top-level duplicates per strict contract


class OmegaResponse(BaseModel):
    """
    Omega Cognitive Kernel response.
    Extends SentinelResponse with structured cognitive state.
    """
    chat_id: UUID
    chat_name: str
    mode: str
    sub_mode: Optional[str] = None     # debate | glass | evidence
    formatted_output: str              # Structured markdown output per mode spec
    data: Dict[str, Any]               # Raw data payload for frontend consumption
    metadata: OmegaMetadata            # Extended Omega metadata
    session_state: OmegaSessionState
    reasoning_trace: OmegaReasoningTrace
    boundary_result: OmegaBoundaryResult
    confidence: float

