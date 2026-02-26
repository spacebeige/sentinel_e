"""
============================================================
Ensemble Schemas — Sentinel-E Cognitive Engine v6.0
============================================================
Strict data contracts for the ensemble-driven cognitive engine.

Every request produces ALL of these structures. No optional bypass.
No mode-based routing. No single-model fallback.

Data Flow:
    Request → StructuredModelOutput[] → AgreementMatrix → DebateRound[]
    → EnsembleMetrics → CalibratedConfidence → SessionIntelligence
    → EnsembleResponse (full metadata)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ============================================================
# Constants
# ============================================================

MIN_MODELS = 3
MIN_DEBATE_ROUNDS = 3
MIN_ANALYTICS_OUTPUTS = 2


# ============================================================
# Structured Model Output — Every model returns this
# ============================================================

class StanceVector(BaseModel):
    """Dimensional stance encoding for agreement matrix computation."""
    certainty: float = Field(0.5, ge=0.0, le=1.0, description="How certain the model is")
    specificity: float = Field(0.5, ge=0.0, le=1.0, description="How specific vs generic")
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0, description="How risk-tolerant the position is")
    evidence_reliance: float = Field(0.5, ge=0.0, le=1.0, description="How evidence-driven vs reasoning-driven")
    novelty: float = Field(0.5, ge=0.0, le=1.0, description="How novel vs conventional the position is")

    def to_vector(self) -> List[float]:
        return [self.certainty, self.specificity, self.risk_tolerance,
                self.evidence_reliance, self.novelty]


class StructuredModelOutput(BaseModel):
    """
    Mandatory structured output from every model on every request.
    
    This is NOT optional. Every model must produce this schema.
    If a model fails to produce structured output, the system
    extracts it from raw text via fallback parsing.
    """
    model_id: str
    model_name: str
    position: str = Field("", description="Clear thesis/position statement")
    reasoning: str = Field("", description="Step-by-step reasoning chain")
    assumptions: List[str] = Field(default_factory=list, description="Explicit assumptions")
    vulnerabilities: List[str] = Field(default_factory=list, description="Self-identified weaknesses")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Model self-reported confidence")
    stance_vector: StanceVector = Field(default_factory=StanceVector)
    raw_output: str = Field("", description="Full raw model output")
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.position)


# ============================================================
# Agreement Matrix
# ============================================================

class PairwiseScore(BaseModel):
    """Similarity score between two models."""
    model_a: str
    model_b: str
    position_similarity: float = Field(0.0, ge=0.0, le=1.0)
    reasoning_similarity: float = Field(0.0, ge=0.0, le=1.0)
    stance_distance: float = Field(0.0, ge=0.0, description="Euclidean distance in stance space")
    assumption_overlap: float = Field(0.0, ge=0.0, le=1.0)
    overall_agreement: float = Field(0.0, ge=0.0, le=1.0)


class AgreementMatrix(BaseModel):
    """Full pairwise agreement matrix across all models."""
    pairs: List[PairwiseScore] = Field(default_factory=list)
    mean_agreement: float = 0.0
    min_agreement: float = 0.0
    max_agreement: float = 0.0
    agreement_clusters: List[List[str]] = Field(
        default_factory=list, description="Groups of models that agree"
    )
    dissenting_models: List[str] = Field(
        default_factory=list, description="Models that disagree with majority"
    )

    def to_matrix_dict(self) -> Dict[str, Dict[str, float]]:
        """Return as {model_a: {model_b: agreement}} dict for frontend."""
        matrix: Dict[str, Dict[str, float]] = {}
        for pair in self.pairs:
            matrix.setdefault(pair.model_a, {})[pair.model_b] = pair.overall_agreement
            matrix.setdefault(pair.model_b, {})[pair.model_a] = pair.overall_agreement
        # Self-agreement = 1.0
        for model_id in matrix:
            matrix[model_id][model_id] = 1.0
        return matrix


# ============================================================
# Debate Structures
# ============================================================

class DebatePosition(BaseModel):
    """A model's position in a single debate round."""
    model_id: str
    model_name: str
    round_number: int
    position: str
    argument: str
    rebuttals: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    vulnerabilities_found: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    stance_vector: StanceVector = Field(default_factory=StanceVector)
    position_shifted: bool = False
    shift_reason: Optional[str] = None


class DebateRound(BaseModel):
    """A single round of structured debate."""
    round_number: int
    positions: List[DebatePosition] = Field(default_factory=list)
    round_disagreement: float = 0.0
    convergence_delta: float = Field(
        0.0, description="Change in agreement from previous round"
    )
    key_conflicts: List[str] = Field(default_factory=list)


class ShiftRecord(BaseModel):
    """Records a model shifting its position across rounds."""
    model_id: str
    model_name: str
    from_round: int
    to_round: int
    old_position_summary: str
    new_position_summary: str
    shift_magnitude: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""


class DebateResult(BaseModel):
    """Complete debate result with all rounds."""
    rounds: List[DebateRound] = Field(default_factory=list)
    total_rounds: int = 0
    shift_table: List[ShiftRecord] = Field(default_factory=list)
    final_consensus: Optional[str] = None
    consensus_strength: float = 0.0
    unresolved_conflicts: List[str] = Field(default_factory=list)


# ============================================================
# Ensemble Metrics
# ============================================================

class EnsembleMetrics(BaseModel):
    """
    Computed metrics from ensemble execution.
    
    These are computed FROM model outputs, NOT self-reported.
    They drive confidence calibration.
    """
    disagreement_entropy: float = Field(
        0.0, ge=0.0,
        description="Shannon entropy of position distribution. Higher = more disagreement."
    )
    contradiction_density: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Fraction of model pairs with contradicting positions."
    )
    stability_index: float = Field(
        0.0, ge=0.0, le=1.0,
        description="How stable positions are across debate rounds. 1.0 = fully stable."
    )
    consensus_velocity: float = Field(
        0.0,
        description="Rate of convergence across rounds. Positive = converging."
    )
    fragility_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="How fragile the consensus is. High = small perturbation breaks it."
    )
    model_count: int = 0
    round_count: int = 0
    successful_models: int = 0
    failed_models: int = 0
    mean_model_confidence: float = 0.0
    confidence_spread: float = Field(
        0.0, description="Max confidence - min confidence across models"
    )


# ============================================================
# Calibrated Confidence
# ============================================================

class CalibratedConfidence(BaseModel):
    """
    System confidence derived from ensemble metrics.
    
    NOT based on any single model's self-reported confidence.
    Computed entirely from inter-model agreement, stability, and entropy.
    """
    final_confidence: float = Field(0.5, ge=0.05, le=0.95)
    calibration_method: str = "entropy_weighted_ensemble"
    components: Dict[str, float] = Field(default_factory=dict)
    evolution: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Confidence at each stage: [initial, post_debate, post_calibration]"
    )
    explanation: str = ""

    @field_validator("final_confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.05, min(0.95, v))


# ============================================================
# Tactical Map
# ============================================================

class TacticalMapEntry(BaseModel):
    """One row in the tactical map."""
    model_id: str
    model_name: str
    position_summary: str
    confidence: float
    stance_vector: StanceVector = Field(default_factory=StanceVector)
    agreement_with_consensus: float = 0.0
    key_differentiator: str = ""
    vulnerabilities: List[str] = Field(default_factory=list)


class TacticalMap(BaseModel):
    """Visual mapping of all model positions for frontend."""
    entries: List[TacticalMapEntry] = Field(default_factory=list)
    consensus_position: str = ""
    primary_axis_of_disagreement: str = ""
    model_clusters: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================
# Session Intelligence
# ============================================================

class SessionIntelligenceSnapshot(BaseModel):
    """Session state updated after every response."""
    session_id: str = ""
    message_count: int = 0
    boundary_hits: int = 0
    depth: float = Field(0.0, description="Conversational depth score")
    volatility: float = Field(0.0, description="Topic/confidence volatility")
    topic_clusters: List[str] = Field(default_factory=list)
    confidence_history: List[float] = Field(default_factory=list)
    entropy_history: List[float] = Field(default_factory=list)
    fragility_history: List[float] = Field(default_factory=list)
    model_reliability: Dict[str, float] = Field(default_factory=dict)
    cumulative_agreement: float = 0.0
    inferred_domain: str = ""
    user_expertise_estimate: float = 0.5


# ============================================================
# Ensemble Response — The Single Frontend Contract
# ============================================================

class EnsembleResponse(BaseModel):
    """
    The ONLY response format from the backend.
    
    Every field is populated on every request.
    No optional bypass. No mode-based variants.
    Frontend renders ALL of this.
    """
    # Core
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_id: str = ""
    chat_name: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Synthesized output
    formatted_output: str = Field("", description="Markdown-formatted synthesis")
    synthesis_method: str = "ensemble_weighted"

    # Per-model structured outputs
    model_outputs: List[StructuredModelOutput] = Field(default_factory=list)
    models_executed: int = 0
    models_succeeded: int = 0
    models_failed: int = 0

    # Debate
    debate_result: DebateResult = Field(default_factory=DebateResult)

    # Agreement
    agreement_matrix: AgreementMatrix = Field(default_factory=AgreementMatrix)

    # Metrics
    ensemble_metrics: EnsembleMetrics = Field(default_factory=EnsembleMetrics)

    # Confidence
    confidence: CalibratedConfidence = Field(default_factory=CalibratedConfidence)

    # Tactical Map
    tactical_map: TacticalMap = Field(default_factory=TacticalMap)

    # Session Intelligence
    session_intelligence: SessionIntelligenceSnapshot = Field(
        default_factory=SessionIntelligenceSnapshot
    )

    # Confidence Evolution (for frontend graph)
    confidence_evolution: Dict[str, float] = Field(default_factory=dict)

    # Omega Metadata (full exposure — no hidden data)
    omega_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Error (only on hard failure)
    error: Optional[str] = None
    error_code: Optional[str] = None

    def to_frontend_payload(self) -> Dict[str, Any]:
        """Serialize to the single frontend contract."""
        return {
            "response_id": self.response_id,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "timestamp": self.timestamp,
            "formatted_output": self.formatted_output,
            "synthesis_method": self.synthesis_method,
            "model_outputs": [m.model_dump() for m in self.model_outputs],
            "models_executed": self.models_executed,
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
            "debate_result": self.debate_result.model_dump(),
            "agreement_matrix": self.agreement_matrix.to_matrix_dict(),
            "agreement_matrix_raw": self.agreement_matrix.model_dump(),
            "ensemble_metrics": self.ensemble_metrics.model_dump(),
            "confidence": self.confidence.model_dump(),
            "tactical_map": self.tactical_map.model_dump(),
            "session_intelligence": self.session_intelligence.model_dump(),
            "confidence_evolution": self.confidence_evolution,
            "omega_metadata": self.omega_metadata,
            "error": self.error,
            "error_code": self.error_code,
        }


# ============================================================
# Hard Failure Codes
# ============================================================

class EnsembleFailureCode(str, Enum):
    INSUFFICIENT_MODELS = "INSUFFICIENT_MODELS"
    INSUFFICIENT_ROUNDS = "INSUFFICIENT_ROUNDS"
    INSUFFICIENT_ANALYTICS = "INSUFFICIENT_ANALYTICS"
    EMPTY_TACTICAL_MAP = "EMPTY_TACTICAL_MAP"
    ZERO_SUCCESSFUL_OUTPUTS = "ZERO_SUCCESSFUL_OUTPUTS"


class EnsembleFailure(Exception):
    """Hard failure — no silent degradation."""
    def __init__(self, code: EnsembleFailureCode, message: str,
                 models_available: int = 0, rounds_completed: int = 0):
        self.code = code
        self.message = message
        self.models_available = models_available
        self.rounds_completed = rounds_completed
        super().__init__(f"[{code.value}] {message}")

    def to_response(self) -> EnsembleResponse:
        """Convert failure to a structured error response."""
        return EnsembleResponse(
            error=self.message,
            error_code=self.code.value,
            formatted_output=f"**System Error**: {self.message}",
            ensemble_metrics=EnsembleMetrics(
                model_count=self.models_available,
                round_count=self.rounds_completed,
            ),
        )
