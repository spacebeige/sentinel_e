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
    latency_ms: float = 0.0
    model_color: str = ""
    risks: List[str] = Field(default_factory=list)
    weaknesses_found: List[str] = Field(default_factory=list)
    position_shift: str = "none"


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
    """Complete debate result with all rounds and drift/rift analytics."""
    rounds: List[DebateRound] = Field(default_factory=list)
    total_rounds: int = 0
    shift_table: List[ShiftRecord] = Field(default_factory=list)
    final_consensus: Optional[str] = None
    consensus_strength: float = 0.0
    unresolved_conflicts: List[str] = Field(default_factory=list)
    # Drift/Rift analytics
    drift_index: float = 0.0
    rift_index: float = 0.0
    confidence_spread: float = 0.0
    fragility_score: float = 0.0
    per_model_drift: Dict[str, List[float]] = Field(default_factory=dict)
    per_round_rift: List[float] = Field(default_factory=list)
    per_round_disagreement: List[float] = Field(default_factory=list)
    overall_confidence: float = 0.5
    # Analysis fields
    conflict_axes: List[str] = Field(default_factory=list)
    disagreement_strength: float = 0.0
    logical_stability: float = 0.5
    convergence_level: str = "none"
    convergence_detail: str = ""
    strongest_argument: str = ""
    weakest_argument: str = ""
    synthesis: str = ""


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

    def _build_debate_result_for_frontend(self) -> Dict[str, Any]:
        """
        Build debate_result in the shape DebateView.js expects:
          { rounds: [[{model_id, model_label, model_name, position, argument,
                        assumptions, risks, rebuttals, weaknesses_found,
                        confidence, latency_ms, role, ...}, ...], ...],
            analysis: { drift_index, rift_index, fragility_score,
                        confidence_spread, strongest_argument, weakest_argument,
                        synthesis, conflict_axes, disagreement_strength,
                        convergence_level, convergence_detail,
                        confidence_recalibration, overall_confidence, ... },
            models_used: [...],
            scores: { model_name: score, ... } }
        """
        dr = self.debate_result
        frontend_rounds: List[List[Dict[str, Any]]] = []
        all_model_names: List[str] = []

        for rnd in dr.rounds:
            round_models = []
            for pos in rnd.positions:
                round_models.append({
                    "model_id": pos.model_id,
                    "model_label": pos.model_name,
                    "model_name": pos.model_name,
                    "model_color": pos.model_color or "",
                    "round_num": pos.round_number,
                    "position": pos.position,
                    "argument": pos.argument,
                    "assumptions": pos.assumptions,
                    "risks": pos.risks,
                    "rebuttals": pos.rebuttals if isinstance(pos.rebuttals, list) else ([pos.rebuttals] if pos.rebuttals else []),
                    "position_shift": pos.position_shift or ("yes" if pos.position_shifted else "none"),
                    "weaknesses_found": pos.weaknesses_found if isinstance(pos.weaknesses_found, list) else ([pos.weaknesses_found] if pos.weaknesses_found else []),
                    "confidence": pos.confidence,
                    "latency_ms": round(pos.latency_ms, 2),
                    "role": pos.model_name,
                    "vulnerabilities_found": pos.vulnerabilities_found,
                })
                if pos.model_name not in all_model_names:
                    all_model_names.append(pos.model_name)
            frontend_rounds.append(round_models)

        # Build scores from model outputs (confidence as score proxy)
        scores: Dict[str, float] = {}
        for m in self.model_outputs:
            if m.succeeded:
                scores[m.model_name] = round(m.confidence, 4)

        analysis = {
            "synthesis": dr.synthesis or "",
            "conflict_axes": dr.conflict_axes or [],
            "disagreement_strength": dr.disagreement_strength,
            "convergence_level": dr.convergence_level or "moderate",
            "convergence_detail": dr.convergence_detail or "",
            "logical_stability": dr.logical_stability,
            "strongest_argument": dr.strongest_argument or "",
            "weakest_argument": dr.weakest_argument or "",
            "confidence_recalibration": round(self.confidence.final_confidence, 4),
            "drift_index": dr.drift_index,
            "rift_index": dr.rift_index,
            "confidence_spread": dr.confidence_spread,
            "fragility_score": dr.fragility_score,
            "per_model_drift": dr.per_model_drift,
            "per_round_rift": dr.per_round_rift,
            "per_round_disagreement": dr.per_round_disagreement,
            "overall_confidence": dr.overall_confidence,
        }

        return {
            "rounds": frontend_rounds,
            "models_used": all_model_names,
            "scores": scores,
            "analysis": analysis,
        }

    def _build_debate_rounds_for_ensemble_view(self) -> List[Dict[str, Any]]:
        """
        Build debate_rounds in the shape EnsembleView.js expects:
          [{ round: N, model_outputs: [{model_id, position, reasoning,
             assumptions, vulnerabilities, confidence, stance_vector,
             error, status, latency_ms}, ...] }, ...]
        """
        rounds_list: List[Dict[str, Any]] = []
        for rnd in self.debate_result.rounds:
            model_outputs = []
            for pos in rnd.positions:
                model_outputs.append({
                    "model_id": pos.model_name,
                    "model_name": pos.model_name,
                    "position": pos.position,
                    "reasoning": pos.argument,
                    "assumptions": pos.assumptions,
                    "vulnerabilities": pos.vulnerabilities_found,
                    "confidence": pos.confidence,
                    "stance_vector": pos.stance_vector.model_dump() if pos.stance_vector else {},
                    "latency_ms": round(pos.latency_ms, 2),
                    "rebuttals": pos.rebuttals,
                    "risks": pos.risks,
                    "weaknesses_found": pos.weaknesses_found,
                    "status": "success",
                    "error": None,
                })
            rounds_list.append({
                "round": rnd.round_number,
                "model_outputs": model_outputs,
                "round_disagreement": rnd.round_disagreement,
                "convergence_delta": rnd.convergence_delta,
                "key_conflicts": rnd.key_conflicts,
            })
        return rounds_list

    def _build_agreement_matrix_for_frontend(self) -> Dict[str, Any]:
        """
        Build agreement_matrix in the shape EnsembleView AgreementHeatmap expects:
          { matrix: [[float, ...], ...], model_ids: [str, ...], clusters: [[str, ...], ...] }
        """
        am = self.agreement_matrix
        dict_form = am.to_matrix_dict()
        model_ids = sorted(dict_form.keys())

        if not model_ids:
            return {"matrix": [], "model_ids": [], "clusters": []}

        grid: List[List[float]] = []
        for row_id in model_ids:
            row = []
            for col_id in model_ids:
                row.append(round(dict_form.get(row_id, {}).get(col_id, 0.0), 4))
            grid.append(row)

        return {
            "matrix": grid,
            "model_ids": model_ids,
            "clusters": am.agreement_clusters,
            "mean_agreement": am.mean_agreement,
            "dissenting_models": am.dissenting_models,
        }

    def _build_tactical_map_for_frontend(self) -> List[Dict[str, Any]]:
        """
        Build tactical_map as flat array EnsembleView TacticalMapView expects:
          [{ finding, confidence, category, evidence_models, dissenting_models }, ...]
        """
        tm = self.tactical_map
        if not tm.entries:
            return []

        result: List[Dict[str, Any]] = []
        for entry in tm.entries:
            finding = entry.position_summary
            if entry.key_differentiator:
                finding = f"{entry.position_summary} — {entry.key_differentiator}"

            evidence_models = [
                e.model_name for e in tm.entries
                if e.model_id != entry.model_id and e.agreement_with_consensus > 0.5
            ]
            dissenting_models = self.agreement_matrix.dissenting_models or []

            result.append({
                "finding": finding,
                "confidence": entry.confidence,
                "category": "position",
                "evidence_models": [entry.model_name],
                "dissenting_models": [
                    m for m in dissenting_models if m != entry.model_name
                ],
                "model_id": entry.model_id,
                "model_name": entry.model_name,
                "position_summary": entry.position_summary,
                "agreement_with_consensus": entry.agreement_with_consensus,
                "vulnerabilities": entry.vulnerabilities,
            })
        return result

    def to_frontend_payload(self) -> Dict[str, Any]:
        """Serialize to the single frontend contract."""
        # Build all frontend-friendly structures
        debate_result = self._build_debate_result_for_frontend()
        debate_rounds = self._build_debate_rounds_for_ensemble_view()
        agreement_matrix = self._build_agreement_matrix_for_frontend()
        tactical_map = self._build_tactical_map_for_frontend()

        confidence_graph = {
            "final_confidence": self.confidence.final_confidence,
            "calibration_method": self.confidence.calibration_method,
            "components": self.confidence.components,
            "evolution": self.confidence.evolution,
            "explanation": self.confidence.explanation,
        }

        drift_metrics = {
            "drift_index": self.debate_result.drift_index,
            "rift_index": self.debate_result.rift_index,
            "fragility_score": self.debate_result.fragility_score,
            "confidence_spread": self.debate_result.confidence_spread,
            "per_model_drift": self.debate_result.per_model_drift,
            "per_round_rift": self.debate_result.per_round_rift,
            "per_round_disagreement": self.debate_result.per_round_disagreement,
        }

        session_analytics = {
            "message_count": self.session_intelligence.message_count,
            "avg_confidence": (
                sum(self.session_intelligence.confidence_history)
                / len(self.session_intelligence.confidence_history)
                if self.session_intelligence.confidence_history else 0.0
            ),
            "topic_clusters": self.session_intelligence.topic_clusters,
            "boundary_hits": self.session_intelligence.boundary_hits,
            "depth": self.session_intelligence.depth,
            "volatility": self.session_intelligence.volatility,
            "model_reliability": self.session_intelligence.model_reliability,
            "inferred_domain": self.session_intelligence.inferred_domain,
        }

        model_status = []
        for m in self.model_outputs:
            model_status.append({
                "model_id": m.model_id,
                "model_name": m.model_name,
                "status": "success" if m.succeeded else "failed",
                "confidence": m.confidence,
                "latency_ms": round(m.latency_ms, 1),
                "error": m.error,
            })

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
            # DebateView.js contract
            "debate_result": debate_result,
            # EnsembleView.js contract
            "debate_rounds": debate_rounds,
            # Agreement matrix (2D grid + model_ids)
            "agreement_matrix": agreement_matrix,
            "agreement_matrix_raw": self.agreement_matrix.model_dump(),
            # Ensemble metrics
            "ensemble_metrics": self.ensemble_metrics.model_dump(),
            # Confidence graph (for EnsembleView ConfidenceDisplay)
            "calibrated_confidence": confidence_graph,
            "confidence_graph": confidence_graph,
            # Tactical map (flat array)
            "tactical_map": tactical_map,
            # Drift/rift metrics
            "drift_metrics": drift_metrics,
            # Session analytics
            "session_intelligence": self.session_intelligence.model_dump(),
            "session_analytics": session_analytics,
            # Model status
            "model_status": model_status,
            # Confidence evolution
            "confidence_evolution": self.confidence_evolution,
            # Raw omega metadata
            "omega_metadata": self.omega_metadata,
            # Error state
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
