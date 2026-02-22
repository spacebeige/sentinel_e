"""
Confidence Computation Engine — Sentinel-E Cognitive Engine 3.X

Production-grade, mathematically defensible confidence computation.

Confidence is computed from weighted components:
  final_confidence =
    base_model_confidence
    + evidence_weight
    + reliability_adjustment
    - boundary_penalty
    - disagreement_penalty
    - fragility_penalty
    - domain_uncertainty

Each component is logged internally.
Only the final value is exposed in STANDARD mode.
All components are exposed in RESEARCH modes.
"""

import logging
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ConfidenceEngine")


# ============================================================
# CONFIDENCE COMPONENTS
# ============================================================

@dataclass
class ConfidenceComponents:
    """Individual components contributing to final confidence."""
    base_model_confidence: float = 0.5
    evidence_weight: float = 0.0
    reliability_adjustment: float = 0.0
    boundary_penalty: float = 0.0
    disagreement_penalty: float = 0.0
    fragility_penalty: float = 0.0
    domain_uncertainty: float = 0.0
    historical_model_reliability: float = 1.0  # Multiplier from learning

    @property
    def raw_confidence(self) -> float:
        """Compute raw confidence before learning adjustment."""
        raw = (
            self.base_model_confidence
            + self.evidence_weight
            + self.reliability_adjustment
            - self.boundary_penalty
            - self.disagreement_penalty
            - self.fragility_penalty
            - self.domain_uncertainty
        )
        return max(0.01, min(0.99, raw))

    @property
    def final_confidence(self) -> float:
        """Compute final confidence with learning adjustment."""
        adjusted = self.raw_confidence * self.historical_model_reliability
        return round(max(0.01, min(0.99, adjusted)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model_confidence": round(self.base_model_confidence, 4),
            "evidence_weight": round(self.evidence_weight, 4),
            "reliability_adjustment": round(self.reliability_adjustment, 4),
            "boundary_penalty": round(self.boundary_penalty, 4),
            "disagreement_penalty": round(self.disagreement_penalty, 4),
            "fragility_penalty": round(self.fragility_penalty, 4),
            "domain_uncertainty": round(self.domain_uncertainty, 4),
            "historical_model_reliability": round(self.historical_model_reliability, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "final_confidence": self.final_confidence,
        }


# ============================================================
# CONFIDENCE EVOLUTION TRACE
# ============================================================

@dataclass
class ConfidenceTrace:
    """Track confidence evolution across processing stages."""
    initial: float = 0.5
    post_debate: Optional[float] = None
    post_boundary: Optional[float] = None
    post_evidence: Optional[float] = None
    post_stress: Optional[float] = None
    final: float = 0.5
    stage_log: List[Dict[str, Any]] = field(default_factory=list)

    def record_stage(self, stage: str, value: float, reason: str = ""):
        """Record a confidence change at a specific stage."""
        self.stage_log.append({
            "stage": stage,
            "value": round(value, 4),
            "reason": reason,
        })
        # Update the appropriate field
        if stage == "initial":
            self.initial = value
        elif stage == "post_debate":
            self.post_debate = value
        elif stage == "post_boundary":
            self.post_boundary = value
        elif stage == "post_evidence":
            self.post_evidence = value
        elif stage == "post_stress":
            self.post_stress = value
        elif stage == "final":
            self.final = value

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "initial": round(self.initial, 4),
            "final": round(self.final, 4),
            "stages": self.stage_log,
        }
        if self.post_debate is not None:
            result["post_debate"] = round(self.post_debate, 4)
        if self.post_boundary is not None:
            result["post_boundary"] = round(self.post_boundary, 4)
        if self.post_evidence is not None:
            result["post_evidence"] = round(self.post_evidence, 4)
        if self.post_stress is not None:
            result["post_stress"] = round(self.post_stress, 4)
        return result

    def get_explanation(self) -> str:
        """Generate human-readable explanation of confidence changes."""
        if not self.stage_log:
            return "No confidence trace available."
        parts = []
        for entry in self.stage_log:
            reason = entry.get("reason", "")
            stage = entry["stage"].replace("_", " ").title()
            val = entry["value"]
            desc = f"{stage}: {val:.0%}"
            if reason:
                desc += f" ({reason})"
            parts.append(desc)
        return " → ".join(parts)


# ============================================================
# CONFIDENCE ENGINE
# ============================================================

class ConfidenceEngine:
    """
    Computes mathematically defensible confidence scores.
    
    Used by:
    - Standard mode: Single final confidence
    - Debate mode: Per-model and consensus confidence
    - Glass mode: Full pipeline visibility
    - Evidence mode: Evidence-weighted confidence
    - Stress mode: Post-stress confidence
    """

    # Domain uncertainty lookup (higher = more uncertain)
    DOMAIN_UNCERTAINTY = {
        "general": 0.05,
        "software_engineering": 0.03,
        "machine_learning": 0.06,
        "cybersecurity": 0.08,
        "business_strategy": 0.10,
        "scientific_research": 0.12,
        "systems_thinking": 0.07,
        "medical": 0.15,
        "legal": 0.14,
        "financial": 0.12,
    }

    def compute_standard(
        self,
        base_confidence: float,
        boundary_result: Dict[str, Any],
        session_state: Dict[str, Any],
        evidence_result: Optional[Dict[str, Any]] = None,
        model_weight: float = 1.0,
    ) -> ConfidenceComponents:
        """
        Compute confidence for STANDARD mode.
        
        Returns ConfidenceComponents with all internal values + final.
        """
        components = ConfidenceComponents()

        # 1. Base model confidence (from multipass reasoning)
        components.base_model_confidence = max(0.1, min(0.95, base_confidence))

        # 2. Evidence weight
        if evidence_result:
            source_count = evidence_result.get("source_count", 0)
            evidence_conf = evidence_result.get("evidence_confidence", 0.0)
            components.evidence_weight = min(0.15, evidence_conf * 0.2 * min(source_count / 3, 1.0))

        # 3. Reliability adjustment (from session expertise + history)
        expertise = session_state.get("user_expertise_score", 0.5)
        reliability = session_state.get("reliability_score", 0.5)
        components.reliability_adjustment = (reliability - 0.5) * 0.1 + (expertise - 0.5) * 0.05

        # 4. Boundary penalty
        severity = boundary_result.get("severity_score", 0)
        if severity > 0:
            # Sigmoid-based penalty: smooth transition, heavy at high severity
            components.boundary_penalty = self._sigmoid_penalty(severity / 100, midpoint=0.5, steepness=6)

        # 5. Disagreement penalty
        disagreement = session_state.get("disagreement_score", 0.0)
        components.disagreement_penalty = disagreement * 0.15

        # 6. Fragility penalty
        fragility = session_state.get("fragility_index", 0.0)
        components.fragility_penalty = fragility * 0.12

        # 7. Domain uncertainty
        domain = session_state.get("inferred_domain", "general")
        components.domain_uncertainty = self.DOMAIN_UNCERTAINTY.get(domain, 0.05)

        # 8. Historical model reliability (from learning system)
        components.historical_model_reliability = max(0.5, min(1.0, model_weight))

        logger.info(
            f"Confidence computed: base={components.base_model_confidence:.3f} "
            f"evidence={components.evidence_weight:.3f} "
            f"boundary_pen={components.boundary_penalty:.3f} "
            f"disagree_pen={components.disagreement_penalty:.3f} "
            f"fragility_pen={components.fragility_penalty:.3f} "
            f"domain_unc={components.domain_uncertainty:.3f} "
            f"model_reliability={components.historical_model_reliability:.3f} "
            f"→ final={components.final_confidence:.3f}"
        )

        return components

    def compute_debate_consensus(
        self,
        model_confidences: Dict[str, float],
        disagreement_score: float,
        convergence_level: str,
        boundary_severity: float,
        model_weights: Dict[str, float] = None,
    ) -> ConfidenceComponents:
        """
        Compute consensus confidence across debate models.
        
        Considers:
        - Weighted average of model confidences
        - Penalized by disagreement
        - Boosted by convergence
        """
        model_weights = model_weights or {}
        components = ConfidenceComponents()

        # Weighted average of model confidences
        total_weight = 0.0
        weighted_sum = 0.0
        for model, conf in model_confidences.items():
            w = model_weights.get(model, 1.0)
            weighted_sum += conf * w
            total_weight += w

        components.base_model_confidence = weighted_sum / max(total_weight, 1.0)

        # Convergence bonus
        convergence_bonus = {
            "high": 0.08,
            "moderate": 0.04,
            "low": 0.01,
            "none": -0.05,
        }
        components.reliability_adjustment = convergence_bonus.get(convergence_level, 0.0)

        # Disagreement penalty
        components.disagreement_penalty = disagreement_score * 0.20

        # Boundary penalty
        components.boundary_penalty = self._sigmoid_penalty(
            boundary_severity / 100, midpoint=0.5, steepness=5
        )

        return components

    def compute_evidence_adjusted(
        self,
        base_components: ConfidenceComponents,
        evidence_result: Dict[str, Any],
    ) -> ConfidenceComponents:
        """
        Adjust confidence based on evidence analysis results.
        """
        # Clone components
        adjusted = ConfidenceComponents(
            base_model_confidence=base_components.base_model_confidence,
            reliability_adjustment=base_components.reliability_adjustment,
            boundary_penalty=base_components.boundary_penalty,
            disagreement_penalty=base_components.disagreement_penalty,
            fragility_penalty=base_components.fragility_penalty,
            domain_uncertainty=base_components.domain_uncertainty,
            historical_model_reliability=base_components.historical_model_reliability,
        )

        # Evidence influence
        source_count = evidence_result.get("source_count", 0)
        source_agreement = evidence_result.get("source_agreement", 0.0)
        contradictions = evidence_result.get("contradiction_count", 0)
        evidence_confidence = evidence_result.get("evidence_confidence", 0.0)

        # Evidence weight: more sources + higher agreement = more evidence boost
        adjusted.evidence_weight = min(
            0.20,
            evidence_confidence * 0.15 + source_agreement * 0.05
        )

        # Contradiction penalty augmentation
        if contradictions > 0:
            adjusted.disagreement_penalty += contradictions * 0.03

        return adjusted

    def compute_stress_adjusted(
        self,
        base_components: ConfidenceComponents,
        stress_result: Dict[str, Any],
    ) -> ConfidenceComponents:
        """
        Adjust confidence after stress testing.
        """
        adjusted = ConfidenceComponents(
            base_model_confidence=base_components.base_model_confidence,
            evidence_weight=base_components.evidence_weight,
            reliability_adjustment=base_components.reliability_adjustment,
            boundary_penalty=base_components.boundary_penalty,
            disagreement_penalty=base_components.disagreement_penalty,
            domain_uncertainty=base_components.domain_uncertainty,
            historical_model_reliability=base_components.historical_model_reliability,
        )

        # Stress survival factor
        stability = stress_result.get("stability_after_stress", 0.5)
        contradictions_found = stress_result.get("contradictions_found", 0)

        # If unstable under stress, increase fragility penalty
        adjusted.fragility_penalty = (1.0 - stability) * 0.20
        adjusted.disagreement_penalty += contradictions_found * 0.04

        return adjusted

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    @staticmethod
    def _sigmoid_penalty(x: float, midpoint: float = 0.5, steepness: float = 10) -> float:
        """
        Compute a sigmoid-scaled penalty.
        
        Returns 0.0 for low x, ~0.5 at midpoint, approaches max_penalty for high x.
        More mathematically defensible than linear scaling.
        """
        try:
            exponent = -steepness * (x - midpoint)
            sigmoid = 1.0 / (1.0 + math.exp(exponent))
            # Scale to max penalty of 0.25
            return round(sigmoid * 0.25, 4)
        except OverflowError:
            return 0.0 if x < midpoint else 0.25
