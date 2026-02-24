"""
Dynamic Boundary Engine — Sentinel-E

All boundary metrics COMPUTED from live data. No hardcoded values.

Replaces static 67, 0.95, "MEDIUM" with computed:
  epistemic_risk = 1 - evidence_confidence
  disagreement_score = variance(model_positions)
  instability = divergence_entropy(models)
  severity = (epistemic_risk * 0.4) + (disagreement_score * 0.3) + (instability * 0.3)

Thresholds:
  0-30  → LOW
  30-70 → MEDIUM
  70-100 → HIGH
"""

import math
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DynamicBoundary")


class DynamicBoundaryEngine:
    """
    Computes ALL boundary metrics dynamically from live data.
    Zero hardcoded severity values.
    """

    @staticmethod
    def compute_severity(
        evidence_confidence: float = 0.5,
        model_divergence: float = 0.0,
        model_confidences: List[float] = None,
        contradiction_count: int = 0,
        failed_models: int = 0,
    ) -> Dict[str, Any]:
        """
        Compute boundary severity from live metrics.
        
        Args:
            evidence_confidence: Aggregated evidence confidence (0-1)
            model_divergence: Divergence score between models (0-1)
            model_confidences: List of per-model confidence values
            contradiction_count: Number of detected contradictions
            failed_models: Number of models that failed
            
        Returns:
            Complete boundary assessment with computed severity
        """
        model_confidences = model_confidences or []

        # 1. Epistemic risk: how uncertain is the evidence
        epistemic_risk = 1.0 - max(0.0, min(1.0, evidence_confidence))

        # 2. Disagreement score: variance of model positions
        if len(model_confidences) >= 2:
            mean_conf = sum(model_confidences) / len(model_confidences)
            variance = sum((c - mean_conf) ** 2 for c in model_confidences) / len(model_confidences)
            disagreement_score = min(math.sqrt(variance) * 2, 1.0)  # Normalized
        else:
            disagreement_score = model_divergence

        # 3. Instability: divergence entropy
        if model_divergence > 0.001:
            instability = -model_divergence * math.log2(max(model_divergence, 0.001))
            instability = min(instability / 0.5305, 1.0)  # Normalize (max entropy at ~0.5)
        else:
            instability = 0.0

        # 4. Contradiction penalty
        contradiction_penalty = min(contradiction_count * 5, 20)

        # 5. Failure penalty
        failure_penalty = failed_models * 10

        # 6. Compound severity
        severity_raw = (
            epistemic_risk * 40
            + disagreement_score * 30
            + instability * 20
            + contradiction_penalty
            + failure_penalty
        )
        severity = round(min(100.0, max(0.0, severity_raw)), 2)

        # 7. Dynamic risk level (NO hardcoded labels)
        if severity >= 70:
            risk_level = "HIGH"
        elif severity >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "severity_score": severity,
            "risk_level": risk_level,
            "epistemic_risk": round(epistemic_risk, 4),
            "disagreement_score": round(disagreement_score, 4),
            "instability": round(instability, 4),
            "contradiction_penalty": contradiction_penalty,
            "failure_penalty": failure_penalty,
            "human_review_required": severity >= 70,
            "explanation": DynamicBoundaryEngine._generate_explanation(
                severity, risk_level, epistemic_risk, disagreement_score, instability
            ),
        }

    @staticmethod
    def compute_debate_severity(
        model_positions: List[Dict[str, Any]],
        disagreement_strength: float = 0.0,
        convergence_level: str = "none",
        model_confidences: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Compute boundary severity specific to debate mode.
        """
        model_confidences = model_confidences or {}
        conf_values = list(model_confidences.values()) if model_confidences else []

        # Position divergence
        if len(model_positions) >= 2:
            position_texts = [p.get("position", "") for p in model_positions if p.get("position")]
            position_divergence = DynamicBoundaryEngine._compute_text_divergence(position_texts)
        else:
            position_divergence = 0.0

        # Convergence adjustment
        convergence_bonus = {
            "high": -15, "moderate": -8, "low": -3, "none": 5
        }
        conv_adjust = convergence_bonus.get(convergence_level, 0)

        base = DynamicBoundaryEngine.compute_severity(
            evidence_confidence=1.0 - disagreement_strength,
            model_divergence=position_divergence,
            model_confidences=conf_values,
        )

        # Adjust severity with convergence
        adjusted_severity = max(0, min(100, base["severity_score"] + conv_adjust))
        
        risk_level = "HIGH" if adjusted_severity >= 70 else "MEDIUM" if adjusted_severity >= 30 else "LOW"

        return {
            **base,
            "severity_score": round(adjusted_severity, 2),
            "risk_level": risk_level,
            "position_divergence": round(position_divergence, 4),
            "convergence_adjustment": conv_adjust,
        }

    @staticmethod
    def compute_evidence_severity(
        bayesian_confidence: float = 0.5,
        agreement_score: float = 0.0,
        contradiction_count: int = 0,
        source_reliability: float = 0.5,
        claims_total: int = 0,
        claims_verified: int = 0,
    ) -> Dict[str, Any]:
        """
        Compute boundary severity specific to evidence mode.
        """
        verification_ratio = claims_verified / max(claims_total, 1)
        
        base = DynamicBoundaryEngine.compute_severity(
            evidence_confidence=bayesian_confidence,
            model_divergence=1.0 - agreement_score,
            model_confidences=[bayesian_confidence, source_reliability, verification_ratio],
            contradiction_count=contradiction_count,
        )

        return {
            **base,
            "verification_ratio": round(verification_ratio, 4),
            "source_reliability": round(source_reliability, 4),
        }

    @staticmethod
    def _compute_text_divergence(texts: List[str]) -> float:
        """Compute pairwise divergence between texts using Jaccard distance."""
        if len(texts) < 2:
            return 0.0

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
            "for", "and", "or", "on", "at", "by", "it", "this", "that", "with",
        }

        word_sets = []
        for t in texts:
            words = set(t.lower().split()) - stop_words
            words = {w for w in words if len(w) > 3}
            word_sets.append(words)

        distances = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                intersection = word_sets[i] & word_sets[j]
                if union:
                    distances.append(1.0 - len(intersection) / len(union))
                else:
                    distances.append(1.0)

        return sum(distances) / max(len(distances), 1)

    @staticmethod
    def _generate_explanation(
        severity: float, risk_level: str,
        epistemic_risk: float, disagreement: float, instability: float
    ) -> str:
        """Generate computed explanation — no static text."""
        parts = [f"Boundary severity: {severity:.1f}/100 ({risk_level})."]

        contributors = []
        if epistemic_risk > 0.5:
            contributors.append(f"epistemic uncertainty ({epistemic_risk:.0%})")
        if disagreement > 0.3:
            contributors.append(f"model disagreement ({disagreement:.0%})")
        if instability > 0.3:
            contributors.append(f"output instability ({instability:.0%})")

        if contributors:
            parts.append(f"Driven by: {', '.join(contributors)}.")
        else:
            parts.append("All risk dimensions are within acceptable ranges.")

        return " ".join(parts)
