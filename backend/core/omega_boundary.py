"""
Omega Boundary Evaluator — Sentinel-E Omega Cognitive Kernel

Enhanced boundary evaluation that wraps the existing BoundaryDetector
with Omega-specific features:
- Multi-dimensional risk scoring
- Boundary trend analysis
- Human review escalation logic
- Integration with session intelligence
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.boundary_detector import BoundaryDetector

logger = logging.getLogger("Omega-Boundary")


class OmegaBoundaryEvaluator:
    """
    Enhanced boundary evaluation for the Omega Cognitive Kernel.
    Wraps BoundaryDetector with session-aware risk analysis.
    """

    def __init__(self):
        self.detector = BoundaryDetector()
        self.evaluation_history: List[Dict[str, Any]] = []

    def evaluate(self, text: str, context_observations: List[str] = None,
                 session_data: Dict = None) -> Dict[str, Any]:
        """
        Full boundary evaluation with Omega enhancements.
        
        Returns structured boundary assessment.
        """
        context_observations = context_observations or []
        session_data = session_data or {}

        # 1. Core boundary detection (existing system)
        core_result = self.detector.extract_boundaries(
            claim=text,
            available_observations=context_observations
        )

        # 2. Multi-dimensional risk scoring
        risk_dimensions = self._compute_risk_dimensions(text, core_result, session_data)

        # 3. Aggregate risk level
        severity_score = core_result.get("severity_score", 0)
        compound_severity = self._compute_compound_severity(severity_score, risk_dimensions)

        risk_level = "LOW"
        if compound_severity >= 70:
            risk_level = "HIGH"
        elif compound_severity >= 40:
            risk_level = "MEDIUM"

        # 4. Construct evaluation result
        evaluation = {
            "boundary_id": core_result.get("boundary_id"),
            "timestamp": datetime.utcnow().isoformat(),
            "claim": text,
            "claim_type": core_result.get("claim_type", "unknown"),
            "risk_level": risk_level,
            "severity_score": compound_severity,
            "core_severity": severity_score,
            "risk_dimensions": risk_dimensions,
            "grounding_score": core_result.get("grounding_score", 0),
            "missing_grounding": core_result.get("missing_grounding", []),
            "human_review_required": compound_severity >= 70,
            "explanation": self._generate_explanation(risk_level, compound_severity, risk_dimensions),
        }

        # 5. Track history
        self.evaluation_history.append(evaluation)

        return evaluation

    def evaluate_debate_boundaries(self, model_positions: List[Dict],
                                    agreements: List[str],
                                    disagreements: List[str]) -> Dict[str, Any]:
        """
        Evaluate boundaries across multi-model debate results.
        Used in EXPERIMENTAL mode.
        """
        boundary_assessments = []

        for position in model_positions:
            pos_text = position.get("position", "") or position.get("answer", "")
            if pos_text:
                assessment = self.evaluate(pos_text, context_observations=[])
                boundary_assessments.append({
                    "model": position.get("model", "unknown"),
                    "severity_score": assessment["severity_score"],
                    "risk_level": assessment["risk_level"],
                    "claim_type": assessment["claim_type"],
                })

        # Compute aggregate debate boundary metrics
        if boundary_assessments:
            avg_severity = sum(a["severity_score"] for a in boundary_assessments) / len(boundary_assessments)
            max_severity = max(a["severity_score"] for a in boundary_assessments)
        else:
            avg_severity = 0
            max_severity = 0

        # Disagreement boundary amplification
        disagreement_penalty = len(disagreements) * 5
        amplified_severity = min(100, int(avg_severity + disagreement_penalty))

        return {
            "per_model_boundaries": boundary_assessments,
            "average_severity": round(avg_severity, 2),
            "max_severity": max_severity,
            "disagreement_amplification": disagreement_penalty,
            "compound_severity": amplified_severity,
            "risk_level": "HIGH" if amplified_severity >= 70 else "MEDIUM" if amplified_severity >= 40 else "LOW",
            "explanation": f"Debate boundary analysis: {len(boundary_assessments)} positions evaluated. "
                          f"Avg severity: {avg_severity:.1f}, Disagreement penalty: +{disagreement_penalty}.",
        }

    def get_trend(self) -> Dict[str, Any]:
        """Analyze boundary severity trend across evaluations."""
        if len(self.evaluation_history) < 2:
            return {"trend": "insufficient_data", "data_points": len(self.evaluation_history)}

        scores = [e["severity_score"] for e in self.evaluation_history]
        recent = scores[-5:]

        if recent[-1] > recent[0] + 10:
            trend = "escalating"
        elif recent[-1] < recent[0] - 10:
            trend = "stabilizing"
        else:
            trend = "flat"

        return {
            "trend": trend,
            "latest_severity": recent[-1],
            "average_severity": round(sum(recent) / len(recent), 2),
            "data_points": len(scores),
        }

    # ============================================================
    # INTERNAL METHODS
    # ============================================================

    def _compute_risk_dimensions(self, text: str, core: Dict, session: Dict) -> Dict[str, float]:
        """Compute multi-dimensional risk scores."""
        text_lower = text.lower()

        dimensions = {}

        # Epistemic risk: How uncertain is the claim?
        grounding = core.get("grounding_score", 0)
        dimensions["epistemic_risk"] = round(max(0, 1.0 - grounding / 100), 4)

        # Domain sensitivity risk
        sensitive_domains = {
            "medical": ["medical", "health", "diagnosis", "treatment", "drug", "patient"],
            "legal": ["legal", "law", "court", "liability", "regulation", "statute"],
            "financial": ["financial", "investment", "stock", "trade", "portfolio", "market"],
            "safety": ["safety", "danger", "hazard", "critical", "life-threatening"],
        }
        max_domain_risk = 0.0
        for domain, keywords in sensitive_domains.items():
            if any(kw in text_lower for kw in keywords):
                max_domain_risk = max(max_domain_risk, 0.8)
        dimensions["domain_sensitivity"] = max_domain_risk

        # Session instability risk
        session_fragility = session.get("fragility_index", 0)
        dimensions["session_instability"] = round(session_fragility, 4)

        # Complexity risk
        word_count = len(text.split())
        if word_count > 200:
            dimensions["complexity_risk"] = 0.7
        elif word_count > 100:
            dimensions["complexity_risk"] = 0.4
        else:
            dimensions["complexity_risk"] = 0.1

        return dimensions

    def _compute_compound_severity(self, base_severity: int, dimensions: Dict[str, float]) -> int:
        """Compute compound severity from base + risk dimensions."""
        # Weighted combination
        weights = {
            "epistemic_risk": 0.3,
            "domain_sensitivity": 0.35,
            "session_instability": 0.15,
            "complexity_risk": 0.2,
        }

        dimension_score = sum(
            dimensions.get(dim, 0) * weight
            for dim, weight in weights.items()
        ) * 100

        # Compound: 60% core, 40% dimensional analysis
        compound = int(base_severity * 0.6 + dimension_score * 0.4)
        return min(100, max(0, compound))

    def _generate_explanation(self, risk_level: str, severity: int,
                               dimensions: Dict[str, float]) -> str:
        """Generate human-readable boundary explanation."""
        parts = []

        if risk_level == "HIGH":
            parts.append(f"High boundary risk (severity {severity}/100).")
        elif risk_level == "MEDIUM":
            parts.append(f"Moderate boundary risk (severity {severity}/100).")
        else:
            parts.append(f"Low boundary risk (severity {severity}/100).")

        # Top contributing dimensions
        sorted_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)
        top_contrib = sorted_dims[:2]
        for dim, val in top_contrib:
            if val > 0.3:
                dim_name = dim.replace("_", " ").title()
                parts.append(f"{dim_name}: {val:.2f}.")

        return " ".join(parts)
