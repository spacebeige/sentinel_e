"""
Boundary Detector: Systematic identification of epistemic and operational boundaries.

Operates ONLY in Sentinel-Σ (Experimental Scope).
Detect required grounding, infer boundary violations, emit structured output.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uuid


class BoundaryDetector:
    """
    Detects boundaries (required grounding) for atomic claims.
    Compares requirements against available observations.
    Emits structured boundary violations.
    """

    def __init__(self):
        # Define grounding requirements for claim types
        self.grounding_requirements = {
            "causal_claim": ["temporal_precedence", "mechanism", "correlation"],
            "factual_claim": ["source_citation", "verifiability", "corroboration"],
            "predictive_claim": ["evidence_base", "model_specification", "confidence_bounds"],
            "normative_claim": ["value_framework", "stakeholder_context", "ethical_grounding"],
            "technical_claim": ["specification", "reproducibility", "validation"],
            "medical_claim": ["clinical_evidence", "statistical_power", "contraindications"],
            "legal_claim": ["jurisdiction", "precedent", "statutory_basis"],
            "scientific_claim": ["peer_review_status", "methodology", "falsifiability"],
        }

        # Severity levels (0-100)
        self.severity_scale = {
            "critical": 90,      # Requires immediate human review
            "high": 70,          # Significant grounding gap
            "medium": 50,        # Substantial grounding gap
            "low": 30,           # Minor grounding gap
            "minimal": 10,       # Fully grounded
        }

    def classify_claim(self, claim: str) -> str:
        """
        Infer claim type from linguistic and semantic markers.
        """
        claim_lower = claim.lower()

        if any(w in claim_lower for w in ["because", "caused", "caused by", "leads to", "results in"]):
            return "causal_claim"
        elif any(w in claim_lower for w in ["it is true that", "fact is", "research shows", "studies indicate"]):
            return "factual_claim"
        elif any(w in claim_lower for w in ["will", "next", "predict", "forecast", "expect"]):
            return "predictive_claim"
        elif any(w in claim_lower for w in ["should", "ought", "must", "better", "worse", "right", "wrong"]):
            return "normative_claim"
        elif any(w in claim_lower for w in ["algorithm", "code", "parameter", "config", "API", "protocol"]):
            return "technical_claim"
        elif any(w in claim_lower for w in ["patient", "disease", "treatment", "drug", "symptoms", "medical"]):
            return "medical_claim"
        elif any(w in claim_lower for w in ["law", "court", "legal", "statute", "regulation", "comply"]):
            return "legal_claim"
        else:
            return "scientific_claim"

    def infer_grounding_requirements(self, claim: str) -> List[str]:
        """
        Infer what grounding is needed for a claim.
        """
        claim_type = self.classify_claim(claim)
        return self.grounding_requirements.get(claim_type, ["general_evidence"])

    def extract_boundaries(self, claim: str, available_observations: List[str]) -> Dict[str, Any]:
        """
        Compare claim against available observations.
        Identify missing or insufficient grounding.
        
        Returns structured boundary violation object.
        """
        claim_type = self.classify_claim(claim)
        required = self.infer_grounding_requirements(claim)

        # Score available observations against requirements
        grounding_score = self._score_grounding(claim, available_observations, required)

        # Determine severity
        if grounding_score >= 80:
            severity_level = "minimal"
            severity_score = self.severity_scale["minimal"]
        elif grounding_score >= 60:
            severity_level = "low"
            severity_score = self.severity_scale["low"]
        elif grounding_score >= 40:
            severity_level = "medium"
            severity_score = self.severity_scale["medium"]
        elif grounding_score >= 20:
            severity_level = "high"
            severity_score = self.severity_scale["high"]
        else:
            severity_level = "critical"
            severity_score = self.severity_scale["critical"]

        boundary_violation = {
            "boundary_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "claim": claim,
            "claim_type": claim_type,
            "severity_level": severity_level,
            "severity_score": severity_score,
            "grounding_score": round(grounding_score, 2),
            "required_grounding": required,
            "available_observations_count": len(available_observations),
            "missing_grounding": self._infer_missing_grounding(required, available_observations),
            "human_review_required": severity_score >= 70,
        }

        return boundary_violation

    def _score_grounding(self, claim: str, observations: List[str], requirements: List[str]) -> float:
        """
        Heuristic scoring of grounding completeness.
        Higher = better grounded.
        """
        if not observations:
            return 0.0

        # Simple keyword matching heuristic
        claim_words = set(claim.lower().split())
        observation_words = set()
        for obs in observations:
            observation_words.update(obs.lower().split())

        # Overlap score
        intersection = len(claim_words & observation_words)
        union = len(claim_words | observation_words)
        overlap_score = (intersection / union) * 100 if union > 0 else 0.0

        # Requirement coverage heuristic
        requirement_words = set()
        for req in requirements:
            requirement_words.update(req.lower().split())

        req_coverage = len(requirement_words & observation_words) / len(requirement_words) if requirement_words else 0.0

        # Weighted average
        grounding_score = (overlap_score * 0.6 + req_coverage * 40) * 0.01
        return min(100, grounding_score * 100)

    def _infer_missing_grounding(self, required: List[str], observations: List[str]) -> List[str]:
        """
        Infer which grounding elements are absent.
        """
        observation_text = " ".join(observations).lower()
        missing = []

        for req in required:
            req_lower = req.lower()
            # Simple heuristic: if requirement keyword not in observations, it's missing
            if not any(keyword in observation_text for keyword in req_lower.split("_")):
                missing.append(req)

        return missing

    def aggregate_boundary_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple boundary violations.
        Compute cumulative severity.
        """
        if not violations:
            return {
                "cumulative_severity": 0.0,
                "violation_count": 0,
                "max_severity": "minimal",
                "violations": [],
            }

        severity_scores = [v["severity_score"] for v in violations]
        severity_levels = [v["severity_level"] for v in violations]

        # Cumulative severity (cap at 100)
        cumulative = min(100, sum(severity_scores) / len(severity_scores))

        # Max severity from list
        max_severity = max(violations, key=lambda v: v["severity_score"])["severity_level"]

        return {
            "cumulative_severity": round(cumulative, 2),
            "violation_count": len(violations),
            "max_severity": max_severity,
            "mean_severity_score": round(sum(severity_scores) / len(severity_scores), 2),
            "violations": violations,
            "human_review_required": cumulative >= 70,
        }

    def check_boundary_threshold(self, cumulative_severity: float, threshold: float = 70.0) -> bool:
        """
        Determine if cumulative severity exceeds refusal threshold.
        Used by Standard mode to decide refusal.
        
        Returns True if SHOULD REFUSE (severity >= threshold).
        """
        return cumulative_severity >= threshold
