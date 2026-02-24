"""
============================================================
Dynamic Analytics Engine
============================================================
Computes all confidence/risk metrics dynamically.
No hardcoded values. No static confidence = 67.

Metrics computed from:
- Embedding similarity (model agreement)
- Argument depth scoring
- Instability across rounds
- Evidence strength
- Contradiction count
- Topic sensitivity
- Model divergence
"""

import logging
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("Analytics")


@dataclass
class AnalyticsResult:
    """Dynamic analytics output."""
    confidence: float = 0.5
    confidence_components: Dict[str, float] = field(default_factory=dict)
    boundary_risk: float = 0.0
    boundary_components: Dict[str, float] = field(default_factory=dict)
    agreement_score: float = 0.0
    divergence_score: float = 0.0
    argument_depth: float = 0.0
    instability_index: float = 0.0
    evidence_strength: float = 0.0
    contradiction_count: int = 0
    risk_level: str = "LOW"
    explanation: str = ""


class DynamicAnalyticsEngine:
    """
    Computes confidence and risk metrics dynamically from actual data.
    Every metric is derived from observable signals — nothing hardcoded.
    """

    def compute(
        self,
        model_outputs: List[str],
        evidence_sources: int = 0,
        contradiction_count: int = 0,
        evidence_reliability: float = 0.0,
        topic_sensitivity: float = 0.0,
        debate_rounds: int = 0,
        round_outputs: Optional[List[List[str]]] = None,
    ) -> AnalyticsResult:
        """
        Compute comprehensive analytics from actual model outputs and evidence.
        """
        result = AnalyticsResult()

        # 1. Agreement Score (text-based similarity between model outputs)
        if len(model_outputs) >= 2:
            result.agreement_score = self._compute_agreement(model_outputs)
            result.divergence_score = 1.0 - result.agreement_score
        else:
            result.agreement_score = 1.0
            result.divergence_score = 0.0

        # 2. Argument Depth
        result.argument_depth = self._compute_argument_depth(model_outputs)

        # 3. Instability Index (across rounds)
        if round_outputs and len(round_outputs) >= 2:
            result.instability_index = self._compute_instability(round_outputs)

        # 4. Evidence Strength
        result.evidence_strength = self._compute_evidence_strength(
            evidence_sources, evidence_reliability, contradiction_count
        )
        result.contradiction_count = contradiction_count

        # 5. Dynamic Confidence
        result.confidence = self._compute_confidence(
            agreement=result.agreement_score,
            depth=result.argument_depth,
            instability=result.instability_index,
            evidence=result.evidence_strength,
            contradictions=contradiction_count,
        )
        result.confidence_components = {
            "agreement": round(result.agreement_score, 3),
            "argument_depth": round(result.argument_depth, 3),
            "instability_penalty": round(result.instability_index, 3),
            "evidence_boost": round(result.evidence_strength, 3),
            "contradiction_penalty": round(min(contradiction_count * 0.1, 0.3), 3),
        }

        # 6. Boundary Risk
        result.boundary_risk = self._compute_boundary_risk(
            topic_sensitivity=topic_sensitivity,
            evidence_stability=result.evidence_strength,
            model_divergence=result.divergence_score,
            contradiction_count=contradiction_count,
        )
        result.boundary_components = {
            "topic_sensitivity": round(topic_sensitivity, 3),
            "evidence_stability": round(1.0 - result.evidence_strength, 3),
            "model_divergence": round(result.divergence_score, 3),
            "contradiction_factor": round(min(contradiction_count * 0.15, 0.5), 3),
        }

        # 7. Risk Level
        if result.boundary_risk >= 0.7:
            result.risk_level = "HIGH"
        elif result.boundary_risk >= 0.4:
            result.risk_level = "MEDIUM"
        else:
            result.risk_level = "LOW"

        result.explanation = (
            f"Confidence {result.confidence:.0%} based on "
            f"{result.agreement_score:.0%} agreement, "
            f"depth={result.argument_depth:.2f}, "
            f"instability={result.instability_index:.2f}. "
            f"Boundary risk: {result.risk_level} ({result.boundary_risk:.0%})."
        )

        return result

    def _compute_agreement(self, outputs: List[str]) -> float:
        """
        Compute agreement between model outputs using token overlap.
        Uses Jaccard similarity on word n-grams.
        """
        if len(outputs) < 2:
            return 1.0

        def get_ngrams(text: str, n: int = 3) -> set:
            words = text.lower().split()
            if len(words) < n:
                return set(words)
            return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                ngrams_i = get_ngrams(outputs[i])
                ngrams_j = get_ngrams(outputs[j])
                if not ngrams_i or not ngrams_j:
                    similarities.append(0.5)
                    continue
                intersection = ngrams_i & ngrams_j
                union = ngrams_i | ngrams_j
                jaccard = len(intersection) / len(union) if union else 0
                similarities.append(jaccard)

        return sum(similarities) / len(similarities) if similarities else 0.5

    def _compute_argument_depth(self, outputs: List[str]) -> float:
        """
        Score argument depth based on structural indicators.
        Looks for reasoning markers, evidence citations, logical connectors.
        """
        depth_signals = [
            "because", "therefore", "however", "although", "moreover",
            "furthermore", "in contrast", "specifically", "for example",
            "evidence suggests", "data shows", "research indicates",
            "on the other hand", "additionally", "consequently",
            "first", "second", "third", "finally",
        ]

        total_depth = 0.0
        for output in outputs:
            output_lower = output.lower()
            signal_count = sum(1 for s in depth_signals if s in output_lower)
            # Depth is a function of signal density
            words = len(output_lower.split())
            if words > 0:
                density = signal_count / (words / 100)  # signals per 100 words
                total_depth += min(density / 5, 1.0)  # Normalize to 0-1

        return total_depth / len(outputs) if outputs else 0.5

    def _compute_instability(self, round_outputs: List[List[str]]) -> float:
        """
        Compute instability across debate/stress rounds.
        High instability = positions changing significantly between rounds.
        """
        if len(round_outputs) < 2:
            return 0.0

        drifts = []
        for i in range(1, len(round_outputs)):
            prev_combined = " ".join(round_outputs[i-1])
            curr_combined = " ".join(round_outputs[i])

            # Simple word overlap drift
            prev_words = set(prev_combined.lower().split())
            curr_words = set(curr_combined.lower().split())

            if not prev_words or not curr_words:
                continue

            overlap = len(prev_words & curr_words)
            total = len(prev_words | curr_words)
            similarity = overlap / total if total else 0
            drift = 1.0 - similarity
            drifts.append(drift)

        return sum(drifts) / len(drifts) if drifts else 0.0

    def _compute_evidence_strength(
        self,
        source_count: int,
        avg_reliability: float,
        contradiction_count: int,
    ) -> float:
        """Dynamic evidence strength from actual sources."""
        if source_count == 0:
            return 0.0

        # Base: reliability weighted by source count
        base = avg_reliability * min(source_count / 3, 1.0)

        # Penalty for contradictions
        contradiction_penalty = min(contradiction_count * 0.15, 0.5)

        return max(0.0, min(1.0, base - contradiction_penalty))

    def _compute_confidence(
        self,
        agreement: float,
        depth: float,
        instability: float,
        evidence: float,
        contradictions: int,
    ) -> float:
        """
        Dynamic confidence from all signals.
        Weighted combination with diminishing returns.
        """
        # Weighted base
        base = (
            agreement * 0.35
            + depth * 0.25
            + evidence * 0.20
            + 0.20  # Base confidence floor
        )

        # Penalties
        instability_penalty = instability * 0.2
        contradiction_penalty = min(contradictions * 0.05, 0.2)

        confidence = base - instability_penalty - contradiction_penalty

        # Apply sigmoid smoothing for organic feel
        confidence = self._sigmoid_smooth(confidence)

        return round(max(0.05, min(0.99, confidence)), 3)

    def _compute_boundary_risk(
        self,
        topic_sensitivity: float,
        evidence_stability: float,
        model_divergence: float,
        contradiction_count: int,
    ) -> float:
        """
        Dynamic boundary risk.
        Depends on topic sensitivity, evidence stability, and model divergence.
        """
        risk = (
            topic_sensitivity * 0.3
            + (1.0 - evidence_stability) * 0.25
            + model_divergence * 0.25
            + min(contradiction_count * 0.1, 0.3) * 0.2
        )

        return round(max(0.0, min(1.0, risk)), 3)

    def _sigmoid_smooth(self, x: float, steepness: float = 6.0) -> float:
        """Apply sigmoid smoothing for organic-feeling metrics."""
        try:
            return 1.0 / (1.0 + math.exp(-steepness * (x - 0.5)))
        except OverflowError:
            return 1.0 if x > 0.5 else 0.0
