"""
============================================================
Confidence Calibrator — Entropy-Weighted Ensemble Confidence
============================================================
Computes system confidence ENTIRELY from ensemble metrics.

Zero dependence on model self-reported confidence.
Uses: disagreement entropy, contradiction density,
stability index, consensus velocity, fragility score.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List

from backend.core.ensemble_schemas import (
    CalibratedConfidence,
    EnsembleMetrics,
    StructuredModelOutput,
    DebateResult,
    AgreementMatrix,
)

logger = logging.getLogger("ConfidenceCalibrator")


class ConfidenceCalibrator:
    """
    Derives system confidence from ensemble dynamics.

    Formula:
        base = agreement_signal × (1 - entropy_penalty)
        stability_bonus = stability_index × weight
        contradiction_penalty = contradiction_density × weight
        fragility_penalty = fragility_score × weight
        velocity_bonus = consensus_velocity × weight (if positive)

        raw = base + stability_bonus + velocity_bonus
              - contradiction_penalty - fragility_penalty
        final = sigmoid_smooth(clamp(raw, 0.05, 0.95))

    Every component is exposed in the output for frontend display.
    """

    # Component weights
    W_AGREEMENT = 0.35
    W_ENTROPY = 0.25
    W_STABILITY = 0.15
    W_CONTRADICTION = 0.15
    W_FRAGILITY = 0.05
    W_VELOCITY = 0.05

    def calibrate(
        self,
        metrics: EnsembleMetrics,
        matrix: AgreementMatrix,
        debate: DebateResult,
        outputs: List[StructuredModelOutput],
    ) -> CalibratedConfidence:
        """
        Compute calibrated confidence from ensemble metrics.

        Returns CalibratedConfidence with full component breakdown.
        """
        # ── Component: Agreement Signal ─────────────────
        agreement_signal = matrix.mean_agreement
        if len(matrix.dissenting_models) > len(outputs) / 2:
            agreement_signal *= 0.6  # Majority dissent penalty

        # ── Component: Entropy Penalty ──────────────────
        # Higher entropy = more disagreement = lower confidence
        entropy_penalty = metrics.disagreement_entropy * self.W_ENTROPY

        # ── Component: Stability Bonus ──────────────────
        stability_bonus = metrics.stability_index * self.W_STABILITY

        # ── Component: Contradiction Penalty ────────────
        contradiction_penalty = metrics.contradiction_density * self.W_CONTRADICTION

        # ── Component: Fragility Penalty ────────────────
        fragility_penalty = metrics.fragility_score * self.W_FRAGILITY

        # ── Component: Consensus Velocity Bonus ─────────
        velocity_bonus = 0.0
        if metrics.consensus_velocity > 0:
            velocity_bonus = min(metrics.consensus_velocity, 0.3) * self.W_VELOCITY

        # ── Base Computation ────────────────────────────
        base = agreement_signal * self.W_AGREEMENT * (1.0 - entropy_penalty)

        # ── Combine ─────────────────────────────────────
        raw = (
            base
            + stability_bonus
            + velocity_bonus
            - contradiction_penalty
            - fragility_penalty
        )

        # ── Model Count Adjustment ──────────────────────
        # More models = more reliable signal
        model_factor = min(metrics.successful_models / 5.0, 1.0)
        raw = raw * (0.7 + 0.3 * model_factor)

        # ── Confidence Spread Penalty ───────────────────
        # Wide spread in model self-confidence = uncertainty
        if metrics.confidence_spread > 0.5:
            raw *= (1.0 - (metrics.confidence_spread - 0.5) * 0.3)

        # ── Round Count Bonus ───────────────────────────
        # More debate rounds = more refined signal
        round_bonus = min((metrics.round_count - 3) * 0.02, 0.06)
        raw += max(0, round_bonus)

        # ── Sigmoid Smoothing ───────────────────────────
        final = self._sigmoid_smooth(raw)
        final = max(0.05, min(0.95, final))

        # ── Evolution Tracking ──────────────────────────
        initial_confidence = agreement_signal * 0.5
        post_debate = initial_confidence + stability_bonus - contradiction_penalty
        post_debate = max(0.05, min(0.95, post_debate))

        evolution = [
            {"stage": "initial", "value": round(initial_confidence, 4)},
            {"stage": "post_debate", "value": round(post_debate, 4)},
            {"stage": "post_calibration", "value": round(final, 4)},
        ]

        # ── Components Breakdown ────────────────────────
        components = {
            "agreement_signal": round(agreement_signal, 4),
            "entropy_penalty": round(entropy_penalty, 4),
            "stability_bonus": round(stability_bonus, 4),
            "contradiction_penalty": round(contradiction_penalty, 4),
            "fragility_penalty": round(fragility_penalty, 4),
            "velocity_bonus": round(velocity_bonus, 4),
            "model_factor": round(model_factor, 4),
            "confidence_spread_penalty": round(
                max(0, (metrics.confidence_spread - 0.5) * 0.3), 4
            ),
            "round_bonus": round(max(0, round_bonus), 4),
            "raw_score": round(raw, 4),
        }

        # ── Explanation ─────────────────────────────────
        explanation = self._generate_explanation(
            final, metrics, matrix, components
        )

        return CalibratedConfidence(
            final_confidence=round(final, 4),
            calibration_method="entropy_weighted_ensemble",
            components=components,
            evolution=evolution,
            explanation=explanation,
        )

    def compute_ensemble_metrics(
        self,
        outputs: List[StructuredModelOutput],
        matrix: AgreementMatrix,
        debate: DebateResult,
        disagreement_entropy: float,
        contradiction_density: float,
    ) -> EnsembleMetrics:
        """
        Build EnsembleMetrics from all available data.
        """
        successful = [o for o in outputs if o.succeeded]
        failed = [o for o in outputs if not o.succeeded]

        # Model confidence stats
        confidences = [o.confidence for o in successful]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        conf_spread = (max(confidences) - min(confidences)) if len(confidences) > 1 else 0.0

        # Stability: How much positions change across debate rounds
        stability = self._compute_stability(debate)

        # Consensus velocity: Rate of convergence
        velocity = self._compute_velocity(debate)

        # Fragility: How sensitive is the consensus
        fragility = self._compute_fragility(
            matrix, disagreement_entropy, stability, conf_spread
        )

        return EnsembleMetrics(
            disagreement_entropy=round(disagreement_entropy, 4),
            contradiction_density=round(contradiction_density, 4),
            stability_index=round(stability, 4),
            consensus_velocity=round(velocity, 4),
            fragility_score=round(fragility, 4),
            model_count=len(outputs),
            round_count=debate.total_rounds,
            successful_models=len(successful),
            failed_models=len(failed),
            mean_model_confidence=round(mean_conf, 4),
            confidence_spread=round(conf_spread, 4),
        )

    # ── Internal Computations ────────────────────────────────

    def _compute_stability(self, debate: DebateResult) -> float:
        """
        How stable are positions across debate rounds?
        1.0 = fully stable (no shifts), 0.0 = completely unstable.
        """
        if not debate.rounds or len(debate.rounds) < 2:
            return 0.5  # Neutral when insufficient data

        total_positions = 0
        shifts = 0

        for round_data in debate.rounds[1:]:  # Skip round 1
            for pos in round_data.positions:
                total_positions += 1
                if pos.position_shifted:
                    shifts += 1

        if total_positions == 0:
            return 1.0

        shift_rate = shifts / total_positions
        return max(0.0, 1.0 - shift_rate)

    def _compute_velocity(self, debate: DebateResult) -> float:
        """
        Rate of convergence across rounds.
        Positive = converging, negative = diverging.
        """
        if not debate.rounds or len(debate.rounds) < 2:
            return 0.0

        disagreements = [r.round_disagreement for r in debate.rounds]
        if len(disagreements) < 2:
            return 0.0

        # Compute slope of disagreement over rounds
        deltas = [
            disagreements[i] - disagreements[i - 1]
            for i in range(1, len(disagreements))
        ]
        avg_delta = sum(deltas) / len(deltas)

        # Negative delta in disagreement = convergence = positive velocity
        return -avg_delta

    def _compute_fragility(
        self,
        matrix: AgreementMatrix,
        entropy: float,
        stability: float,
        conf_spread: float,
    ) -> float:
        """
        How fragile is the consensus?
        High fragility = small perturbation could break consensus.
        """
        # Fragility increases with:
        # - Low agreement variance (barely above threshold)
        # - High entropy
        # - Low stability
        # - High confidence spread

        agreement_fragility = 0.0
        if matrix.mean_agreement > 0:
            # Close to 0.5 = very fragile
            agreement_fragility = 1.0 - abs(matrix.mean_agreement - 0.5) * 2

        fragility = (
            0.30 * agreement_fragility
            + 0.30 * entropy
            + 0.20 * (1.0 - stability)
            + 0.20 * min(conf_spread, 1.0)
        )

        return max(0.0, min(1.0, fragility))

    def _sigmoid_smooth(self, x: float) -> float:
        """Smooth clipping via sigmoid centered at 0.5."""
        # Map x to roughly [0, 1] via logistic function
        # Shift and scale so 0.5 input → 0.5 output
        try:
            return 1.0 / (1.0 + math.exp(-6 * (x - 0.5)))
        except OverflowError:
            return 0.0 if x < 0.5 else 1.0

    def _generate_explanation(
        self,
        confidence: float,
        metrics: EnsembleMetrics,
        matrix: AgreementMatrix,
        components: Dict[str, float],
    ) -> str:
        """Generate human-readable confidence explanation."""
        parts = []

        # Level description
        if confidence >= 0.8:
            parts.append("Strong ensemble agreement.")
        elif confidence >= 0.6:
            parts.append("Moderate ensemble agreement with some divergence.")
        elif confidence >= 0.4:
            parts.append("Mixed signals across models.")
        else:
            parts.append("Significant disagreement across models.")

        # Notable factors
        if metrics.disagreement_entropy > 0.6:
            parts.append(
                f"High position entropy ({metrics.disagreement_entropy:.2f}) "
                f"indicates diverse viewpoints."
            )
        if metrics.contradiction_density > 0.3:
            parts.append(
                f"Contradiction density ({metrics.contradiction_density:.2f}) "
                f"suggests conflicting evidence."
            )
        if metrics.stability_index > 0.8:
            parts.append("Positions remained stable through debate.")
        elif metrics.stability_index < 0.4:
            parts.append("Significant position shifts during debate.")

        if matrix.dissenting_models:
            parts.append(
                f"Dissenting models: {', '.join(matrix.dissenting_models)}."
            )

        parts.append(
            f"Based on {metrics.successful_models} models "
            f"across {metrics.round_count} debate rounds."
        )

        return " ".join(parts)
