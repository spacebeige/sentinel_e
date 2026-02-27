"""
============================================================
Drift Tracker — Sentinel-E Cognitive Engine v7.0
============================================================
Computes per-model stance drift across debate rounds.

drift = cosine_distance(stance_vector_round1, stance_vector_roundN)

Tracks:
  - Total drift per model (R1 → Rfinal)
  - Per-round incremental drift
  - Direction: converging / diverging / stable
  - Aggregate metrics: mean, max, min, convergence detection
============================================================
"""

from __future__ import annotations

import math
from typing import Dict, List

from core.ensemble_schemas import (
    DriftMetrics,
    ModelDrift,
    StanceVector,
    StructuredModelOutput,
)


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance = 1 - cosine_similarity. Range [0, 2]."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    similarity = dot / (norm_a * norm_b)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _determine_direction(per_round_drifts: List[float]) -> str:
    """Determine if a model is converging, diverging, or stable."""
    if len(per_round_drifts) < 2:
        return "stable"
    # Look at trend: are drifts decreasing (converging) or increasing (diverging)?
    increasing = sum(1 for i in range(1, len(per_round_drifts))
                     if per_round_drifts[i] > per_round_drifts[i-1])
    decreasing = sum(1 for i in range(1, len(per_round_drifts))
                     if per_round_drifts[i] < per_round_drifts[i-1])
    if decreasing > increasing:
        return "converging"
    elif increasing > decreasing:
        return "diverging"
    return "stable"


class DriftTracker:
    """
    Tracks stance vector drift across debate rounds for all models.

    Usage:
        tracker = DriftTracker()
        tracker.record_round(round_number=1, outputs=[...])
        tracker.record_round(round_number=2, outputs=[...])
        metrics = tracker.compute()
    """

    def __init__(self):
        # {model_id: [StanceVector per round]}
        self._stances: Dict[str, List[StanceVector]] = {}

    def record_round(
        self,
        round_number: int,
        outputs: List[StructuredModelOutput],
    ) -> None:
        """Record stance vectors from a debate round."""
        for out in outputs:
            if out.model_id not in self._stances:
                self._stances[out.model_id] = []

            # Always record — even for failed models (uses default stance)
            stance = out.stance_vector if out.stance_vector else StanceVector()
            self._stances[out.model_id].append(stance)

    def compute(self) -> DriftMetrics:
        """Compute drift metrics across all recorded rounds."""
        model_drifts: List[ModelDrift] = []
        all_total_drifts: List[float] = []

        for model_id, stances in self._stances.items():
            if len(stances) < 2:
                model_drifts.append(ModelDrift(
                    model_id=model_id,
                    round_stances=stances,
                    total_drift=0.0,
                    per_round_drift=[],
                    drift_direction="stable",
                ))
                all_total_drifts.append(0.0)
                continue

            # Per-round incremental drift
            per_round = []
            for i in range(1, len(stances)):
                d = _cosine_distance(
                    stances[i-1].to_vector(),
                    stances[i].to_vector(),
                )
                per_round.append(d)

            # Total drift: R1 → Rfinal
            total = _cosine_distance(
                stances[0].to_vector(),
                stances[-1].to_vector(),
            )

            direction = _determine_direction(per_round)

            model_drifts.append(ModelDrift(
                model_id=model_id,
                round_stances=stances,
                total_drift=total,
                per_round_drift=per_round,
                drift_direction=direction,
            ))
            all_total_drifts.append(total)

        if not all_total_drifts:
            return DriftMetrics()

        mean_drift = sum(all_total_drifts) / len(all_total_drifts)
        max_drift = max(all_total_drifts)
        min_drift = min(all_total_drifts)

        # Convergence: most models are converging and mean drift is meaningful
        converging_count = sum(1 for md in model_drifts if md.drift_direction == "converging")
        convergence_detected = converging_count > len(model_drifts) / 2 and mean_drift > 0.05

        directions = [md.drift_direction for md in model_drifts]
        if directions.count("converging") > directions.count("diverging"):
            overall_dir = "converging"
        elif directions.count("diverging") > directions.count("converging"):
            overall_dir = "diverging"
        else:
            overall_dir = "stable"

        return DriftMetrics(
            model_drifts=model_drifts,
            mean_drift=mean_drift,
            max_drift=max_drift,
            min_drift=min_drift,
            convergence_detected=convergence_detected,
            overall_direction=overall_dir,
        )
