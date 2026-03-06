"""
============================================================
ELO Ranking Engine — Sentinel-E Battle Platform v2
============================================================
Maintains a persistent ELO leaderboard across all model debates.

Why ELO for model comparisons:
    ELO was designed for round-robin tournaments where not every
    pair competes in every round — exactly the structure of Sentinel-E
    debates, where dynamic model selection means different subsets
    compete per prompt.

    Key properties that make ELO ideal here:
      1. Sparse comparison graphs: A model only needs to have competed
         in a few debates to have a stable score. It does not need to
         face every other model to be ranked.
      2. Self-correcting: A new model that consistently beats highly-
         rated opponents rises quickly. A strong model that loses a
         surprise defeats falls proportionally.
      3. Interpretable: A 100-point gap means the higher-rated model
         wins approximately 64% of pairwise encounters.
      4. No ground-truth required: Rankings emerge from relative
         performance, not absolute task correctness — exactly what
         a debate-based evaluation platform produces.

    Variant used: Glicko-style K-factor decay.
      - New models (< 30 debates): K = 40  (fast calibration)
      - Mid-tier (30–100 debates): K = 20  (stable improvement)
      - Veteran (> 100 debates):   K = 10  (conservative update)

    Multi-model debates:
      In a 4-model debate, the winner faces each loser in a pairwise
      comparison (3 battles). Draws are counted when consensus
      stability > 0.85 (models converged on the same position).

Persistence:
    Rankings are stored in backend/data/elo_rankings.json.
    The file is loaded at startup and saved after every update.
    Thread-safe write via atomic rename.

Storage format:
    {
      "last_updated": "...",
      "total_debates": 0,
      "rankings": {
        "llama-3.3": { "elo_score": 1560, "wins": 42, ... }
      }
    }

Module:  ranking/elo_engine.py
============================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.ensemble_schemas import ELORankingEntry, EvaluationRecord

logger = logging.getLogger("ELORankingEngine")

# ── ELO constants ─────────────────────────────────────────────
INITIAL_ELO        = 1200.0
K_NEW              = 40.0    # < 30 debates
K_MID              = 20.0    # 30–100 debates
K_VETERAN          = 10.0    # > 100 debates

# Thresholds
DRAW_STABILITY_THRESHOLD = 0.85   # consensus_stability > this → treat as draw

# Storage path
_DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
_ELO_PATH   = os.path.join(_DATA_DIR, "elo_rankings.json")


class ELORankingEngine:
    """
    Persistent ELO leaderboard for Sentinel-E model battles.

    Usage:
        engine = ELORankingEngine()
        # After a debate where llama-3.3 won:
        engine.record_battle(
            winner_id="llama-3.3",
            loser_ids=["mixtral-8x7b", "qwen2.5-32b"],
            reasoning_scores={"llama-3.3": 0.84, ...},
        )
        leaderboard = engine.get_leaderboard()
    """

    def __init__(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        self._rankings: Dict[str, ELORankingEntry] = {}
        self._total_debates: int = 0
        self._last_updated: str = ""
        self._load()
        self._seed_known_models()

    # ── Public interface ──────────────────────────────────────

    def record_battle(
        self,
        winner_id: str,
        loser_ids: List[str],
        reasoning_scores: Optional[Dict[str, float]] = None,
        is_draw: bool = False,
        consensus_stability: float = 0.0,
    ) -> None:
        """
        Record the outcome of a debate and update all ELO scores.

        In a multi-model debate:
          - Winner faces each loser in a pairwise comparison.
          - If consensus_stability > DRAW_STABILITY_THRESHOLD,
            all models are treated as drawing.

        Args:
            winner_id:           Registry key of the winning model.
            loser_ids:           Registry keys of all other models.
            reasoning_scores:    Optional dict of model_id → score
                                 (used to update rolling averages).
            is_draw:             Force-draw flag.
            consensus_stability: From the Consensus Engine output.
        """
        all_participants = [winner_id] + loser_ids
        self._ensure_entries(all_participants)

        # Detect draw via consensus stability
        if consensus_stability >= DRAW_STABILITY_THRESHOLD:
            is_draw = True

        if is_draw:
            self._record_draws(all_participants)
        else:
            self._record_winner_vs_all(winner_id, loser_ids)

        # Update rolling reasoning metric averages
        if reasoning_scores:
            for model_id, score in reasoning_scores.items():
                if model_id in self._rankings:
                    self._update_rolling_score(model_id, score)

        self._total_debates += 1
        self._last_updated = datetime.now(timezone.utc).isoformat()
        self._save()
        logger.info(
            "ELO: battle recorded — winner=%s, losers=%s, draw=%s",
            winner_id, loser_ids, is_draw,
        )

    def record_from_evaluation(self, record: EvaluationRecord) -> None:
        """
        Convenience method: ingest an EvaluationRecord directly.

        Determines winner from record.winner or highest reasoning_score.
        """
        all_models = record.models_debated
        if not all_models:
            return

        winner = record.winner
        if not winner and record.reasoning_scores:
            winner = max(record.reasoning_scores, key=record.reasoning_scores.get)

        if not winner or winner not in all_models:
            logger.warning("ELO: no valid winner in evaluation record %s", record.record_id)
            return

        losers = [m for m in all_models if m != winner]
        self.record_battle(
            winner_id=winner,
            loser_ids=losers,
            reasoning_scores=record.reasoning_scores or None,
            consensus_stability=record.consensus_stability,
        )

    def get_leaderboard(self, top_n: Optional[int] = None) -> List[ELORankingEntry]:
        """
        Return models sorted by ELO score descending.

        Args:
            top_n: If set, return only the top N models.
        """
        ranked = sorted(
            self._rankings.values(),
            key=lambda e: e.elo_score,
            reverse=True,
        )
        return ranked[:top_n] if top_n else ranked

    def get_model_entry(self, model_id: str) -> Optional[ELORankingEntry]:
        """Return the ELO entry for a specific model."""
        return self._rankings.get(model_id)

    def get_leaderboard_dict(self) -> Dict[str, Any]:
        """Serialise leaderboard for API response."""
        entries = self.get_leaderboard()
        return {
            "last_updated": self._last_updated,
            "total_debates": self._total_debates,
            "rankings": [
                {
                    "rank": i + 1,
                    "model_id": e.model_id,
                    "model_name": e.model_name,
                    "elo_score": round(e.elo_score, 1),
                    "wins": e.wins,
                    "losses": e.losses,
                    "draws": e.draws,
                    "total_debates": e.total_debates,
                    "win_rate": round(e.win_rate, 4),
                    "avg_reasoning_score": round(e.avg_reasoning_score, 4),
                    "avg_evidence_density": round(e.avg_evidence_density, 4),
                    "avg_contradiction_rate": round(e.avg_contradiction_rate, 4),
                    "last_updated": e.last_updated,
                }
                for i, e in enumerate(entries)
            ],
        }

    def reset_rankings(self) -> None:
        """Reset all rankings to initial ELO (use in testing only)."""
        for entry in self._rankings.values():
            entry.elo_score = INITIAL_ELO
            entry.wins = 0
            entry.losses = 0
            entry.draws = 0
            entry.total_debates = 0
        self._total_debates = 0
        self._save()

    # ── Internal — ELO computation ────────────────────────────

    def _expected_score(self, ra: float, rb: float) -> float:
        """Standard ELO expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _k_factor(self, total_debates: int) -> float:
        """Glicko-style K-factor: decreases as a model matures."""
        if total_debates < 30:
            return K_NEW
        if total_debates < 100:
            return K_MID
        return K_VETERAN

    def _update_elo(
        self, model_a: str, model_b: str, result_a: float
    ) -> None:
        """
        Update ELO scores for a single pairwise comparison.

        result_a: 1.0 = A wins, 0.5 = draw, 0.0 = B wins
        """
        ea = self._rankings[model_a]
        eb = self._rankings[model_b]

        k_a = self._k_factor(ea.total_debates)
        k_b = self._k_factor(eb.total_debates)

        expected_a = self._expected_score(ea.elo_score, eb.elo_score)
        expected_b = 1.0 - expected_a

        ea.elo_score += k_a * (result_a - expected_a)
        eb.elo_score += k_b * ((1.0 - result_a) - expected_b)

        # Update counts
        ea.total_debates += 1
        eb.total_debates += 1

        if result_a == 1.0:
            ea.wins   += 1
            eb.losses += 1
        elif result_a == 0.0:
            ea.losses += 1
            eb.wins   += 1
        else:
            ea.draws += 1
            eb.draws += 1

        ts = datetime.now(timezone.utc).isoformat()
        ea.last_updated = ts
        eb.last_updated = ts

    def _record_winner_vs_all(
        self, winner_id: str, loser_ids: List[str]
    ) -> None:
        """Winner gains ELO against each loser individually."""
        for loser_id in loser_ids:
            self._update_elo(winner_id, loser_id, result_a=1.0)

    def _record_draws(self, model_ids: List[str]) -> None:
        """All pairs draw."""
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                self._update_elo(model_ids[i], model_ids[j], result_a=0.5)

    def _update_rolling_score(
        self, model_id: str, score: float
    ) -> None:
        """Exponential moving average for rolling quality metrics."""
        e = self._rankings[model_id]
        alpha = 0.15   # smoothing factor
        if e.avg_reasoning_score == 0.0:
            e.avg_reasoning_score = score
        else:
            e.avg_reasoning_score = (
                (1 - alpha) * e.avg_reasoning_score + alpha * score
            )

    # ── Internal — Storage ────────────────────────────────────

    def _ensure_entries(self, model_ids: List[str]) -> None:
        """Create ELO entries for models that don't have one yet."""
        from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
        for mid in model_ids:
            if mid not in self._rankings:
                spec = COGNITIVE_MODEL_REGISTRY.get(mid)
                name = spec.name if spec else mid
                self._rankings[mid] = ELORankingEntry(
                    model_id=mid,
                    model_name=name,
                    elo_score=INITIAL_ELO,
                )

    def _seed_known_models(self) -> None:
        """Ensure all known registry models have an ELO entry at startup."""
        try:
            from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
            for mid, spec in COGNITIVE_MODEL_REGISTRY.items():
                if mid not in self._rankings:
                    self._rankings[mid] = ELORankingEntry(
                        model_id=mid,
                        model_name=spec.name,
                        elo_score=INITIAL_ELO,
                    )
        except Exception as exc:
            logger.warning("ELO: could not seed known models: %s", exc)

    def _load(self) -> None:
        """Load rankings from disk. Silent on missing file."""
        if not os.path.exists(_ELO_PATH):
            logger.info("ELO: no rankings file found — starting fresh")
            return
        try:
            with open(_ELO_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._total_debates = data.get("total_debates", 0)
            self._last_updated  = data.get("last_updated", "")
            for mid, raw in data.get("rankings", {}).items():
                self._rankings[mid] = ELORankingEntry(**raw)
            logger.info("ELO: loaded %d model rankings", len(self._rankings))
        except Exception as exc:
            logger.error("ELO: failed to load rankings: %s", exc)

    def _save(self) -> None:
        """
        Atomically save rankings to disk.
        Uses write-to-temp + rename to prevent corruption on crash.
        """
        data = {
            "last_updated": self._last_updated,
            "total_debates": self._total_debates,
            "rankings": {
                mid: entry.model_dump()
                for mid, entry in self._rankings.items()
            },
        }
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=_DATA_DIR, prefix=".elo_tmp_", suffix=".json"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, _ELO_PATH)
        except Exception as exc:
            logger.error("ELO: failed to save rankings: %s", exc)


# ── Module-level singleton ────────────────────────────────────
_elo_engine_instance: Optional[ELORankingEngine] = None


def get_elo_engine() -> ELORankingEngine:
    """
    Return the module-level ELO engine singleton.
    Initialised lazily on first call.
    """
    global _elo_engine_instance
    if _elo_engine_instance is None:
        _elo_engine_instance = ELORankingEngine()
    return _elo_engine_instance
