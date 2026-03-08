"""
============================================================
Battle Visualization Engine — Sentinel-E Battle Platform v2
============================================================
Research-grade visualization dashboard for model debates.

Dashboard layout:
    ┌─────────────────────────────────────────────────────┐
    │  PROMPT DISPLAY                                      │
    │  "Explain nuclear fusion…"                           │
    ├──────────────┬──────────────┬───────────────────────┤
    │  MODEL A     │  MODEL B     │  MODEL C               │
    │  [preview]   │  [preview]   │  [preview]             │
    ├──────────────┴──────────────┴───────────────────────┤
    │  DEBATE TIMELINE                                     │
    │  Round 1 ────────────────────────────────────────   │
    │  Round 2 ────────────────────────────────────────   │
    │  Round 3 ────────────────────────────────────────   │
    ├─────────────────────────────────────────────────────┤
    │  REASONING METRICS                                   │
    │  Coherence  Evidence  Agreement  Confidence          │
    ├─────────────────────────────────────────────────────┤
    │  CONSENSUS STABILITY SCORE: 0.74                    │
    └─────────────────────────────────────────────────────┘

Additional visualizations:
    - Conflict Graph     (viz/conflict_graph.py)
    - Reasoning Heatmap  (viz/heatmap.py)
    - Model vs Metric radar chart
    - Debate progression timeline (confidence per round)

How visualization improves interpretability:
    1. Conflict Graph makes disagreement legible. When two models hold
       contradictory positions, the edge between them in the conflict
       graph immediately signals that human review may be needed.
       Without visualization, disagreements are buried in JSON fields.

    2. Agreement Heatmap reveals clusters. A 6-model debate with two
       tight clusters (3 models agree on one thesis, 3 on another)
       is immediately visible in the heatmap but invisible in text output.

    3. Reasoning Metrics bar charts allow side-by-side comparison of
       model quality on multiple dimensions simultaneously. A model with
       high coherence but low evidence density is a different failure mode
       from a model with low coherence but high evidence — the chart
       makes this distinction immediate.

    4. Debate progression timeline shows whether models converged or
       diverged across rounds. Convergence signals confidence building;
       divergence signals a genuinely contested question.

    5. Consensus Stability Score reduces a complex multi-model debate
       to a single interpretable signal. Platform users can filter
       battles by stability score to find the most contested debates.

This module assembles the full BattleVisualizationPayload and
optionally renders matplotlib/seaborn charts for reporting.
============================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.ensemble_schemas import (
    BattleVisualizationPayload,
    ConsensusScore,
    ModelReasoningMetrics,
    StructuredModelOutput,
)
from analysis.metrics_engine import MetricsEngine
from analysis.consensus_engine import ConsensusEngine, ConsensusEngineOutput
from analysis.reasoning_analytics import ReasoningAnalyticsEngine
from viz.conflict_graph import build_conflict_edges, plot_conflict_graph
from viz.heatmap import plot_agreement_heatmap

logger = logging.getLogger("BattleVisualization")


class BattleVisualizationEngine:
    """
    Assembles the full visualization payload for the battle dashboard.

    Usage:
        engine = BattleVisualizationEngine()
        payload = engine.build(
            prompt="Explain nuclear fusion",
            prompt_type="conceptual",
            round_outputs={
                "1": [{"model": "llama-3.3", "output": "...", "tokens_used": 198}],
                "2": [...],
                "3": [...],
            },
            final_round_outputs=[StructuredModelOutput(...)],
        )
        # payload.consensus_stability_score, payload.conflict_edges, etc.
    """

    def __init__(self):
        self._metrics  = MetricsEngine()
        self._consensus = ConsensusEngine()
        self._analytics = ReasoningAnalyticsEngine()

    # ── Public interface ──────────────────────────────────────

    def build(
        self,
        prompt: str,
        prompt_type: str,
        round_outputs: Dict[str, List[Dict[str, Any]]],
        final_round_outputs: List[StructuredModelOutput],
        include_charts: bool = False,
    ) -> BattleVisualizationPayload:
        """
        Build the complete BattleVisualizationPayload.

        Args:
            prompt:               The user prompt.
            prompt_type:          Detected prompt type (code/conceptual/general).
            round_outputs:        {round_key: [{model, output, tokens_used}]}
            final_round_outputs:  StructuredModelOutput list from the last round.
            include_charts:       If True, render PNG charts and embed in payload.

        Returns:
            BattleVisualizationPayload with all dashboard data.
        """
        # Step 1 — Reasoning metrics per model
        metrics: List[ModelReasoningMetrics] = self._metrics.evaluate_all(
            final_round_outputs
        )

        # Step 2 — Consensus engine
        consensus_result: ConsensusEngineOutput = self._consensus.evaluate(
            final_round_outputs
        )

        # Step 3 — Agreement heatmap data
        sim_matrix, model_labels = self._build_similarity_matrix(
            final_round_outputs
        )

        # Step 4 — Conflict edges
        similarities: Dict[Tuple[str, str], float] = {}
        n = len(final_round_outputs)
        for i in range(n):
            for j in range(i + 1, n):
                key = (
                    final_round_outputs[i].model_id,
                    final_round_outputs[j].model_id,
                )
                val = sim_matrix[i][j] if sim_matrix and len(sim_matrix) > i else 0.5
                similarities[key] = val

        conflict_edges = build_conflict_edges(
            models=model_labels,
            similarities=similarities,
            threshold=0.6,
        )

        # Step 5 — Debate progression timeline
        timeline = self._build_timeline(round_outputs)

        # Step 6 — Determine winner
        winner_score  = None
        winner_model  = None
        if consensus_result.scores:
            top = consensus_result.scores[0]
            winner_model = top.model
            winner_score = top.composite_score

        # Step 7 — Selected models list
        models_selected = [o.model_id for o in final_round_outputs]

        payload = BattleVisualizationPayload(
            prompt=prompt,
            prompt_type=prompt_type,
            models_selected=models_selected,
            round_outputs=round_outputs,
            reasoning_metrics=metrics,
            consensus_scores=consensus_result.scores,
            consensus_stability_score=consensus_result.stability_score,
            agreement_heatmap=sim_matrix,
            model_labels=model_labels,
            conflict_edges=conflict_edges,
            debate_timeline=timeline,
            winner=winner_model,
            winner_score=winner_score or 0.0,
        )

        if include_charts:
            payload = self._attach_charts(payload, similarities, sim_matrix, model_labels)

        # Step 8 — Reasoning analytics (heatmap, disagreements, steps)
        model_output_dicts = [
            {
                "model_id": o.model_id,
                "model_name": getattr(o, "model_name", o.model_id),
                "position": o.position,
                "reasoning": o.reasoning,
            }
            for o in final_round_outputs
        ]
        metric_dicts = [
            {
                "model": m.model,
                "model_name": m.model_name,
                "reasoning_score": m.reasoning_score,
                "evidence_density": m.evidence_density,
                "argument_depth": m.argument_depth,
                "logical_consistency": m.logical_consistency,
                "contradiction_rate": m.contradiction_rate,
                "confidence_alignment": m.confidence_calibration,
                "token_efficiency": m.token_efficiency,
            }
            for m in metrics
        ]
        payload.reasoning_analytics = self._analytics.analyze_debate(
            model_output_dicts, metric_dicts
        )

        logger.info(
            "BattleViz: built payload — %d models, stability=%.3f, winner=%s",
            len(models_selected),
            consensus_result.stability_score,
            winner_model,
        )
        return payload

    def to_frontend_dict(
        self, payload: BattleVisualizationPayload
    ) -> Dict[str, Any]:
        """
        Serialise BattleVisualizationPayload to the frontend contract.

        All Pydantic objects are converted to plain dicts/lists.
        Charts are included as base64 strings if present.
        """
        return {
            "prompt": payload.prompt,
            "prompt_type": payload.prompt_type,
            "models_selected": payload.models_selected,
            # Round-by-round outputs for the Debate Timeline panel
            "round_outputs": payload.round_outputs,
            # Per-model reasoning metrics for the Metrics panel
            "reasoning_metrics": [
                {
                    "model": m.model,
                    "model_name": m.model_name,
                    "reasoning_score": m.reasoning_score,
                    "evidence_density": m.evidence_density,
                    "argument_depth": m.argument_depth,
                    "logical_consistency": m.logical_consistency,
                    "contradiction_rate": m.contradiction_rate,
                    "confidence_alignment": m.confidence_calibration,
                    "token_efficiency": m.token_efficiency,
                }
                for m in payload.reasoning_metrics
            ],
            # Consensus scores for the Model Ranking panel
            "consensus_scores": [
                {
                    "model": s.model,
                    "model_name": s.model_name,
                    "reasoning_coherence": s.reasoning_coherence,
                    "evidence_support": s.evidence_support,
                    "consensus_alignment": s.consensus_alignment,
                    "confidence_calibration": s.confidence_calibration,
                    "contradiction_rate": s.contradiction_rate,
                    "composite_score": s.composite_score,
                    "rank": s.rank,
                }
                for s in payload.consensus_scores
            ],
            # Consensus Stability Score (headline metric)
            "consensus_stability_score": payload.consensus_stability_score,
            # Agreement heatmap (2D array for frontend heatmap renderer)
            "agreement_heatmap": payload.agreement_heatmap,
            "model_labels": payload.model_labels,
            # Conflict edges for the Conflict Graph panel
            "conflict_edges": payload.conflict_edges,
            # Debate progression timeline
            "debate_timeline": payload.debate_timeline,
            # Winner
            "winner": payload.winner,
            "winner_score": payload.winner_score,
            # Reasoning analytics (heatmap, disagreements, transparency steps)
            "reasoning_analytics": getattr(payload, "reasoning_analytics", None),
            # Confidence evolution: per-model confidence across rounds (for line chart)
            "confidence_evolution": self._build_confidence_evolution(payload.debate_timeline),
        }

    # ── Internal — Data builders ──────────────────────────────

    def _build_similarity_matrix(
        self,
        outputs: List[StructuredModelOutput],
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Build pairwise similarity matrix for heatmap rendering.

        Returns: (sim_matrix 2D list, model_labels list)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        labels = [o.model_id for o in outputs]
        texts  = [f"{o.position} {o.reasoning}" for o in outputs]
        n = len(texts)

        if n == 0:
            return [], []

        try:
            tfidf   = TfidfVectorizer(stop_words="english", max_features=500)
            vectors = tfidf.fit_transform(texts)
            matrix  = cosine_similarity(vectors).tolist()
        except Exception as exc:
            logger.warning("BattleViz: similarity fallback: %s", exc)
            matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        return matrix, labels

    def _build_timeline(
        self,
        round_outputs: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Build debate progression timeline from round outputs.

        Returns list of {round, model, output_preview, tokens_used, confidence}.
        Used for the Debate Timeline panel (Round 1 → Round 2 → Round 3).
        """
        timeline: List[Dict[str, Any]] = []
        for round_key in sorted(round_outputs.keys(), key=lambda k: int(k)):
            round_items = round_outputs[round_key]
            for item in round_items:
                output_text = item.get("output", "")
                timeline.append({
                    "round": int(round_key),
                    "model": item.get("model", ""),
                    "output_preview": output_text[:200] if output_text else "",
                    "tokens_used": item.get("tokens_used", 0),
                    "confidence": item.get("confidence", 0.5),
                    "position_shift": item.get("position_shift", 0.0),
                })
        return timeline

    def _build_confidence_evolution(
        self,
        timeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Transform timeline into per-round confidence evolution data.

        Returns: [{round: 1, model_a: 0.7, model_b: 0.6, ...}, ...]
        Suitable for a recharts LineChart.
        """
        if not timeline:
            return []

        rounds_map: Dict[int, Dict[str, float]] = {}
        for entry in timeline:
            r = entry.get("round", 1)
            model = entry.get("model", "unknown")
            conf = entry.get("confidence", 0.5)
            if r not in rounds_map:
                rounds_map[r] = {}
            rounds_map[r][model] = conf

        return [
            {"round": r, **models}
            for r, models in sorted(rounds_map.items())
        ]

    def _attach_charts(
        self,
        payload: BattleVisualizationPayload,
        similarities: Dict,
        sim_matrix: List[List[float]],
        model_labels: List[str],
    ) -> BattleVisualizationPayload:
        """
        Render and attach base64 PNG charts to the payload.
        Only called when include_charts=True (report mode).
        """
        try:
            conflict_png = plot_conflict_graph(
                models=model_labels,
                similarities=similarities,
                threshold=0.6,
                display=False,
            )
            if conflict_png:
                payload.round_outputs["_conflict_graph_png"] = [{"png": conflict_png}]
        except Exception as exc:
            logger.warning("BattleViz: conflict graph render failed: %s", exc)

        try:
            heatmap_png = plot_agreement_heatmap(
                similarity_matrix=sim_matrix,
                labels=model_labels,
                display=False,
                title="Reasoning Agreement Heatmap",
            )
            if heatmap_png:
                payload.round_outputs["_heatmap_png"] = [{"png": heatmap_png}]
        except Exception as exc:
            logger.warning("BattleViz: heatmap render failed: %s", exc)

        return payload


# ── Module-level singleton ────────────────────────────────────
_viz_engine_instance: Optional[BattleVisualizationEngine] = None


def get_battle_viz_engine() -> BattleVisualizationEngine:
    global _viz_engine_instance
    if _viz_engine_instance is None:
        _viz_engine_instance = BattleVisualizationEngine()
    return _viz_engine_instance
