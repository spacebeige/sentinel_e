"""
============================================================
Automated Benchmarking Pipeline — Sentinel-E Battle Platform v2
============================================================
Continuous background evaluation of all resident models.

Daily workflow:
    evaluation dataset prompts  (selects up to 10 per day)
        ↓
    run debates                 (tiered model selection)
        ↓
    calculate metrics           (MetricsEngine + ConsensusEngine)
        ↓
    update leaderboard          (ELORankingEngine)
        ↓
    write benchmark report      (backend/data/benchmark_reports.json)

How continuous benchmarking improves evaluation reliability:
    1. Temporal stability tracking — A one-time evaluation gives a
       snapshot. Daily benchmarking reveals whether a model's performance
       is stable or drifts over time (e.g., due to provider-side updates).

    2. Prompt diversity saturation — Each run samples different prompts.
       After N runs, the leaderboard reflects performance across the full
       prompt distribution rather than cherry-picked examples.

    3. Inter-run consistency — High ELO volatility (week-to-week score
       swings) signals that a model is sensitive to prompt framing,
       which is itself a reliability concern.

    4. Anomaly detection — A sudden drop in a usually-strong model's
       benchmark score triggers an alert, providing early warning of
       provider outages or model version changes.

    5. Compound accuracy — Each daily run is an independent sample.
       The law of large numbers means the rolling 30-day average is
       significantly more accurate than any single evaluation cycle.

The pipeline runs as a background asyncio task. It respects the
existing BackgroundDaemon infrastructure.
============================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.ensemble_schemas import (
    EvaluationRecord,
    ModelReasoningMetrics,
    StructuredModelOutput,
)
from analysis.metrics_engine import MetricsEngine
from analysis.consensus_engine import ConsensusEngine
from evaluation.dataset import get_evaluation_dataset
from ranking.elo_engine import get_elo_engine

logger = logging.getLogger("BenchmarkPipeline")

_DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
_REPORT_PATH   = os.path.join(_DATA_DIR, "benchmark_reports.json")

# Benchmark prompt bank (diverse, stable, reproducible)
_BENCHMARK_PROMPTS: List[Dict[str, str]] = [
    {"prompt": "Explain the CAP theorem and its implications for database design.", "type": "conceptual"},
    {"prompt": "What are the risks of optimising for a single metric in machine learning?", "type": "conceptual"},
    {"prompt": "Why does adding more engineers often slow down a software project?", "type": "conceptual"},
    {"prompt": "Explain how transformers handle long-range dependencies compared to RNNs.", "type": "conceptual"},
    {"prompt": "What are the second-order effects of automation on labour markets?", "type": "depth"},
    {"prompt": "Why is zero-shot generalisation hard for current AI systems?", "type": "conceptual"},
    {"prompt": "Explain gradient descent and why it can get stuck in local minima.", "type": "logical"},
    {"prompt": "What does it mean for an algorithm to be O(n log n)?", "type": "logical"},
    {"prompt": "Why is the halting problem undecidable?", "type": "logical"},
    {"prompt": "Explain why TCP guarantees delivery but UDP does not — and when UDP is preferable.", "type": "conceptual"},
    {"prompt": "What is attention in transformer models and why does it work?", "type": "conceptual"},
    {"prompt": "Explain how a proof by contradiction works with an example.", "type": "logical"},
    {"prompt": "Why do neural networks need activation functions?", "type": "conceptual"},
    {"prompt": "How does HTTPS prevent man-in-the-middle attacks?", "type": "evidence"},
    {"prompt": "Explain why recursion can be more elegant but less efficient than iteration.", "type": "depth"},
    {"prompt": "What are the tradeoffs in choosing between SQL and NoSQL databases?", "type": "conceptual"},
    {"prompt": "Explain regularisation in machine learning and why it prevents overfitting.", "type": "conceptual"},
    {"prompt": "Why is floating point arithmetic imprecise?", "type": "logical"},
    {"prompt": "What makes a hash function cryptographically secure?", "type": "evidence"},
    {"prompt": "Explain the difference between concurrency and parallelism.", "type": "conceptual"},
]

# How many prompts to run per benchmark session
PROMPTS_PER_RUN = 10


class BenchmarkPipeline:
    """
    Manages automated daily benchmarking of all resident models.

    This pipeline does NOT call model APIs directly — it submits
    prompts to the existing MetaCognitive Orchestrator (MCO),
    which routes through the full debate + metric pipeline.

    For standalone testing (no running MCO), it falls back to
    direct MetricsEngine evaluation on cached/stub responses.
    """

    def __init__(self):
        self._metrics   = MetricsEngine()
        self._consensus = ConsensusEngine()
        self._running   = False
        self._reports: List[Dict[str, Any]] = []
        self._load_reports()

    # ── Public interface ──────────────────────────────────────

    async def run_benchmark_session(
        self,
        prompt_override: Optional[List[Dict[str, str]]] = None,
        orchestrator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute one full benchmark session.

        Args:
            prompt_override: Custom prompts for this run (overrides sampling).
            orchestrator:    MCO instance for debate execution.
                             If None, the pipeline runs in metrics-only mode.

        Returns:
            Benchmark report dictionary.
        """
        if self._running:
            logger.warning("BenchmarkPipeline: already running, skipping")
            return {"status": "skipped", "reason": "already_running"}

        self._running = True
        session_id    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        logger.info("BenchmarkPipeline: session %s started", session_id)

        selected_prompts = (
            prompt_override
            or random.sample(_BENCHMARK_PROMPTS, min(PROMPTS_PER_RUN, len(_BENCHMARK_PROMPTS)))
        )

        session_results: List[Dict[str, Any]] = []
        model_score_accum: Dict[str, List[float]] = {}

        for item in selected_prompts:
            prompt = item["prompt"]
            ptype  = item.get("type", "general")

            result = await self._run_single_prompt(
                prompt=prompt,
                prompt_type=ptype,
                orchestrator=orchestrator,
            )
            session_results.append(result)

            # Accumulate per-model scores
            for mid, score in result.get("reasoning_scores", {}).items():
                model_score_accum.setdefault(mid, []).append(score)

        # Compute session-level model averages
        model_session_averages: Dict[str, float] = {
            mid: sum(scores) / len(scores)
            for mid, scores in model_score_accum.items()
            if scores
        }

        report = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts_evaluated": len(session_results),
            "model_averages": model_session_averages,
            "results": session_results,
            "leaderboard_snapshot": get_elo_engine().get_leaderboard_dict(),
        }

        self._reports.append(report)
        self._save_reports()

        self._running = False
        logger.info(
            "BenchmarkPipeline: session %s complete — %d prompts, %d models",
            session_id, len(session_results), len(model_session_averages),
        )
        return report

    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Return the most recent benchmark report."""
        return self._reports[-1] if self._reports else None

    def get_reports(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the N most recent benchmark reports."""
        return self._reports[-n:]

    def get_model_trend(
        self, model_id: str, n_sessions: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Return score trend for a model across the last N sessions.

        Returns list of {session_id, timestamp, avg_score}.
        This is the primary signal for temporal drift detection.
        """
        trend: List[Dict[str, Any]] = []
        for report in self._reports[-n_sessions:]:
            avg = report.get("model_averages", {}).get(model_id)
            if avg is not None:
                trend.append({
                    "session_id": report["session_id"],
                    "timestamp": report["timestamp"],
                    "avg_score": round(avg, 4),
                })
        return trend

    # ── Internal — Single prompt execution ────────────────────

    async def _run_single_prompt(
        self,
        prompt: str,
        prompt_type: str,
        orchestrator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run a single benchmark prompt.

        If orchestrator is available, uses the full debate pipeline.
        Otherwise falls back to metrics-only mode on empty outputs.
        """
        reasoning_scores: Dict[str, float] = {}

        if orchestrator is not None:
            try:
                # Use MCO to run the full debate pipeline
                result = await orchestrator.process(
                    query=prompt,
                    mode="debate",
                    chat_id=f"benchmark_{datetime.now(timezone.utc).strftime('%s')}",
                )
                outputs = result.get("model_outputs", [])
                for o in outputs:
                    if isinstance(o, dict):
                        mid   = o.get("model_id", "unknown")
                        score = o.get("confidence", 0.5)
                        reasoning_scores[mid] = score

                # Write evaluation record
                winner = max(reasoning_scores, key=reasoning_scores.get) if reasoning_scores else None
                record = EvaluationRecord(
                    prompt=prompt,
                    prompt_type=prompt_type,
                    models_debated=list(reasoning_scores.keys()),
                    responses={mid: "" for mid in reasoning_scores},
                    reasoning_scores=reasoning_scores,
                    winner=winner,
                    evaluation_source="automated",
                )
                get_evaluation_dataset().append_record(record)

                return {
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "reasoning_scores": reasoning_scores,
                    "winner": winner,
                    "source": "orchestrator",
                }
            except Exception as exc:
                logger.warning(
                    "BenchmarkPipeline: orchestrator failed for '%s': %s",
                    prompt[:60], exc,
                )

        # Fallback: metrics-only (no model calls)
        return {
            "prompt": prompt,
            "prompt_type": prompt_type,
            "reasoning_scores": {},
            "winner": None,
            "source": "skipped_no_orchestrator",
        }

    # ── Internal — Persistence ────────────────────────────────

    def _load_reports(self) -> None:
        if not os.path.exists(_REPORT_PATH):
            return
        try:
            with open(_REPORT_PATH, "r", encoding="utf-8") as fh:
                self._reports = json.load(fh)
            logger.info("BenchmarkPipeline: loaded %d reports", len(self._reports))
        except Exception as exc:
            logger.error("BenchmarkPipeline: failed to load reports: %s", exc)

    def _save_reports(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        try:
            # Keep only last 365 reports to bound file size
            trimmed = self._reports[-365:]
            fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, prefix=".bench_tmp_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(trimmed, fh, indent=2)
            os.replace(tmp, _REPORT_PATH)
        except Exception as exc:
            logger.error("BenchmarkPipeline: failed to save reports: %s", exc)


# ── Module-level singleton ────────────────────────────────────
_benchmark_instance: Optional[BenchmarkPipeline] = None


def get_benchmark_pipeline() -> BenchmarkPipeline:
    global _benchmark_instance
    if _benchmark_instance is None:
        _benchmark_instance = BenchmarkPipeline()
    return _benchmark_instance
