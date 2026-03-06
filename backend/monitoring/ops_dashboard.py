"""
============================================================
Operations Dashboard — Sentinel-E Battle Platform v2
============================================================
Real-time operational telemetry for the evaluation platform.

Metrics tracked per debate session:
    • model_latency          — p50/p95 per model (ms)
    • token_usage            — total tokens per model per session
    • error_rate             — fraction of failed model calls
    • response_length        — avg tokens per response
    • consensus_confidence   — rolling avg stability from ConsensusEngine
    • debate_stability       — rolling avg debate_stability_score

Operations dashboard display:
    ┌─────────────────────────────────────────────┐
    │  Average Debate Time     342 ms              │
    │  Consensus Confidence    0.74                │
    │  Conflict Rate           0.22                │
    │  System Error Rate       0.03                │
    ├─────────────────────────────────────────────┤
    │  Model Latency (p50)                         │
    │    llama-3.3           210 ms                │
    │    mixtral-8x7b        280 ms                │
    │    deepseek-chat       190 ms                │
    ├─────────────────────────────────────────────┤
    │  Token Usage (last hour)                     │
    │    llama-3.3           82,400 tokens         │
    │    mixtral-8x7b        61,200 tokens         │
    └─────────────────────────────────────────────┘

How operational analytics maintain system reliability:
    1. Latency monitoring: If a model's p95 latency spikes above its
       historical baseline, the system can proactively deprioritise it
       in model selection or trigger a fallback before timeouts occur.

    2. Token usage tracking: Token budgets are finite (rate limits,
       cost limits). Monitoring per-model usage allows the cost governor
       to rebalance token allocation in real time before hitting limits.

    3. Error rate surveillance: A model that consistently fails (> 10%
       error rate) is deprioritised by the tiered selector even if its
       theoretical quality is high. Reliability is a prerequisite for
       quality in production systems.

    4. Consensus confidence trending: A sudden drop in consensus
       confidence across unrelated prompts suggests that model responses
       have become less coherent — an early signal of provider-side issues.

    5. Debate stability: High conflict rate (many low-similiarity model
       pairs) is a signal of high epistemic uncertainty on the incoming
       prompts, which may require human escalation.

Architecture: in-process ring buffer (no external time-series DB required).
The buffer holds the last 1,000 events. For production deployments, this
module can be swapped for a Prometheus/Grafana integration.
============================================================
"""

from __future__ import annotations

import collections
import logging
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from core.ensemble_schemas import OperationsSnapshot

logger = logging.getLogger("OpsDashboard")

# Ring buffer size (number of debate events to retain in memory)
_BUFFER_SIZE = 1000


class DebateEvent:
    """
    Single recorded debate event.
    Appended to the ring buffer after every debate completes.
    """
    __slots__ = (
        "timestamp", "debate_id", "prompt_type",
        "model_latencies", "model_tokens", "model_errors",
        "response_lengths", "consensus_confidence", "debate_stability",
        "total_duration_ms",
    )

    def __init__(
        self,
        debate_id: str,
        prompt_type: str,
        model_latencies: Dict[str, float],      # model_id → ms
        model_tokens: Dict[str, int],            # model_id → tokens
        model_errors: Dict[str, bool],           # model_id → error flag
        response_lengths: Dict[str, int],        # model_id → response len
        consensus_confidence: float,
        debate_stability: float,
        total_duration_ms: float,
    ):
        self.timestamp           = time.monotonic()
        self.debate_id           = debate_id
        self.prompt_type         = prompt_type
        self.model_latencies     = model_latencies
        self.model_tokens        = model_tokens
        self.model_errors        = model_errors
        self.response_lengths    = response_lengths
        self.consensus_confidence = consensus_confidence
        self.debate_stability    = debate_stability
        self.total_duration_ms   = total_duration_ms


class OpsDashboard:
    """
    In-process operational telemetry for the Battle Platform.

    Thread-safe for read access (append-only deque).
    Write access assumes single-writer (asyncio event loop).

    Usage:
        ops = OpsDashboard()
        ops.record_debate(debate_id=..., ...)
        snapshot = ops.get_snapshot()
        summary  = ops.get_model_latency_summary()
    """

    def __init__(self, buffer_size: int = _BUFFER_SIZE):
        self._events: Deque[DebateEvent] = collections.deque(maxlen=buffer_size)
        self._total_debates: int = 0
        self._total_errors: int  = 0
        self._started_at: float  = time.monotonic()

    # ── Write API ─────────────────────────────────────────────

    def record_debate(
        self,
        debate_id: str,
        prompt_type: str,
        model_latencies: Dict[str, float],
        model_tokens: Dict[str, int],
        model_errors: Dict[str, bool],
        response_lengths: Dict[str, int],
        consensus_confidence: float,
        debate_stability: float,
        total_duration_ms: float,
    ) -> None:
        """Record a completed debate event."""
        event = DebateEvent(
            debate_id=debate_id,
            prompt_type=prompt_type,
            model_latencies=model_latencies,
            model_tokens=model_tokens,
            model_errors=model_errors,
            response_lengths=response_lengths,
            consensus_confidence=consensus_confidence,
            debate_stability=debate_stability,
            total_duration_ms=total_duration_ms,
        )
        self._events.append(event)
        self._total_debates += 1
        if any(model_errors.values()):
            self._total_errors += 1
        logger.debug(
            "OpsDashboard: debate %s recorded (%.0f ms, consensus=%.2f)",
            debate_id[:8], total_duration_ms, consensus_confidence,
        )

    # ── Read API ──────────────────────────────────────────────

    def get_snapshot(self, window_seconds: float = 3600.0) -> OperationsSnapshot:
        """
        Return a real-time OperationsSnapshot for the given time window.

        Args:
            window_seconds: Look back this many seconds (default: 1 hour).
        """
        recent = self._recent_events(window_seconds)
        n = len(recent)

        if n == 0:
            return OperationsSnapshot(
                total_debates_last_hour=0,
                avg_debate_time_ms=0.0,
                consensus_confidence_avg=0.0,
                conflict_rate=0.0,
                system_error_rate=0.0,
            )

        # Average debate time
        avg_debate_ms = statistics.mean(e.total_duration_ms for e in recent)

        # Consensus confidence average
        conf_values = [e.consensus_confidence for e in recent]
        avg_conf = statistics.mean(conf_values)

        # Conflict rate: fraction of debates with stability < 0.45
        conflicts = sum(1 for e in recent if e.debate_stability < 0.45)
        conflict_rate = conflicts / n

        # Error rate: fraction of debates with at least one model error
        errors = sum(1 for e in recent if any(e.model_errors.values()))
        error_rate = errors / n

        # Per-model aggregation
        model_latencies: Dict[str, float] = {}
        model_tokens: Dict[str, int]      = {}
        model_error_rates: Dict[str, float] = {}
        response_lengths: Dict[str, int]  = {}

        for mid in self._all_model_ids(recent):
            lats   = [e.model_latencies.get(mid, 0) for e in recent if mid in e.model_latencies]
            toks   = [e.model_tokens.get(mid, 0)    for e in recent if mid in e.model_tokens]
            errs   = [e.model_errors.get(mid, False) for e in recent if mid in e.model_errors]
            lens   = [e.response_lengths.get(mid, 0) for e in recent if mid in e.response_lengths]

            if lats:
                model_latencies[mid] = round(statistics.median(lats), 1)
            if toks:
                model_tokens[mid] = sum(toks)
            if errs:
                model_error_rates[mid] = round(sum(errs) / len(errs), 4)
            if lens:
                response_lengths[mid] = round(statistics.mean(lens))

        elo = None
        try:
            from ranking.elo_engine import get_elo_engine
            elo = get_elo_engine()._last_updated
        except Exception:
            pass

        return OperationsSnapshot(
            total_debates_last_hour=n,
            avg_debate_time_ms=round(avg_debate_ms, 1),
            consensus_confidence_avg=round(avg_conf, 4),
            conflict_rate=round(conflict_rate, 4),
            system_error_rate=round(error_rate, 4),
            model_latencies=model_latencies,
            model_token_usage=model_tokens,
            model_error_rates=model_error_rates,
            avg_response_lengths=response_lengths,
            leaderboard_last_updated=elo,
        )

    def get_model_latency_summary(
        self, window_seconds: float = 3600.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Per-model latency percentiles: p50, p95, p99.

        Returns:
            {model_id: {"p50": ..., "p95": ..., "p99": ..., "mean": ...}}
        """
        recent = self._recent_events(window_seconds)
        summary: Dict[str, Dict[str, float]] = {}
        for mid in self._all_model_ids(recent):
            lats = sorted(
                e.model_latencies[mid] for e in recent
                if mid in e.model_latencies
            )
            if not lats:
                continue
            n = len(lats)
            summary[mid] = {
                "p50":  round(lats[int(n * 0.50)], 1),
                "p95":  round(lats[min(n - 1, int(n * 0.95))], 1),
                "p99":  round(lats[min(n - 1, int(n * 0.99))], 1),
                "mean": round(statistics.mean(lats), 1),
                "count": n,
            }
        return summary

    def get_confidence_trend(
        self, window_seconds: float = 86400.0, bucket_count: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Consensus confidence trend over time (for sparkline charts).

        Divides the window into `bucket_count` equal buckets and
        returns the mean confidence per bucket.
        """
        recent = self._recent_events(window_seconds)
        if not recent:
            return []

        now = time.monotonic()
        start = now - window_seconds
        bucket_width = window_seconds / bucket_count

        buckets: Dict[int, List[float]] = {}
        for event in recent:
            bucket = int((event.timestamp - start) / bucket_width)
            bucket = max(0, min(bucket_count - 1, bucket))
            buckets.setdefault(bucket, []).append(event.consensus_confidence)

        trend = []
        for i in range(bucket_count):
            values = buckets.get(i, [])
            trend.append({
                "bucket": i,
                "avg_confidence": round(statistics.mean(values), 4) if values else None,
                "event_count": len(values),
            })
        return trend

    def system_health(self) -> Dict[str, Any]:
        """
        Overall system health summary for the admin dashboard.

        Returns a traffic-light status: healthy / degraded / critical.
        """
        snapshot = self.get_snapshot(window_seconds=3600)

        error_status = (
            "healthy"   if snapshot.system_error_rate < 0.05 else
            "degraded"  if snapshot.system_error_rate < 0.15 else
            "critical"
        )
        latency_status = (
            "healthy"  if snapshot.avg_debate_time_ms < 800  else
            "degraded" if snapshot.avg_debate_time_ms < 2000 else
            "critical"
        )
        confidence_status = (
            "healthy"  if snapshot.consensus_confidence_avg > 0.60 else
            "degraded" if snapshot.consensus_confidence_avg > 0.40 else
            "critical"
        )

        statuses = [error_status, latency_status, confidence_status]
        overall = (
            "critical"  if "critical" in statuses  else
            "degraded"  if "degraded" in statuses  else
            "healthy"
        )

        return {
            "overall": overall,
            "components": {
                "error_rate":   {"status": error_status,      "value": snapshot.system_error_rate},
                "latency":      {"status": latency_status,     "value": snapshot.avg_debate_time_ms},
                "confidence":   {"status": confidence_status, "value": snapshot.consensus_confidence_avg},
            },
            "total_debates": self._total_debates,
            "uptime_seconds": round(time.monotonic() - self._started_at),
            "buffer_utilisation": f"{len(self._events)}/{self._events.maxlen}",
        }

    # ── Internal ──────────────────────────────────────────────

    def _recent_events(self, window_seconds: float) -> List[DebateEvent]:
        """Return events within the last `window_seconds` seconds."""
        cutoff = time.monotonic() - window_seconds
        return [e for e in self._events if e.timestamp >= cutoff]

    @staticmethod
    def _all_model_ids(events: List[DebateEvent]) -> List[str]:
        """Collect all unique model IDs across events."""
        seen: set = set()
        for e in events:
            seen.update(e.model_latencies.keys())
        return sorted(seen)


# ── Module-level singleton ────────────────────────────────────
_ops_dashboard_instance: Optional[OpsDashboard] = None


def get_ops_dashboard() -> OpsDashboard:
    """Return the module-level OpsDashboard singleton."""
    global _ops_dashboard_instance
    if _ops_dashboard_instance is None:
        _ops_dashboard_instance = OpsDashboard()
    return _ops_dashboard_instance
