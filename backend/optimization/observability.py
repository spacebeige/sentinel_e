"""
============================================================
Sentinel-E — Observability Engine
============================================================
Structured logging and metrics for production monitoring.

Captures:
  - Model used per request
  - Tokens sent / received
  - Cache hit / miss (which tier)
  - Fallback / circuit-breaker events
  - Latency breakdown
  - Cost estimate
  - Session budget status

Security:
  - NO user-submitted content in logs
  - NO API keys or secrets
  - Session IDs only (no PII)

Memory safety: stdlib only. Bounded ring buffer for metrics.
============================================================
"""

import time
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("Observability")


# ============================================================
# EVENT TYPES
# ============================================================

class EventType(str, Enum):
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    MODEL_CALL = "model_call"
    FALLBACK_TRIGGERED = "fallback_triggered"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_RECOVER = "circuit_recover"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    DOWNGRADE = "model_downgrade"
    TOKEN_OPTIMIZATION = "token_optimization"
    ERROR = "error"


# ============================================================
# STRUCTURED EVENT
# ============================================================

@dataclass
class ObservabilityEvent:
    """Single observable event."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    request_id: str = ""
    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    cache_tier: str = ""          # "exact", "lexical", "semantic", ""
    fallback_from: str = ""
    fallback_to: str = ""
    error_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "event": self.event_type.value,
            "ts": round(self.timestamp, 3),
        }
        # Only include non-empty fields
        if self.session_id:
            d["session_id"] = self.session_id
        if self.request_id:
            d["request_id"] = self.request_id
        if self.model_id:
            d["model"] = self.model_id
        if self.input_tokens:
            d["in_tokens"] = self.input_tokens
        if self.output_tokens:
            d["out_tokens"] = self.output_tokens
        if self.latency_ms:
            d["latency_ms"] = round(self.latency_ms, 1)
        if self.cost_estimate:
            d["cost"] = round(self.cost_estimate, 8)
        if self.cache_tier:
            d["cache_tier"] = self.cache_tier
        if self.fallback_from:
            d["fallback_from"] = self.fallback_from
        if self.fallback_to:
            d["fallback_to"] = self.fallback_to
        if self.error_type:
            d["error"] = self.error_type
        if self.metadata:
            d["meta"] = self.metadata
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


# ============================================================
# REQUEST TRACER
# ============================================================

class RequestTracer:
    """
    Traces a single request through the pipeline.
    Collects timing and events, emits structured log on completion.
    """

    def __init__(self, session_id: str, request_id: str):
        self.session_id = session_id
        self.request_id = request_id
        self.start_time = time.time()
        self.events: List[ObservabilityEvent] = []
        self._spans: Dict[str, float] = {}

    def start_span(self, name: str):
        """Start a named timing span."""
        self._spans[name] = time.time()

    def end_span(self, name: str) -> float:
        """End a named timing span, return duration in ms."""
        start = self._spans.pop(name, None)
        if start is None:
            return 0.0
        return (time.time() - start) * 1000

    def record_event(self, event: ObservabilityEvent):
        """Add an event to this trace."""
        event.session_id = self.session_id
        event.request_id = self.request_id
        self.events.append(event)

    def record_cache_hit(self, tier: str, latency_ms: float = 0.0):
        self.record_event(ObservabilityEvent(
            event_type=EventType.CACHE_HIT,
            cache_tier=tier,
            latency_ms=latency_ms,
        ))

    def record_cache_miss(self):
        self.record_event(ObservabilityEvent(
            event_type=EventType.CACHE_MISS,
        ))

    def record_model_call(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_estimate: float,
    ):
        self.record_event(ObservabilityEvent(
            event_type=EventType.MODEL_CALL,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_estimate=cost_estimate,
        ))

    def record_fallback(self, from_model: str, to_model: str, reason: str = ""):
        self.record_event(ObservabilityEvent(
            event_type=EventType.FALLBACK_TRIGGERED,
            fallback_from=from_model,
            fallback_to=to_model,
            metadata={"reason": reason} if reason else {},
        ))

    def record_token_optimization(self, original_tokens: int, optimized_tokens: int):
        saved = original_tokens - optimized_tokens
        self.record_event(ObservabilityEvent(
            event_type=EventType.TOKEN_OPTIMIZATION,
            metadata={
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "tokens_saved": saved,
                "reduction_pct": round(saved / max(original_tokens, 1) * 100, 1),
            },
        ))

    def record_error(self, error_type: str, metadata: Optional[Dict] = None):
        self.record_event(ObservabilityEvent(
            event_type=EventType.ERROR,
            error_type=error_type,
            metadata=metadata or {},
        ))

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the trace and return summary.
        Emits structured log line.
        """
        total_latency = (time.time() - self.start_time) * 1000

        # Aggregate metrics
        total_input = sum(e.input_tokens for e in self.events)
        total_output = sum(e.output_tokens for e in self.events)
        total_cost = sum(e.cost_estimate for e in self.events)
        model_calls = [e for e in self.events if e.event_type == EventType.MODEL_CALL]
        cache_hits = [e for e in self.events if e.event_type == EventType.CACHE_HIT]
        fallbacks = [e for e in self.events if e.event_type == EventType.FALLBACK_TRIGGERED]
        errors = [e for e in self.events if e.event_type == EventType.ERROR]

        summary = {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "total_latency_ms": round(total_latency, 1),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": round(total_cost, 8),
            "model_calls": len(model_calls),
            "cache_hit": len(cache_hits) > 0,
            "cache_tier": cache_hits[0].cache_tier if cache_hits else None,
            "fallbacks": len(fallbacks),
            "errors": len(errors),
            "models_used": list(set(e.model_id for e in model_calls)),
        }

        # Structured log emit
        logger.info(
            json.dumps(
                {"event": "request_complete", **summary},
                separators=(",", ":"),
            )
        )

        return summary


# ============================================================
# METRICS AGGREGATOR
# ============================================================

class MetricsAggregator:
    """
    Lightweight metrics aggregation.

    Keeps bounded ring buffer of recent request summaries
    and running counters for dashboard consumption.
    """

    def __init__(self, buffer_size: int = 500):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._counters = {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallbacks": 0,
            "errors": 0,
            "downgrades": 0,
        }
        self._model_counters: Dict[str, Dict[str, int]] = {}
        self._latency_buckets = {
            "p50": [],
            "p95": [],
            "p99": [],
        }
        self._max_latency_samples = 200

    def record_request(self, summary: Dict[str, Any]):
        """Record a completed request summary."""
        self._buffer.append(summary)
        self._counters["total_requests"] += 1
        self._counters["total_input_tokens"] += summary.get("total_input_tokens", 0)
        self._counters["total_output_tokens"] += summary.get("total_output_tokens", 0)
        self._counters["total_cost"] += summary.get("total_cost", 0.0)

        if summary.get("cache_hit"):
            self._counters["cache_hits"] += 1
        else:
            self._counters["cache_misses"] += 1

        self._counters["fallbacks"] += summary.get("fallbacks", 0)
        self._counters["errors"] += summary.get("errors", 0)

        # Track per-model usage
        for model in summary.get("models_used", []):
            if model not in self._model_counters:
                self._model_counters[model] = {"requests": 0, "tokens": 0}
            self._model_counters[model]["requests"] += 1
            self._model_counters[model]["tokens"] += (
                summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0)
            )

        # Track latency
        latency = summary.get("total_latency_ms", 0)
        if latency > 0:
            self._latency_buckets["p50"].append(latency)
            self._latency_buckets["p95"].append(latency)
            self._latency_buckets["p99"].append(latency)
            # Trim
            for key in self._latency_buckets:
                if len(self._latency_buckets[key]) > self._max_latency_samples:
                    self._latency_buckets[key] = self._latency_buckets[key][-self._max_latency_samples:]

    def _percentile(self, data: List[float], p: float) -> float:
        """Compute percentile from sorted data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        total_reqs = self._counters["total_requests"]
        cache_total = self._counters["cache_hits"] + self._counters["cache_misses"]

        return {
            "counters": {
                **self._counters,
                "total_cost": round(self._counters["total_cost"], 6),
            },
            "rates": {
                "cache_hit_rate": (
                    round(self._counters["cache_hits"] / cache_total, 3)
                    if cache_total > 0 else 0.0
                ),
                "error_rate": (
                    round(self._counters["errors"] / total_reqs, 3)
                    if total_reqs > 0 else 0.0
                ),
                "fallback_rate": (
                    round(self._counters["fallbacks"] / total_reqs, 3)
                    if total_reqs > 0 else 0.0
                ),
            },
            "latency": {
                "p50": round(self._percentile(self._latency_buckets["p50"], 50), 1),
                "p95": round(self._percentile(self._latency_buckets["p95"], 95), 1),
                "p99": round(self._percentile(self._latency_buckets["p99"], 99), 1),
            },
            "models": self._model_counters,
            "recent_count": len(self._buffer),
        }


# ============================================================
# OBSERVABILITY HUB (Singleton)
# ============================================================

class ObservabilityHub:
    """
    Central observability hub.

    Usage:
        hub = get_observability_hub()
        tracer = hub.start_request("session_123", "req_abc")
        tracer.record_cache_miss()
        tracer.record_model_call(...)
        summary = tracer.finalize()
        hub.record(summary)
    """

    def __init__(self):
        self.metrics = MetricsAggregator()

    def start_request(self, session_id: str, request_id: str) -> RequestTracer:
        """Create a new request tracer."""
        return RequestTracer(session_id=session_id, request_id=request_id)

    def record(self, summary: Dict[str, Any]):
        """Record a completed request summary into metrics."""
        self.metrics.record_request(summary)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return self.metrics.get_metrics()


# ── Module-level singleton ──
_hub: Optional[ObservabilityHub] = None


def get_observability_hub() -> ObservabilityHub:
    """Get or create the singleton ObservabilityHub."""
    global _hub
    if _hub is None:
        _hub = ObservabilityHub()
    return _hub
