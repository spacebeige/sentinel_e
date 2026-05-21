"""
============================================================
OrchestrationRun — Canonical Cognitive Identity Layer
============================================================
Sentinel-E v8.0 — Persistent Hybrid Cognitive Runtime

Every execution path ATTACHES to an OrchestrationRun.
This becomes the central cognitive identity for each query.

Design Principles:
  - Pure Python dataclasses (no external deps)
  - Fully serializable to JSON
  - Thread-safe event appending
  - Additive: zero changes to existing execution paths
  - Graceful degradation: all fields optional / safe defaults

Lifecycle:
  CREATED → ROUTING → EXECUTING → DEBATING → SYNTHESIZING
  → REFLECTING → COMPLETED | FAILED | RECOVERED
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enumerations ──────────────────────────────────────────────

class RunLifecycle(str, Enum):
    """Top-level lifecycle state of an OrchestrationRun."""
    CREATED       = "created"
    ROUTING       = "routing"
    EXECUTING     = "executing"
    DEBATING      = "debating"
    SYNTHESIZING  = "synthesizing"
    REFLECTING    = "reflecting"
    COMPLETED     = "completed"
    FAILED        = "failed"
    RECOVERED     = "recovered"


class CognitivePhase(str, Enum):
    """
    Fine-grained cognitive phase within the orchestration pipeline.
    Maps 1:1 to visible UI state labels.
    """
    OBSERVE         = "observe"
    ANALYZE         = "analyze"
    ROUTE           = "route"
    RETRIEVE_MEMORY = "retrieve_memory"
    SPAWN_AGENTS    = "spawn_agents"
    DEBATE          = "debate"
    VERIFY          = "verify"
    SYNTHESIZE      = "synthesize"
    REFLECT         = "reflect"
    STORE_SNAPSHOT  = "store_snapshot"
    STREAM_VIZ      = "stream_visualization"
    IDLE            = "idle"


# ── Human-readable labels for the frontend ───────────────────
PHASE_LABELS: Dict[CognitivePhase, str] = {
    CognitivePhase.OBSERVE:         "🔍 Observing — parsing intent",
    CognitivePhase.ANALYZE:         "🧠 Analyzing — routing decision in progress",
    CognitivePhase.ROUTE:           "🗺️ Routing — selecting execution path",
    CognitivePhase.RETRIEVE_MEMORY: "💾 Retrieving Memory — injecting context",
    CognitivePhase.SPAWN_AGENTS:    "⚡ Spawning Agents — parallel model execution",
    CognitivePhase.DEBATE:          "⚔️ Debating — multi-model reasoning",
    CognitivePhase.VERIFY:          "✅ Verifying — contradiction check",
    CognitivePhase.SYNTHESIZE:      "🔗 Synthesizing — ensemble convergence",
    CognitivePhase.REFLECT:         "🪞 Reflecting — metacognitive analysis",
    CognitivePhase.STORE_SNAPSHOT:  "💾 Storing Snapshot — persisting cognitive state",
    CognitivePhase.STREAM_VIZ:      "📡 Streaming — broadcasting to frontend",
    CognitivePhase.IDLE:            "⏸️ Idle",
}


class EventSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


# ── Core Data Structures ──────────────────────────────────────

@dataclass
class CognitiveEvent:
    """
    A single structured event emitted during orchestration.
    These are the "neural signals" of the cognitive runtime.
    """
    event_type: str          # e.g. "routing_decision_made"
    phase: str               # CognitivePhase value
    payload: Dict[str, Any]  # event-specific data
    severity: str = EventSeverity.INFO
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""         # filled in when attached to a run

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "phase": self.phase,
            "payload": self.payload,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


@dataclass
class ProviderTelemetry:
    """Latency and token metrics for a single provider/model call."""
    model_id: str
    model_name: str
    provider: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    succeeded: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "succeeded": self.succeeded,
            "error": self.error,
        }


@dataclass
class MemoryRetrieval:
    """A single memory retrieval event during context building."""
    layer: str               # "working", "episodic", "semantic", "deliberative", "tactical"
    key: str
    content_preview: str
    relevance_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "key": self.key,
            "content_preview": self.content_preview[:200],
            "relevance_score": round(self.relevance_score, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class DebateRoundSnapshot:
    """Snapshot of a single debate round for the event timeline."""
    round_number: int
    positions: List[Dict[str, Any]]
    contradiction_density: float = 0.0
    consensus_delta: float = 0.0
    drift_index: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "positions": self.positions,
            "contradiction_density": round(self.contradiction_density, 4),
            "consensus_delta": round(self.consensus_delta, 4),
            "drift_index": round(self.drift_index, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class ConfidenceSnapshot:
    """A single point in confidence evolution."""
    phase: str
    value: float
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "value": round(self.value, 4),
            "method": self.method,
            "timestamp": self.timestamp,
        }


@dataclass
class RecoveryState:
    """Tracks fallback and recovery operations."""
    triggered: bool = False
    reason: str = ""
    fallback_path: str = ""
    recovery_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "fallback_path": self.fallback_path,
            "recovery_timestamp": self.recovery_timestamp,
        }


# ── Main OrchestrationRun ─────────────────────────────────────

class OrchestrationRun:
    """
    The canonical cognitive identity layer for a single request execution.

    Every execution path (single_model, fast_standard, ensemble,
    legacy_fallback) ATTACHES to one OrchestrationRun instance.

    This becomes the central nervous system: routing decisions,
    phase transitions, events, telemetry, memory retrievals,
    debate states, and synthesis all flow through here.

    Thread-safe: event appending uses a reentrant lock.
    Serializable: to_summary() and to_frontend_dict() produce
    JSON-safe dicts.
    """

    def __init__(
        self,
        chat_id: str = "",
        user_id: str = "",
        query_preview: str = "",
        execution_path: str = "unknown",
    ):
        self.orchestration_run_id: str = str(uuid.uuid4())
        self.chat_id: str = chat_id
        self.user_id: str = user_id
        self.query_preview: str = query_preview[:100]  # never store full query in metadata
        self.execution_path: str = execution_path

        # ── Lifecycle & Phase ─────────────────────────────────
        self.lifecycle_state: RunLifecycle = RunLifecycle.CREATED
        self.cognitive_phase: CognitivePhase = CognitivePhase.IDLE
        self.phase_label: str = PHASE_LABELS[CognitivePhase.IDLE]

        # ── Timing ───────────────────────────────────────────
        self._start_monotonic: float = time.monotonic()
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.total_latency_ms: float = 0.0

        # ── Routing ───────────────────────────────────────────
        self.routing_decision: Dict[str, Any] = {}

        # ── Agents & Providers ────────────────────────────────
        self.active_agents: List[str] = []
        self.provider_telemetry: List[ProviderTelemetry] = []

        # ── Memory ───────────────────────────────────────────
        self.memory_retrievals: List[MemoryRetrieval] = []

        # ── Debate ───────────────────────────────────────────
        self.debate_rounds: List[DebateRoundSnapshot] = []
        self.models_executed: int = 0
        self.models_succeeded: int = 0
        self.models_failed: int = 0

        # ── Confidence ───────────────────────────────────────
        self.confidence_evolution: List[ConfidenceSnapshot] = []
        self.final_confidence: float = 0.0

        # ── Contradiction & Verification ─────────────────────
        self.contradiction_density: float = 0.0
        self.verification_events: List[Dict[str, Any]] = []

        # ── Synthesis & Reflection ────────────────────────────
        self.synthesis_state: Dict[str, Any] = {}
        self.reflection_analysis: str = ""

        # ── Recovery ─────────────────────────────────────────
        self.recovery_state: RecoveryState = RecoveryState()

        # ── Event Timeline (thread-safe) ──────────────────────
        self._lock: threading.RLock = threading.RLock()
        self.event_timeline: List[CognitiveEvent] = []

        # ── Emit creation event ───────────────────────────────
        self.emit_event(
            event_type="orchestration_started",
            payload={
                "run_id": self.orchestration_run_id,
                "chat_id": chat_id,
                "execution_path": execution_path,
                "query_preview": self.query_preview,
            },
        )

    # ── Phase Transitions ─────────────────────────────────────

    def transition_to(self, phase: CognitivePhase, payload: Optional[Dict] = None) -> None:
        """Atomically update cognitive phase and emit transition event."""
        with self._lock:
            previous = self.cognitive_phase
            self.cognitive_phase = phase
            self.phase_label = PHASE_LABELS.get(phase, phase.value)

            # Update lifecycle based on phase
            if phase == CognitivePhase.ROUTE:
                self.lifecycle_state = RunLifecycle.ROUTING
            elif phase == CognitivePhase.DEBATE:
                self.lifecycle_state = RunLifecycle.DEBATING
            elif phase == CognitivePhase.SYNTHESIZE:
                self.lifecycle_state = RunLifecycle.SYNTHESIZING
            elif phase == CognitivePhase.REFLECT:
                self.lifecycle_state = RunLifecycle.REFLECTING
            elif phase in (
                CognitivePhase.ANALYZE,
                CognitivePhase.RETRIEVE_MEMORY,
                CognitivePhase.SPAWN_AGENTS,
                CognitivePhase.VERIFY,
                CognitivePhase.STORE_SNAPSHOT,
                CognitivePhase.STREAM_VIZ,
            ):
                self.lifecycle_state = RunLifecycle.EXECUTING

            self._append_event(CognitiveEvent(
                event_type="phase_transition",
                phase=phase.value,
                payload={
                    "previous_phase": previous.value,
                    "current_phase": phase.value,
                    "phase_label": self.phase_label,
                    **(payload or {}),
                },
                run_id=self.orchestration_run_id,
            ))

    # ── Event Emission ────────────────────────────────────────

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        severity: str = EventSeverity.INFO,
    ) -> CognitiveEvent:
        """Emit a structured cognitive event onto the timeline."""
        event = CognitiveEvent(
            event_type=event_type,
            phase=self.cognitive_phase.value,
            payload=payload,
            severity=severity,
            run_id=self.orchestration_run_id,
        )
        self._append_event(event)
        return event

    def _append_event(self, event: CognitiveEvent) -> None:
        with self._lock:
            self.event_timeline.append(event)

    # ── Telemetry Recording ───────────────────────────────────

    def record_provider_call(
        self,
        model_id: str,
        model_name: str,
        provider: str,
        latency_ms: float,
        succeeded: bool = True,
        error: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record telemetry for a model/provider call."""
        with self._lock:
            telemetry = ProviderTelemetry(
                model_id=model_id,
                model_name=model_name,
                provider=provider,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                succeeded=succeeded,
                error=error,
            )
            self.provider_telemetry.append(telemetry)

        self.emit_event(
            event_type="provider_called",
            payload={
                "model_id": model_id,
                "model_name": model_name,
                "provider": provider,
                "latency_ms": round(latency_ms, 1),
                "succeeded": succeeded,
                "error": error,
            },
            severity=EventSeverity.WARNING if not succeeded else EventSeverity.INFO,
        )

    def record_memory_retrieval(
        self,
        layer: str,
        key: str,
        content_preview: str,
        relevance_score: float = 1.0,
    ) -> None:
        """Record a memory retrieval event."""
        with self._lock:
            retrieval = MemoryRetrieval(
                layer=layer,
                key=key,
                content_preview=content_preview,
                relevance_score=relevance_score,
            )
            self.memory_retrievals.append(retrieval)

        self.emit_event(
            event_type="memory_retrieved",
            payload={
                "layer": layer,
                "key": key,
                "relevance_score": round(relevance_score, 4),
            },
        )

    def record_debate_round(
        self,
        round_number: int,
        positions: List[Dict[str, Any]],
        contradiction_density: float = 0.0,
        consensus_delta: float = 0.0,
        drift_index: float = 0.0,
    ) -> None:
        """Record a completed debate round."""
        with self._lock:
            snapshot = DebateRoundSnapshot(
                round_number=round_number,
                positions=positions,
                contradiction_density=contradiction_density,
                consensus_delta=consensus_delta,
                drift_index=drift_index,
            )
            self.debate_rounds.append(snapshot)
            self.contradiction_density = contradiction_density

        event_type = "contradiction_detected" if contradiction_density > 0.6 else "debate_round_completed"
        self.emit_event(
            event_type=event_type,
            payload={
                "round": round_number,
                "contradiction_density": round(contradiction_density, 4),
                "consensus_delta": round(consensus_delta, 4),
                "drift_index": round(drift_index, 4),
                "positions_count": len(positions),
            },
            severity=EventSeverity.WARNING if contradiction_density > 0.6 else EventSeverity.INFO,
        )

    def record_confidence_snapshot(
        self,
        phase: str,
        value: float,
        method: str = "calibrated",
    ) -> None:
        """Record a confidence evolution checkpoint."""
        with self._lock:
            snap = ConfidenceSnapshot(phase=phase, value=value, method=method)
            self.confidence_evolution.append(snap)
            self.final_confidence = value

        self.emit_event(
            event_type="confidence_shift",
            payload={
                "phase": phase,
                "value": round(value, 4),
                "method": method,
            },
        )

    def record_synthesis_start(self, method: str, model_count: int) -> None:
        self.synthesis_state = {
            "method": method,
            "model_count": model_count,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed": False,
        }
        self.emit_event(
            event_type="synthesis_started",
            payload={"method": method, "model_count": model_count},
        )

    def record_synthesis_complete(self, output_length: int) -> None:
        self.synthesis_state["completed"] = True
        self.synthesis_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.synthesis_state["output_length"] = output_length
        self.emit_event(
            event_type="synthesis_completed",
            payload={"output_length": output_length},
        )

    def record_reflection(self, reflection: str) -> None:
        self.reflection_analysis = reflection
        self.emit_event(
            event_type="reflection_generated",
            payload={"reflection_preview": reflection[:200]},
        )

    def record_routing_decision(self, routing_decision: Dict[str, Any]) -> None:
        self.routing_decision = routing_decision
        self.execution_path = routing_decision.get("path", self.execution_path)
        self.emit_event(
            event_type="routing_decision_made",
            payload=routing_decision,
        )

    def trigger_recovery(self, reason: str, fallback_path: str) -> None:
        """Mark run as entering recovery mode."""
        self.recovery_state = RecoveryState(
            triggered=True,
            reason=reason,
            fallback_path=fallback_path,
            recovery_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.lifecycle_state = RunLifecycle.RECOVERED
        self.emit_event(
            event_type="recovery_initiated",
            payload={"reason": reason, "fallback_path": fallback_path},
            severity=EventSeverity.WARNING,
        )

    def mark_failed(self, error: str, error_code: str = "UNKNOWN") -> None:
        """Transition run to FAILED state."""
        self.lifecycle_state = RunLifecycle.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.total_latency_ms = (time.monotonic() - self._start_monotonic) * 1000
        self.emit_event(
            event_type="orchestration_failed",
            payload={"error": error, "error_code": error_code},
            severity=EventSeverity.CRITICAL,
        )

    def mark_completed(self) -> None:
        """Transition run to COMPLETED state and finalize timing."""
        self.lifecycle_state = RunLifecycle.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.total_latency_ms = (time.monotonic() - self._start_monotonic) * 1000
        self.emit_event(
            event_type="snapshot_saved",
            payload={
                "total_latency_ms": round(self.total_latency_ms, 1),
                "events_emitted": len(self.event_timeline),
                "final_confidence": round(self.final_confidence, 4),
            },
        )

    # ── Serialization ─────────────────────────────────────────

    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for API responses and admin feeds."""
        return {
            "orchestration_run_id": self.orchestration_run_id,
            "chat_id": self.chat_id,
            "lifecycle_state": self.lifecycle_state.value,
            "cognitive_phase": self.cognitive_phase.value,
            "phase_label": self.phase_label,
            "execution_path": self.execution_path,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "models_executed": self.models_executed,
            "models_succeeded": self.models_succeeded,
            "final_confidence": round(self.final_confidence, 4),
            "contradiction_density": round(self.contradiction_density, 4),
            "debate_rounds": len(self.debate_rounds),
            "events_count": len(self.event_timeline),
            "memory_retrievals": len(self.memory_retrievals),
            "recovery_triggered": self.recovery_state.triggered,
        }

    def to_frontend_dict(self) -> Dict[str, Any]:
        """
        Full serialization for frontend consumption.
        Attached to omega_metadata.orchestration_run in every response.
        """
        with self._lock:
            return {
                "orchestration_run_id": self.orchestration_run_id,
                "lifecycle_state": self.lifecycle_state.value,
                "cognitive_phase": self.cognitive_phase.value,
                "phase_label": self.phase_label,
                "execution_path": self.execution_path,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "total_latency_ms": round(self.total_latency_ms, 1),
                "routing_decision": self.routing_decision,
                "active_agents": self.active_agents,
                "models_executed": self.models_executed,
                "models_succeeded": self.models_succeeded,
                "models_failed": self.models_failed,
                "provider_telemetry": [p.to_dict() for p in self.provider_telemetry],
                "confidence_evolution": [c.to_dict() for c in self.confidence_evolution],
                "final_confidence": round(self.final_confidence, 4),
                "contradiction_density": round(self.contradiction_density, 4),
                "debate_rounds_timeline": [d.to_dict() for d in self.debate_rounds],
                "memory_retrievals": [m.to_dict() for m in self.memory_retrievals],
                "verification_events": self.verification_events,
                "synthesis_state": self.synthesis_state,
                "reflection_analysis": self.reflection_analysis[:500] if self.reflection_analysis else "",
                "recovery_state": self.recovery_state.to_dict(),
                "event_timeline": [e.to_dict() for e in self.event_timeline[-50:]],  # cap at 50 for wire size
                "event_count": len(self.event_timeline),
            }


# ── Global Run Registry ───────────────────────────────────────

class OrchestrationRunRegistry:
    """
    In-memory registry of recent OrchestrationRun instances.
    Used by admin endpoints to expose live run state.
    Thread-safe, max 200 runs retained (LRU).
    """

    MAX_RUNS = 200

    def __init__(self):
        self._runs: Dict[str, OrchestrationRun] = {}
        self._order: List[str] = []  # insertion order for LRU eviction
        self._lock = threading.RLock()

    def register(self, run: OrchestrationRun) -> None:
        with self._lock:
            run_id = run.orchestration_run_id
            self._runs[run_id] = run
            if run_id not in self._order:
                self._order.append(run_id)
            # Evict oldest if over capacity
            while len(self._order) > self.MAX_RUNS:
                oldest = self._order.pop(0)
                self._runs.pop(oldest, None)

    def get(self, run_id: str) -> Optional[OrchestrationRun]:
        with self._lock:
            return self._runs.get(run_id)

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return summaries of the most recent N runs."""
        with self._lock:
            recent_ids = self._order[-limit:][::-1]  # newest first
            return [
                self._runs[rid].to_summary()
                for rid in recent_ids
                if rid in self._runs
            ]

    def get_active(self) -> List[Dict[str, Any]]:
        """Return summaries of currently running (non-terminal) runs."""
        with self._lock:
            terminal = {RunLifecycle.COMPLETED, RunLifecycle.FAILED}
            return [
                run.to_summary()
                for run in self._runs.values()
                if run.lifecycle_state not in terminal
            ]

    def get_latest_for_chat(self, chat_id: str) -> Optional[OrchestrationRun]:
        """Return the newest run attached to a chat/session identifier."""
        if not chat_id:
            return None
        with self._lock:
            for run_id in reversed(self._order):
                run = self._runs.get(run_id)
                if run and str(run.chat_id) == str(chat_id):
                    return run
        return None


# ── Module-level singleton ─────────────────────────────────────
_registry: Optional[OrchestrationRunRegistry] = None


def get_run_registry() -> OrchestrationRunRegistry:
    """Get or create the global run registry singleton."""
    global _registry
    if _registry is None:
        _registry = OrchestrationRunRegistry()
    return _registry


def create_orchestration_run(
    chat_id: str = "",
    user_id: str = "",
    query_preview: str = "",
    execution_path: str = "unknown",
) -> OrchestrationRun:
    """
    Factory: create a new OrchestrationRun and register it globally.
    Called at the very start of _run_sentinel_core().
    """
    run = OrchestrationRun(
        chat_id=chat_id,
        user_id=user_id,
        query_preview=query_preview,
        execution_path=execution_path,
    )
    get_run_registry().register(run)
    return run
