"""
============================================================
Layered Cognitive Memory — Sentinel-E v8.0
============================================================
Wraps the existing MemoryEngine with a 5-layer cognitive
memory architecture.

EXISTING CODE IS NOT MODIFIED. This module wraps and extends.

Memory Layers:
  1. Working Memory    — active run context (per-request)
  2. Episodic Memory   — session continuity (existing MemoryEngine)
  3. Semantic Memory   — persistent knowledge (existing UserMemory DB)
  4. Deliberative Memory — debate history + contradiction logs
  5. Tactical Memory   — routing strategy history

How it integrates:
  - LayeredMemoryContext wraps the existing MemoryEngine instance
  - build_layered_context() produces a structured context injection
    that EXTENDS (not replaces) MemoryEngine.build_prompt_context()
  - OrchestrationRun.record_memory_retrieval() is called for each
    layer that contributes context
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("LayeredMemory")


# ── Working Memory ─────────────────────────────────────────────

@dataclass
class WorkingMemoryEntry:
    """Active context for the current request."""
    key: str
    value: str
    source: str  # "rag", "user_context", "routing", "debate"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkingMemory:
    """
    Per-request transient memory.
    Reset on each new request; not persisted.
    """
    def __init__(self):
        self._entries: List[WorkingMemoryEntry] = []

    def add(self, key: str, value: str, source: str = "unknown") -> None:
        self._entries.append(WorkingMemoryEntry(key=key, value=value, source=source))

    def get_context_str(self, max_chars: int = 800) -> str:
        """Build a compact context string from working memory entries."""
        if not self._entries:
            return ""
        parts = []
        total = 0
        for entry in self._entries:
            snippet = f"[{entry.source}] {entry.key}: {entry.value}"
            if total + len(snippet) > max_chars:
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n".join(parts)

    def to_list(self) -> List[Dict[str, str]]:
        return [{"key": e.key, "value": e.value[:200], "source": e.source} for e in self._entries]


# ── Deliberative Memory ────────────────────────────────────────

@dataclass
class DeliberativeMemoryEntry:
    """A stored debate/contradiction record for future routing influence."""
    topic_hash: str
    contradiction_density: float
    consensus_reached: bool
    drift_index: float
    key_conflicts: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    chat_id: str = ""


class DeliberativeMemory:
    """
    Stores debate history to influence future debate configuration.
    Held in memory with an optional JSON snapshot path.
    Max 100 entries (LRU).
    """
    _MAX_ENTRIES = 100

    def __init__(self):
        self._entries: List[DeliberativeMemoryEntry] = []

    def record(
        self,
        topic_hash: str,
        contradiction_density: float,
        consensus_reached: bool,
        drift_index: float,
        key_conflicts: List[str],
        chat_id: str = "",
    ) -> None:
        self._entries.append(DeliberativeMemoryEntry(
            topic_hash=topic_hash,
            contradiction_density=contradiction_density,
            consensus_reached=consensus_reached,
            drift_index=drift_index,
            key_conflicts=key_conflicts[:5],
            chat_id=chat_id,
        ))
        # LRU eviction
        if len(self._entries) > self._MAX_ENTRIES:
            self._entries = self._entries[-self._MAX_ENTRIES:]

    def get_prior_context(self, topic_hash: str) -> Optional[str]:
        """
        Return a formatted string describing prior debate outcomes
        for a similar topic. Used to prime debate calibration.
        """
        matches = [e for e in self._entries if e.topic_hash == topic_hash]
        if not matches:
            return None
        e = matches[-1]  # most recent
        status = "reached consensus" if e.consensus_reached else "remained unresolved"
        conflicts = "; ".join(e.key_conflicts[:3])
        return (
            f"Prior debate on similar topic {status}. "
            f"Contradiction density: {e.contradiction_density:.2f}, "
            f"drift: {e.drift_index:.2f}. "
            f"Key conflicts: {conflicts}."
        )

    def recent_entries(self, limit: int = 5) -> List[Dict[str, Any]]:
        return [
            {
                "topic_hash": e.topic_hash,
                "contradiction_density": e.contradiction_density,
                "consensus_reached": e.consensus_reached,
                "drift_index": e.drift_index,
                "timestamp": e.timestamp,
            }
            for e in self._entries[-limit:]
        ]


# ── Tactical Memory ────────────────────────────────────────────

@dataclass
class TacticalMemoryEntry:
    """A routing strategy history entry."""
    query_complexity: str
    execution_path: str
    model_count: int
    latency_ms: float
    confidence: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TacticalMemory:
    """
    Stores routing strategy history to influence future routing decisions.
    """
    _MAX_ENTRIES = 200

    def __init__(self):
        self._entries: List[TacticalMemoryEntry] = []

    def record(
        self,
        query_complexity: str,
        execution_path: str,
        model_count: int,
        latency_ms: float,
        confidence: float,
        success: bool = True,
    ) -> None:
        self._entries.append(TacticalMemoryEntry(
            query_complexity=query_complexity,
            execution_path=execution_path,
            model_count=model_count,
            latency_ms=latency_ms,
            confidence=confidence,
            success=success,
        ))
        if len(self._entries) > self._MAX_ENTRIES:
            self._entries = self._entries[-self._MAX_ENTRIES:]

    def get_strategy_hint(self, query_complexity: str) -> Optional[str]:
        """Return a hint about routing strategy based on historical performance."""
        relevant = [
            e for e in self._entries[-50:]
            if e.query_complexity == query_complexity and e.success
        ]
        if not relevant:
            return None
        avg_confidence = sum(e.confidence for e in relevant) / len(relevant)
        avg_latency = sum(e.latency_ms for e in relevant) / len(relevant)
        best_path = max(
            set(e.execution_path for e in relevant),
            key=lambda p: sum(1 for e in relevant if e.execution_path == p)
        )
        return (
            f"Historical routing for {query_complexity} queries: "
            f"best path={best_path}, avg confidence={avg_confidence:.2f}, "
            f"avg latency={avg_latency:.0f}ms"
        )

    def get_stats(self) -> Dict[str, Any]:
        if not self._entries:
            return {"total_routed": 0}
        recent = self._entries[-20:]
        return {
            "total_routed": len(self._entries),
            "recent_success_rate": sum(1 for e in recent if e.success) / len(recent),
            "avg_confidence": sum(e.confidence for e in recent) / len(recent),
            "avg_latency_ms": sum(e.latency_ms for e in recent) / len(recent),
        }


# ── Layered Memory Context Builder ────────────────────────────

class LayeredMemoryContext:
    """
    Wraps an existing MemoryEngine instance and enriches it with
    deliberative and tactical layers.

    Importantly: does NOT replace MemoryEngine. Calls through to
    the existing engine for episodic context, then appends additional
    layers.

    Usage (in main.py, additive injection):
        layered = LayeredMemoryContext(memory, user_id=user_id)
        layered.working.add("rag_context", rag_summary, source="rag")
        context_str = layered.build_layered_context(run=orch_run)
        # context_str is appended to history as system message
    """

    def __init__(
        self,
        memory_engine,  # existing MemoryEngine instance
        user_id: str = "",
        deliberative: Optional[DeliberativeMemory] = None,
        tactical: Optional[TacticalMemory] = None,
        behavioral_profile_hint: str = "",
    ):
        self._engine = memory_engine
        self.user_id = user_id
        self.working = WorkingMemory()
        self.deliberative = deliberative or DeliberativeMemory()
        self.tactical = tactical or TacticalMemory()
        self.behavioral_profile_hint = behavioral_profile_hint

    def build_layered_context(
        self,
        query: str = "",
        topic_hash: str = "",
        query_complexity: str = "unknown",
        run=None,  # OrchestrationRun — optional, for recording retrievals
        max_chars: int = 1200,
    ) -> str:
        """
        Build a merged context string from all memory layers.

        Layer priority (highest influence first):
          1. Working (current run context)
          2. Episodic (existing MemoryEngine.build_prompt_context())
          3. Deliberative (prior debate priors)
          4. Tactical (routing strategy hint)
          5. Behavioral (Adaptive User Profile)
          6. Semantic (existing UserMemory — handled by MemoryEngine)

        Returns a single system prompt injection string.
        """
        parts: List[str] = []
        retrieved_layers: List[str] = []

        # ── Layer 1: Working Memory ──────────────────────────
        working_ctx = self.working.get_context_str(max_chars=400)
        if working_ctx:
            parts.append(f"[Working Context]\n{working_ctx}")
            retrieved_layers.append("working")

        # ── Layer 2: Episodic Memory (existing engine) ────────
        try:
            episodic_ctx = self._engine.build_prompt_context()
            if episodic_ctx and episodic_ctx.strip():
                parts.append(f"[Session Memory]\n{episodic_ctx}")
                retrieved_layers.append("episodic")
        except Exception as e:
            logger.debug(f"[LayeredMemory] Episodic retrieval failed: {e}")

        # ── Layer 3: Deliberative Memory ──────────────────────
        if topic_hash:
            deliberative_ctx = self.deliberative.get_prior_context(topic_hash)
            if deliberative_ctx:
                parts.append(f"[Prior Debate Context]\n{deliberative_ctx}")
                retrieved_layers.append("deliberative")

        # ── Layer 4: Tactical Memory ──────────────────────────
        if query_complexity and query_complexity != "unknown":
            tactical_hint = self.tactical.get_strategy_hint(query_complexity)
            if tactical_hint:
                parts.append(f"[Routing Intelligence]\n{tactical_hint}")
                retrieved_layers.append("tactical")

        # ── Layer 5: Behavioral Memory ────────────────────────
        if getattr(self, "behavioral_profile_hint", ""):
            parts.append(f"[Behavioral Context]\n{self.behavioral_profile_hint}")
            retrieved_layers.append("behavioral")

        # ── Record retrievals on OrchestrationRun ─────────────
        if run is not None:
            for layer in retrieved_layers:
                try:
                    run.record_memory_retrieval(
                        layer=layer,
                        key=f"{layer}_context",
                        content_preview=f"Context from {layer} memory layer",
                        relevance_score=1.0,
                    )
                except Exception:
                    pass

        if not parts:
            return ""

        # Merge and cap
        merged = "\n\n".join(parts)
        if len(merged) > max_chars:
            merged = merged[:max_chars] + "...[truncated]"
        return merged

    def get_memory_state(self) -> Dict[str, Any]:
        """Serializable snapshot of all memory layers for admin/observability."""
        return {
            "working": self.working.to_list(),
            "deliberative_recent": self.deliberative.recent_entries(),
            "tactical_stats": self.tactical.get_stats(),
        }


# ── Global deliberative + tactical memory (singleton per process) ─

_deliberative_memory: Optional[DeliberativeMemory] = None
_tactical_memory: Optional[TacticalMemory] = None


def get_deliberative_memory() -> DeliberativeMemory:
    global _deliberative_memory
    if _deliberative_memory is None:
        _deliberative_memory = DeliberativeMemory()
    return _deliberative_memory


def get_tactical_memory() -> TacticalMemory:
    global _tactical_memory
    if _tactical_memory is None:
        _tactical_memory = TacticalMemory()
    return _tactical_memory


def create_layered_context(memory_engine, user_id: str = "", behavioral_profile_hint: str = "") -> LayeredMemoryContext:
    """
    Factory: create a LayeredMemoryContext wrapping an existing MemoryEngine,
    sharing the global deliberative and tactical memory stores.
    """
    return LayeredMemoryContext(
        memory_engine=memory_engine,
        user_id=user_id,
        deliberative=get_deliberative_memory(),
        tactical=get_tactical_memory(),
        behavioral_profile_hint=behavioral_profile_hint,
    )
