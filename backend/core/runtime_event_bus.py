"""
============================================================
Runtime Event Bus — Sentinel-E v8.0
============================================================
Non-blocking, per-run event queue with optional Redis mirror.

Architecture:
  - Each OrchestrationRun has an associated EventBus channel.
  - Events are appended to an in-memory asyncio.Queue.
  - SSE endpoint drains the queue for live frontend streaming.
  - Redis mirror is optional (non-fatal if unavailable).

Design:
  - Zero blocking on the critical request path.
  - Graceful degradation: if queue is full, event is dropped (logged).
  - No external dependencies beyond existing Redis client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger("RuntimeEventBus")

# Maximum events buffered per run before oldest are dropped
_MAX_QUEUE_SIZE = 256


class RunEventBus:
    """
    Per-run event bus that supports async iteration (SSE streaming).

    Usage:
        bus = RunEventBus(run_id)
        bus.publish({"event_type": "debate_round_completed", ...})
        async for event in bus.stream(timeout=60):
            yield f"data: {json.dumps(event)}\\n\\n"
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._closed = False
        self._created_at = time.monotonic()

    def publish(self, event_dict: Dict[str, Any]) -> None:
        """
        Non-blocking publish. If queue is full, oldest event is dropped.
        Called from synchronous orchestration code via run_coroutine_threadsafe
        or directly from async context.
        """
        if self._closed:
            return

        # Ensure timestamp and run_id are always present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        event_dict.setdefault("run_id", self.run_id)

        # ── Payload Compression (Truncate large text fields) ──
        # To optimize websocket/SSE traffic, limit large strings.
        for key in ["content", "output", "reasoning", "prompt", "chunk"]:
            if key in event_dict and isinstance(event_dict[key], str) and len(event_dict[key]) > 2000:
                event_dict[key] = event_dict[key][:2000] + "... [truncated]"

        try:
            self._queue.put_nowait(event_dict)
        except asyncio.QueueFull:
            # Drop oldest to make room for newest (sliding window)
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event_dict)
            except Exception:
                logger.debug(f"[EventBus] Queue overflow for run {self.run_id}")

    async def publish_async(self, event_dict: Dict[str, Any]) -> None:
        """Async publish — awaits if queue is full."""
        if self._closed:
            return
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        event_dict.setdefault("run_id", self.run_id)

        for key in ["content", "output", "reasoning", "prompt", "chunk"]:
            if key in event_dict and isinstance(event_dict[key], str) and len(event_dict[key]) > 2000:
                event_dict[key] = event_dict[key][:2000] + "... [truncated]"
        try:
            await asyncio.wait_for(self._queue.put(event_dict), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            logger.debug(f"[EventBus] Async publish failed for run {self.run_id}")

    async def stream(self, timeout: float = 120.0) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that yields events as they arrive.
        Yields a heartbeat every 15s to keep SSE connection alive.
        Stops after `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._closed:
            try:
                remaining = max(0.0, deadline - time.monotonic())
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=min(15.0, remaining),
                )
                
                # Batching: drain available events to reduce overhead
                batch = [event]
                while len(batch) < 15:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                for e in batch:
                    yield e
            except asyncio.TimeoutError:
                # Send heartbeat to keep SSE alive
                yield {
                    "event_type": "heartbeat",
                    "run_id": self.run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                logger.warning(f"[EventBus] Stream error for run {self.run_id}: {e}")
                break

    def close(self) -> None:
        """Signal end of run — streams will drain then stop."""
        # Publish terminal event
        self.publish({
            "event_type": "orchestration_completed",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._closed = True

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self._created_at


class EventBusRegistry:
    """
    Global registry of per-run event buses.
    Auto-evicts buses older than TTL (30 minutes).
    """

    _TTL_SECONDS = 1800  # 30 minutes
    _MAX_BUSES = 500

    def __init__(self):
        self._buses: Dict[str, RunEventBus] = {}
        self._lock = asyncio.Lock()

    def create(self, run_id: str) -> RunEventBus:
        """Create and register a new event bus for a run."""
        bus = RunEventBus(run_id)
        self._buses[run_id] = bus
        self._evict_stale()
        return bus

    def get(self, run_id: str) -> Optional[RunEventBus]:
        return self._buses.get(run_id)

    def get_or_create(self, run_id: str) -> RunEventBus:
        if run_id not in self._buses:
            return self.create(run_id)
        return self._buses[run_id]

    def _evict_stale(self) -> None:
        """Remove closed or expired buses."""
        now_age_limit = self._TTL_SECONDS
        stale = [
            rid for rid, bus in self._buses.items()
            if bus._closed or bus.age_seconds > now_age_limit
        ]
        for rid in stale:
            self._buses.pop(rid, None)
        # Hard cap
        if len(self._buses) > self._MAX_BUSES:
            excess = len(self._buses) - self._MAX_BUSES
            for rid in list(self._buses.keys())[:excess]:
                self._buses.pop(rid, None)

    def close(self, run_id: str) -> None:
        bus = self._buses.get(run_id)
        if bus:
            bus.close()


# ── Module-level singleton ─────────────────────────────────────
_event_bus_registry: Optional[EventBusRegistry] = None


def get_event_bus_registry() -> EventBusRegistry:
    global _event_bus_registry
    if _event_bus_registry is None:
        _event_bus_registry = EventBusRegistry()
    return _event_bus_registry


def create_run_bus(run_id: str) -> RunEventBus:
    """Create a new event bus for a run. Returns the bus."""
    return get_event_bus_registry().create(run_id)


def get_run_bus(run_id: str) -> Optional[RunEventBus]:
    """Retrieve an existing event bus by run ID."""
    return get_event_bus_registry().get(run_id)


# ── Redis Mirror (optional, non-fatal) ─────────────────────────

async def mirror_event_to_redis(
    redis_client,
    run_id: str,
    event_dict: Dict[str, Any],
    ttl: int = 3600,
) -> None:
    """
    Best-effort mirror of events to Redis pub/sub.
    Non-fatal: exceptions are logged and suppressed.
    Used for multi-process observability (e.g. if admin runs in separate process).
    """
    if not redis_client:
        return
    try:
        channel = f"orch:events:{run_id}"
        payload = json.dumps(event_dict, default=str)
        await redis_client.publish(channel, payload)
    except Exception as e:
        logger.debug(f"[EventBus] Redis mirror failed (non-fatal): {e}")
