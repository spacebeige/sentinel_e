"""
============================================================
Orchestration API Routes — Sentinel-E v8.0
============================================================
New endpoints for OrchestrationRun observability.

All existing routes remain unchanged.
These are additive new routes only.

Endpoints:
  GET  /api/orchestration/recent          — recent runs (admin)
  GET  /api/orchestration/active          — currently running
  GET  /api/orchestration/{run_id}        — full run detail
  GET  /api/orchestration/{run_id}/events — SSE live event stream
  GET  /api/orchestration/{run_id}/phases — phase timeline only
  GET  /api/orchestration/{run_id}/summary — compact summary
  GET  /api/memory/layers/{user_id}       — layered memory state (admin)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.orchestration_run import get_run_registry, CognitivePhase, PHASE_LABELS
from core.runtime_event_bus import get_event_bus_registry
from gateway.admin_access import require_runtime_admin, require_runtime_admin_for_stream

logger = logging.getLogger("OrchestrationAPI")

router = APIRouter(prefix="/api/orchestration", tags=["orchestration"])


@router.get("/meta/phases")
async def get_phase_labels(admin: dict = Depends(require_runtime_admin)):
    """Return all cognitive phase labels (for frontend rendering)."""
    return {
        "phases": {p.value: PHASE_LABELS[p] for p in CognitivePhase},
    }


# ── Recent Runs ───────────────────────────────────────────────

@router.get("/recent")
async def get_recent_runs(limit: int = 20, admin: dict = Depends(require_runtime_admin)):
    """
    Return summaries of the most recent orchestration runs.
    Used by admin Mission Control panel.
    """
    try:
        registry = get_run_registry()
        runs = registry.get_recent(limit=min(limit, 50))
        return {
            "success": True,
            "runs": runs,
            "count": len(runs),
        }
    except Exception as e:
        logger.error(f"[OrchestrationAPI] /recent failed: {e}")
        return {"success": False, "runs": [], "count": 0, "error": str(e)}


@router.get("/active")
async def get_active_runs(admin: dict = Depends(require_runtime_admin)):
    """Return currently running (non-terminal) orchestration runs."""
    try:
        registry = get_run_registry()
        active = registry.get_active()
        return {
            "success": True,
            "active_runs": active,
            "count": len(active),
        }
    except Exception as e:
        logger.error(f"[OrchestrationAPI] /active failed: {e}")
        return {"success": False, "active_runs": [], "count": 0, "error": str(e)}


@router.get("/chat/{chat_id}/latest")
async def get_latest_run_for_chat(chat_id: str, admin: dict = Depends(require_runtime_admin)):
    """Return the newest OrchestrationRun attached to a chat/session id."""
    registry = get_run_registry()
    run = registry.get_latest_for_chat(chat_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"No orchestration run found for chat '{chat_id}'")
    return {"success": True, "run": run.to_frontend_dict()}


# ── Run Detail ────────────────────────────────────────────────

@router.get("/{run_id}")
async def get_run_detail(run_id: str, admin: dict = Depends(require_runtime_admin)):
    """Return full detail of a specific orchestration run."""
    registry = get_run_registry()
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Orchestration run '{run_id}' not found")
    try:
        return {
            "success": True,
            "run": run.to_frontend_dict(),
        }
    except Exception as e:
        logger.error(f"[OrchestrationAPI] /{run_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/summary")
async def get_run_summary(run_id: str, admin: dict = Depends(require_runtime_admin)):
    """Return compact summary of a specific orchestration run."""
    registry = get_run_registry()
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Orchestration run '{run_id}' not found")
    return {"success": True, "summary": run.to_summary()}


@router.get("/{run_id}/phases")
async def get_run_phases(run_id: str, admin: dict = Depends(require_runtime_admin)):
    """Return the phase timeline for a specific run."""
    registry = get_run_registry()
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Orchestration run '{run_id}' not found")

    # Filter events to phase transitions only
    phase_events = [
        e.to_dict() for e in run.event_timeline
        if e.event_type == "phase_transition"
    ]

    return {
        "success": True,
        "run_id": run_id,
        "current_phase": run.cognitive_phase.value,
        "current_phase_label": run.phase_label,
        "lifecycle_state": run.lifecycle_state.value,
        "phase_timeline": phase_events,
        "available_phases": {p.value: PHASE_LABELS[p] for p in CognitivePhase},
    }


# ── SSE Live Event Stream ──────────────────────────────────────

@router.get("/{run_id}/events")
async def stream_run_events(
    run_id: str,
    timeout: int = 120,
    admin: dict = Depends(require_runtime_admin_for_stream),
):
    """
    Server-Sent Events stream of live cognitive events for a run.
    
    Frontend connects here to receive real-time phase updates.
    Falls back to polling recent events if run not found in bus.
    
    SSE format:
        data: {"event_type": "...", "phase": "...", ...}\n\n
    """
    timeout = min(timeout, 300)  # max 5 minutes

    async def _event_generator() -> AsyncIterator[str]:
        bus_registry = get_event_bus_registry()
        bus = bus_registry.get(run_id)

        if not bus:
            # Run already completed or not in bus — yield static events from registry
            registry = get_run_registry()
            run = registry.get(run_id)
            if run:
                for event in run.event_timeline[-20:]:
                    yield f"data: {json.dumps(event.to_dict(), default=str)}\n\n"
                yield f"data: {json.dumps({'event_type': 'stream_end', 'run_id': run_id})}\n\n"
            else:
                yield f"data: {json.dumps({'event_type': 'not_found', 'run_id': run_id})}\n\n"
            return

        # Stream live events from bus
        try:
            async for event in bus.stream(timeout=float(timeout)):
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("event_type") in ("orchestration_completed", "orchestration_failed"):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'event_type': 'stream_error', 'error': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'event_type': 'stream_end', 'run_id': run_id})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
