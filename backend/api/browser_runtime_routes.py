from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.browser_runtime.browser_actions import BrowserActionExecutor
from agents.browser_runtime.browser_observer import BrowserObserver
from agents.browser_runtime.memory_manager import get_browser_memory_manager
from agents.browser_runtime.permissions import BrowserPermissionPolicy
from agents.browser_runtime.planner import BrowserPlanner
from agents.browser_runtime.reflection import BrowserReflection
from agents.browser_runtime.tab_manager import get_tab_manager
from gateway.auth_v2 import get_current_user
from gateway.prompt_firewall import get_firewall
from utils.api_response import api_error, api_success

try:
    from core.orchestration_run import CognitivePhase, create_orchestration_run
except ImportError:
    CognitivePhase = None
    create_orchestration_run = None


logger = logging.getLogger("BrowserRuntimeAPI")
router = APIRouter(prefix="/api/browser-runtime", tags=["Browser Runtime"])


class ObserveRequest(BaseModel):
    session_id: Optional[str] = None
    url: Optional[str] = None
    cdp_url: Optional[str] = None
    allow_ocr: bool = True


class ActionRequest(BaseModel):
    session_id: Optional[str] = None
    cdp_url: Optional[str] = None
    action: Dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False


class WorkflowRequest(BaseModel):
    task: str
    session_id: Optional[str] = None
    start_url: Optional[str] = None
    cdp_url: Optional[str] = None
    max_steps: int = 4
    confirmed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    allow_ocr: bool = True


def _runtime_components():
    observer = BrowserObserver()
    memory = get_browser_memory_manager()
    permissions = BrowserPermissionPolicy()
    actions = BrowserActionExecutor(observer=observer, permissions=permissions, memory=memory)
    return observer, memory, actions


def _start_run(user_id: str, task: str, chat_id: str = ""):
    if create_orchestration_run is None:
        return None
    try:
        run = create_orchestration_run(
            chat_id=chat_id,
            user_id=user_id,
            query_preview=task[:80],
            execution_path="browser_runtime",
        )
        if CognitivePhase is not None:
            run.transition_to(CognitivePhase.OBSERVE, {"subsystem": "browser_runtime"})
        return run
    except Exception as exc:
        logger.debug("Browser runtime OrchestrationRun creation failed: %s", exc)
        return None


def _event(run, event_type: str, payload: Dict[str, Any], phase=None) -> None:
    if run is None:
        return
    try:
        if phase is not None:
            run.transition_to(phase, payload)
        else:
            run.emit_event(event_type, payload)
    except Exception:
        pass


@router.post("/observe")
async def observe_browser(
    payload: ObserveRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    observer, _, _ = _runtime_components()
    manager = get_tab_manager()
    user_id = user.get("user_id") or user.get("id") or "unknown"
    run = _start_run(user_id, payload.url or "observe", payload.session_id or "")

    try:
        session = await manager.get_or_create(payload.session_id, cdp_url=payload.cdp_url)
        if payload.url:
            await session.page.goto(payload.url, wait_until="domcontentloaded", timeout=30000)
        state = (await observer.observe(session.page, allow_ocr=payload.allow_ocr)).to_dict()
        _event(run, "browser_state_observed", {"url": state.get("url"), "session_id": session.session_id})
        if run:
            run.mark_completed()
        return api_success({
            "session_id": session.session_id,
            "state": state,
            "orchestration_run": run.to_frontend_dict() if run else None,
        })
    except Exception as exc:
        if run:
            run.mark_failed(str(exc), "BROWSER_OBSERVE_FAILED")
        return api_error(f"Browser observe failed: {exc}", status_code=500)


@router.post("/act")
async def execute_browser_action(
    payload: ActionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    observer, _, actions = _runtime_components()
    manager = get_tab_manager()
    user_id = user.get("user_id") or user.get("id") or "unknown"
    run = _start_run(user_id, str(payload.action), payload.session_id or "")

    try:
        action = dict(payload.action or {})
        if payload.confirmed:
            action["confirmed"] = True
        session = await manager.get_or_create(payload.session_id, cdp_url=payload.cdp_url)
        state = (await observer.observe(session.page)).to_dict()
        _event(run, "browser_action_requested", {"action": action}, CognitivePhase.ROUTE if CognitivePhase else None)
        result = await actions.execute(session.page, action, page_state=state)
        if run:
            if result.permission and result.permission.get("requires_confirmation"):
                run.emit_event("browser_permission_required", result.permission)
            if result.ok and CognitivePhase is not None:
                run.transition_to(CognitivePhase.REFLECT, {"action_type": action.get("type")})
            run.mark_completed() if result.ok else run.mark_failed(result.error, "BROWSER_ACTION_FAILED")
        return api_success({
            "session_id": session.session_id,
            "result": result.to_dict(),
            "orchestration_run": run.to_frontend_dict() if run else None,
        })
    except Exception as exc:
        if run:
            run.mark_failed(str(exc), "BROWSER_ACTION_EXCEPTION")
        return api_error(f"Browser action failed: {exc}", status_code=500)


@router.post("/run")
async def run_browser_workflow(
    payload: WorkflowRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    firewall = get_firewall()
    verdict = firewall.analyze(payload.task)
    if verdict.blocked:
        return api_error("Browser task blocked by governance firewall.", status_code=400)

    observer, memory, actions = _runtime_components()
    manager = get_tab_manager()
    planner = BrowserPlanner()
    reflection = BrowserReflection()
    user_id = user.get("user_id") or user.get("id") or "unknown"
    run = _start_run(user_id, verdict.sanitized_text or payload.task, payload.session_id or "")

    try:
        session = await manager.get_or_create(payload.session_id, cdp_url=payload.cdp_url)
        if payload.start_url:
            await session.page.goto(payload.start_url, wait_until="domcontentloaded", timeout=30000)

        steps: List[Dict[str, Any]] = []
        previous_error = ""
        max_steps = max(1, min(payload.max_steps, 8))
        task = verdict.sanitized_text or payload.task

        for _ in range(max_steps):
            state = (await observer.observe(session.page, allow_ocr=payload.allow_ocr)).to_dict()
            _event(run, "browser_state_observed", {"url": state.get("url")})
            if CognitivePhase is not None:
                _event(run, "browser_planning_started", {"step": len(steps) + 1}, CognitivePhase.ANALYZE)

            plan = await planner.plan(
                task=task,
                state=state,
                memory_hints=memory.selector_hints(state.get("url", "")),
                previous_error=previous_error,
            )
            action = dict(plan.get("action") or {"type": "extract"})
            if _action_is_confirmed(action, payload.confirmed_actions):
                action["confirmed"] = True

            if plan.get("done"):
                if run:
                    run.mark_completed()
                memory.record_workflow(task, steps, success=True)
                return api_success({
                    "session_id": session.session_id,
                    "status": "completed",
                    "state": state,
                    "steps": steps,
                    "orchestration_run": run.to_frontend_dict() if run else None,
                })

            if CognitivePhase is not None:
                _event(run, "browser_action_planned", {"action": action, "rationale": plan.get("rationale", "")}, CognitivePhase.ROUTE)
            action_result = await actions.execute(session.page, action, page_state=state)
            reflected = await reflection.reflect(task=task, action=action, result=action_result.to_dict())
            step = {
                "plan": plan,
                "action_result": action_result.to_dict(),
                "reflection": reflected,
            }
            steps.append(step)

            if action_result.permission and action_result.permission.get("requires_confirmation"):
                if run:
                    run.emit_event("browser_permission_required", action_result.permission)
                    run.mark_completed()
                return api_success({
                    "session_id": session.session_id,
                    "status": "permission_required",
                    "permission": action_result.permission,
                    "state": state,
                    "steps": steps,
                    "orchestration_run": run.to_frontend_dict() if run else None,
                })

            if reflected.get("done") or (action.get("type") == "extract" and action_result.ok):
                final_state = action_result.state or state
                if run:
                    run.mark_completed()
                memory.record_workflow(task, [s["plan"].get("action", {}) for s in steps], success=True)
                return api_success({
                    "session_id": session.session_id,
                    "status": "completed",
                    "state": final_state,
                    "steps": steps,
                    "orchestration_run": run.to_frontend_dict() if run else None,
                })

            if not action_result.ok:
                previous_error = action_result.error
                memory.record_retry_strategy(task, action, reflected.get("retry_strategy", ""))
                if not reflected.get("retry"):
                    break

        final_state = (await observer.observe(session.page, allow_ocr=payload.allow_ocr)).to_dict()
        if run:
            run.mark_completed()
        return api_success({
            "session_id": session.session_id,
            "status": "max_steps_reached",
            "state": final_state,
            "steps": steps,
            "orchestration_run": run.to_frontend_dict() if run else None,
        })
    except Exception as exc:
        if run:
            run.mark_failed(str(exc), "BROWSER_WORKFLOW_FAILED")
        return api_error(f"Browser workflow failed: {exc}", status_code=500)


@router.get("/sessions")
async def list_browser_sessions(user: Dict[str, Any] = Depends(get_current_user)):
    return api_success({"sessions": get_tab_manager().list_sessions()})


@router.delete("/sessions/{session_id}")
async def close_browser_session(session_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    closed = await get_tab_manager().close(session_id)
    return api_success({"session_id": session_id, "closed": closed})


@router.get("/memory")
async def get_browser_runtime_memory(user: Dict[str, Any] = Depends(get_current_user)):
    return api_success(get_browser_memory_manager().snapshot())


def _action_is_confirmed(action: Dict[str, Any], confirmed_actions: List[Dict[str, Any]]) -> bool:
    action_type = action.get("type")
    selector = action.get("selector")
    url = action.get("url")
    for confirmed in confirmed_actions or []:
        if confirmed.get("type") != action_type:
            continue
        if selector and confirmed.get("selector") == selector:
            return True
        if url and confirmed.get("url") == url:
            return True
    return False
