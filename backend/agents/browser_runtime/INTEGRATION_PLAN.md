# Embedded Browser Runtime Integration Plan

## 1. Full Integration Plan

The dumped `backend/browser_use/` folders remain source material only. Sentinel-E now owns the browser execution path through `backend/agents/browser_runtime/`, which provides compact observation, governance-aware actions, Groq planning, Groq reflection, tab/session handling, and compact JSON memory.

The browser runtime is an internal subsystem. It does not replace the MCO, ensemble engine, cognitive runtime, persistence, auth, or governance layers.

## 2. Workspace Integration Points

- `backend/api/browser_runtime_routes.py` exposes `/api/browser-runtime/*` endpoints through the existing FastAPI app.
- `backend/main.py` includes the browser runtime router alongside existing API routers.
- `core.orchestration_run` is used opportunistically so browser workflows can attach runtime events and lifecycle state.
- `gateway.auth_v2.get_current_user` protects all browser runtime endpoints.
- `gateway.prompt_firewall` screens autonomous workflow tasks before planning.

## 3. Refactoring Strategy

Keep `backend/browser_use/` out of the live import path. Extract only the useful ideas into small modules:

- browser session handling -> `tab_manager.py`
- DOM state extraction -> `browser_observer.py`
- action execution -> `browser_actions.py`
- planning loop -> `planner.py`
- reflection/retry -> `reflection.py`
- permission model -> `permissions.py`
- compact memory -> `memory_manager.py`

## 4. Browser Runtime Cleanup Plan

Disable Browser-Use framework layers by non-use first, then remove later when import checks confirm there are no dependencies:

- cloud modules
- watchdog framework
- recordings/video/gif helpers
- framework memory/message manager
- benchmark/demo code
- Browser-Use system prompts
- Browser-Use agent service lifecycle

## 5. Lightweight Architecture Mapping

`observe -> plan -> permission -> act -> reflect -> retry/replan`

The loop is capped by request-level `max_steps` and hard-limited to 8 steps. Browser state is always compact JSON, never raw HTML.

## 6. Browser Runtime Interfaces

- `POST /api/browser-runtime/observe`
- `POST /api/browser-runtime/act`
- `POST /api/browser-runtime/run`
- `GET /api/browser-runtime/sessions`
- `DELETE /api/browser-runtime/sessions/{session_id}`
- `GET /api/browser-runtime/memory`

## 7. Orchestration Hooks

Each observe/action/workflow request can create an `OrchestrationRun` with `execution_path=browser_runtime`. The runtime emits real events such as:

- `browser_state_observed`
- `browser_action_requested`
- `browser_action_planned`
- `browser_permission_required`

## 8. Dependency Cleanup

Required:

- `playwright`
- `PyMuPDF` for PDF extraction, already present

Optional:

- `easyocr`, loaded only when OCR fallback is needed

Runtime install note:

```bash
python -m playwright install chromium
```

## 9. Groq Integration Layer

`groq_router.py` calls Groq's OpenAI-compatible endpoint directly. It uses `GROQ_API_KEY` or `GROQ_LLAMA_INSTANT_KEY`, with `BROWSER_RUNTIME_GROQ_MODEL` defaulting to `llama-3.1-8b-instant`.

No Browser-Use LLM abstractions are used.

## 10. Reflection/Retry Integration

`reflection.py` asks Groq for a compact JSON reflection after each action. Failed selectors and retry strategies are persisted through `memory_manager.py`.

## 11. Lightweight Autonomous Execution Flow

The workflow endpoint:

1. Authenticates user.
2. Runs prompt firewall.
3. Opens/reuses browser session.
4. Observes compact browser state.
5. Plans one action with Groq.
6. Checks permission policy.
7. Executes safe actions.
8. Reflects with Groq.
9. Retries only when reflection says retry and the step cap allows it.

## 12. OCR/PDF Integration Strategy

DOM extraction is primary. OCR is triggered only when the DOM state is empty and `allow_ocr=true`. PDF text extraction uses PyMuPDF through `BrowserObserver.extract_pdf_bytes()` and `extract_pdf_from_page()`.

No screenshot multimodal API is used.

## 13. Governance Integration

`permissions.py` requires confirmation before:

- sending/submitting content
- purchases/payments
- deleting/removing content
- upload/submit actions
- form submission and enter-to-submit behavior

Safe read/navigation actions can run automatically.

## 14. Runtime Optimization Strategy

- Lazy import Playwright, EasyOCR, PIL, NumPy, and PyMuPDF.
- Keep browser sessions TTL-bound.
- Cap DOM text and element counts.
- Cap autonomous loop depth.
- Store compact JSON memory only.
- Avoid vector databases and framework event layers.

## 15. Runnable Integration Code

Runtime code is implemented in:

- `browser_observer.py`
- `browser_actions.py`
- `planner.py`
- `reflection.py`
- `groq_router.py`
- `tab_manager.py`
- `permissions.py`
- `memory_manager.py`

API integration is implemented in:

- `backend/api/browser_runtime_routes.py`
- `backend/main.py`
