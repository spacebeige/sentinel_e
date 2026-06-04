# Sentinel-E EVO — Frontend-Backend Mapping Audit
*Authoritative extraction from main (commit 248ef3c)*

## Overview
This document represents the absolute end-to-end trace of frontend components to backend database models as they existed in commit `248ef3c`.

---

## 1. Authentication Mapping

*   **Frontend Components**: `LandingPage.js`, `ProtectedRoute.js`, `SessionSidebar.js`
*   **Auth Service**: `@supabase/supabase-js` (via `useSupabaseAuth.js`)
*   **Session Logic**: Tokens stored in `localStorage`, intercepted by `api.js` (`Authorization: Bearer <token>`).
*   **Backend Sync Endpoint**: All routes implicitly sync using `backend/gateway/auth_v2.py` dependency `ensure_user_exists(supabase_id)`.
*   **Backend Database**: Validates against `users` table via `crud_v2.get_user(supabase_id)`.

---

## 2. Chat System Mapping

*   **Frontend Components**: `ChatEngineV5.js`, `InputArea.js`, `ChatThread.js`, `Sidebar.js`
*   **Session Initialization**: 
    *   **Frontend**: `createSession()` 
    *   **Backend Route**: `POST /api/session` (`backend/api/endpoints_v2.py`)
    *   **Database**: Inserts into `device_sessions`.
*   **Chat History**:
    *   **Frontend**: `getHistory()`
    *   **Backend Route**: `GET /api/history`
    *   **Database**: Queries `chats` table.
*   **Message Fetching**:
    *   **Frontend**: `getChatMessages(chat_id)`
    *   **Backend Route**: `GET /api/chat/{chat_id}/messages`
    *   **Database**: Queries `messages` table filtered by `chat_id`.

---

## 3. Execution Modes Mapping (The Orchestrator)

*   **Frontend Component**: `ChatEngineV5.js` triggers `sendMCOQuery(query, { mode, subMode })` in `api.js`.
*   **Backend Route**: `POST /api/mco/run` (`backend/api/orchestration_routes.py` -> `run_mco_query()`).
*   **Backend Service Trace**:
    *   Route passes payload to `AgenticOrchestrator` (`backend/core/agentic_orchestrator.py`).
    *   If `mode == 'debate'`: Routes to `StructuredDebateEngine` (`backend/core/structured_debate_engine.py`).
    *   If `mode == 'glass'` or `evidence`: Routes to `EvidenceEngine` (`backend/core/evidence_engine.py`) and `CognitiveRAG` (`backend/retrieval/cognitive_rag.py`).
    *   If `mode == 'omega'`: Routes to `OmegaKernel` (`backend/core/omega_kernel.py`).
*   **Execution Models**: Invokes `CognitiveGateway` (`backend/metacognitive/cognitive_gateway.py`) to connect to LLM providers (Anthropic, Qwen, Gemini).
*   **Response Schema**: Frontend expects an envelope `{ success: true, data: { response, machine_metadata: { ... } } }`.
*   **Visualizers**: `DebateArena.js`, `GlassConsole.js`, `EvidenceConsole.js` consume `machine_metadata` to render agent avatars and debate rounds.

---

## 4. Admin Mapping

*   **Frontend Component**: `AdminDashboard.js`
*   **Services**: Calls endpoints directly via `api.js` (e.g. `api.get('/api/admin/system/stats')`).
*   **Backend Routes**: `backend/gateway/admin_routes.py`
    *   `get_system_stats()` -> Queries count of `users`, `chats`, `messages`.
    *   `get_orchestrator_performance()` -> Queries `orchestration_runs` table.
    *   `get_memory_learning_stats()` -> Queries `knowledge_graph` and `semantic_vectors`.
*   **Database Persistence**: Direct `asyncpg` execution against `sentinel_sigma.db` (Postgres schema).

---

## 5. Settings / Registry Mapping

*   **Frontend Component**: `ModelsPage.js`
*   **Services**: `useModels.js` hook -> `fetchMCOModels()`
*   **Backend Route**: `GET /api/mco/models` (`backend/api/orchestration_routes.py`)
*   **Backend Service**: Static dictionary lookup from `backend.models.model_registry`.

---

## Summary of API Flow

```text
User Input -> FigmaChatShell / ChatEngineV5
              |
              v
       services/api.js (Intercepts Auth, Attach X-Request-ID)
              |
              v
[HTTPS /api/mco/run]
              |
              v
 FastAPI (backend/api/orchestration_routes.py)
              |
              v
 Core Engine (AgenticOrchestrator)
              |
              v
 Database (crud_v2.py -> SQLAlchemy -> asyncpg -> Neon Postgres)
```
