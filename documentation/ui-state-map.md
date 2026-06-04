# Sentinel-E EVO — UI State Map

This document traces how the new Figma UI components will map to the existing, authoritative Zustand stores, services, and backend APIs.

## Chat Interface

### `ChatPage.tsx`
*   **Store (`useStore`):** Binds to `session_id`, `chat_id`, `messages`, `mode`.
*   **Service (`api.js`):** Binds to `createSession()`, `getHistory()`, `sendMCOQuery()`.
*   **API (Backend):** `POST /api/mco/run`, `POST /api/session`, `GET /api/history`.

### `Sidebar` / History Panel
*   **Store (`useStore`):** Binds to `chats` (array of historical sessions).
*   **Service (`api.js`):** `getHistory()`.
*   **API (Backend):** `GET /api/history`.

## Orchestration Visualizers

### `CinematicDebatePanel.tsx`
*   **Store (`cognitiveStore`):** Binds to `debate_rounds`.
*   **Service (`api.js`):** Receives data indirectly via `sendMCOQuery()` response resolution.
*   **API (Backend):** Consumes `machine_metadata` emitted by `POST /api/mco/run`.

### `CinematicEvidencePanel.tsx`
*   **Store (`cognitiveStore`):** Binds to `evidence_chain`.
*   **Service (`api.js`):** Receives data indirectly via `sendMCOQuery()` response resolution.
*   **API (Backend):** Consumes `reasoning_json` emitted by `POST /api/mco/run`.

### `OmegaInsightPanel.tsx`
*   **Store (`cognitiveStore`):** Global cognitive state / trust metrics.
*   **Service (`api.js`):** Indirect consumption.
*   **API (Backend):** Consumes complex transparency payloads from Omega execution.

## Management & Administration

### `AdminPage.tsx`
*   **Store (Auth):** Binds to `useAdminRole` hook.
*   **Service (`api.js`):** Multiple standard GET requests (e.g., `getAdminStats()`).
*   **API (Backend):** `GET /api/admin/*`.

### `SettingsPage.tsx`
*   **Store (Context):** Theme Context, user preferences.
*   **Service (`api.js`):** Settings endpoints.
*   **API (Backend):** `GET /api/user/settings`, `PUT /api/user/settings`.

### `ProfilePage.tsx`
*   **Store (Auth):** `useSupabaseAuth` (provides current user context).
*   **Service (`api.js`):** Fetches profile details.
*   **API (Backend):** `GET /api/user`.
