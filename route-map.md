# Sentinel-E EVO — Route Map
*Authoritative extraction from main (commit 248ef3c)*

## Frontend Routes -> UI Components

| URL Route | Top-level Component | Primary Purpose |
| :--- | :--- | :--- |
| `/` | `LandingPage.js` -> `HomePage.js` | Splash, marketing, entry |
| `/chat` | `ChatPage.js` -> `FigmaChatShell.js` | Core chat and interaction loop |
| `/admin` | `AdminDashboard.js` | Admin telemetry & system monitoring |
| `/models` | `ModelsPageWrapper.js` -> `ModelsPage.js` | AI Model registry catalog |
| `/pricing` | `PricingPageWrapper.js` -> `PricingPage.js` | Pro subscriptions |

## Frontend Pages -> Backend APIs

### `/chat` (ChatPage.js, ChatEngineV5.js)
*   **Session Start**: `POST /api/session` (Initialize device session)
*   **Load History**: `GET /api/history` (Sidebar session history)
*   **Create Chat**: `POST /api/chat`
*   **Load Messages**: `GET /api/chat/{chat_id}/messages`
*   **Message Generation**: `POST /api/mco/run` (or `/run/experimental`, `/battle/debate`, `/run/standard` based on mode)
*   **Message Edits**: `PUT /api/messages/{message_id}`
*   **Message Regenerate**: `POST /api/messages/{message_id}/regenerate`
*   **Telemetry**: `GET /api/mco/analytics/{session_id}`

### `/admin` (AdminDashboard.js)
*   `GET /api/admin/system/stats`
*   `GET /api/admin/system/architecture`
*   `GET /api/admin/web-analytics`
*   `GET /api/admin/feedback-summary`
*   `GET /api/admin/orchestrator/performance`
*   `GET /api/admin/memory/learning`
*   `GET /api/admin/models/performance`
*   `GET /api/orchestration/recent?limit=20`
*   `GET /api/orchestration/active`
*   `GET /api/orchestration/{run_id}`

### `/models` (ModelsPage.js)
*   `GET /api/mco/models` (Fetch cognitive engine models)
*   `GET /chat/models/available` (Fetch fallback / legacy standard models)
*   `GET /api/models/claude/usage` (Track specific model usage)

### Authentication (Supabase)
*   Frontend interacts directly with Supabase via `@supabase/supabase-js`.
*   Backend intercepts auth via `Authorization: Bearer <token>` in `api.js` interceptors.
