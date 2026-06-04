# Sentinel-E EVO — Service Map
*Authoritative extraction from main (commit 248ef3c)*

## Service Layer (`frontend/src/services/`)

### `api.js` (The Global HTTP Client)
This is the ONLY module authorized to make HTTP requests to the backend. It uses Axios and attaches Supabase JWT tokens (`Authorization: Bearer <token>`) via request interceptors.

**Mappings:**
*   `sendStandard()` -> `POST /run/standard`
*   `sendExperimental()` -> `POST /run/experimental`
*   `sendKill()` -> `POST /run/omega/kill`
*   `sendFeedback()` -> `POST /feedback`
*   `createSession()` -> `POST /api/session`
*   `getHistory()` -> `GET /api/history`
*   `createChat()` -> `POST /api/chat`
*   `getChatMessages()` -> `GET /api/chat/{chat_id}/messages`
*   `getSessionDescriptive()` -> `GET /api/session/{chat_id}/descriptive`
*   `getOmegaSession()` -> `GET /api/omega/session/{chat_id}`
*   `runCrossAnalysis()` -> `POST /api/cross-analysis`
*   `sendMCOQuery()` -> `POST /api/mco/run` (Meta-Cognitive Orchestrator entry point)
*   `fetchMCOModels()` -> `GET /api/mco/models`
*   `fetchChatModels()` -> `GET /chat/models/available`
*   `toggleClaude()` -> `POST /api/models/claude/toggle`
*   `getClaudeUsage()` -> `GET /api/models/claude/usage`
*   `sendDirectModelQuery()` -> `POST /chat/{modelId}`
*   `sendDebateQuery()` -> `POST /battle/debate`
*   `fetchMCOAnalytics()` -> `GET /api/mco/analytics/{sessionId}`
*   `editMessage()` -> `PUT /api/messages/{messageId}`
*   `regenerateMessage()` -> `POST /api/messages/{messageId}/regenerate`

### `sessionManager.js` & `sessionPersistence.js`
*   Orchestrates local `localStorage` state with remote backend state.
*   Calls `createSession()` from `api.js` on mount to acquire device session ID.
*   Calls `getHistory()` to populate the sidebar.

### `supabaseSessionManager.js`
*   Purely handles Supabase OAuth token lifecycle.
*   Caches token snapshots so `api.js` interceptors can attach them synchronously without blocking renders.

### `themeManager.js`
*   Local service for toggling dark/light/system themes in `localStorage` and `document.documentElement.classList`. Does not call the backend API directly.
