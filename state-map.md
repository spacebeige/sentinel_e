# Sentinel-E EVO — State Map
*Authoritative extraction from main (commit 248ef3c)*

## Stores (`frontend/src/stores/`)

### `useStore.js` (Zustand)
Global application state.
*   **State**: `session_id`, `chat_id`, `chats` (history array), `messages` (current chat array), `mode` (current query mode).
*   **Consumers**: `ChatEngineV5.js`, `ChatPage.js`, `Sidebar.js`.
*   **Actions**: `setSessionId`, `setChatId`, `addMessage`, `setMessages`.

### `cognitiveStore.js` (Zustand)
State specifically for orchestrator visualizations (Debate, Evidence, Synthesis).
*   **State**: `active_models`, `debate_rounds`, `evidence_chain`.
*   **Consumers**: `DebateArena.js`, `CognitionStreamPanel.js`, `GlassConsole.js`.

## Contexts & Providers (`frontend/src/`)

### `AuthContext` (provided via `App.js` or `Layout.js`)
*   **Provider**: Wraps the React tree with Supabase Auth session listener.
*   **Consumers**: `ProtectedRoute.js`, `SessionSidebar.js` (to show user profile).

## Hooks (`frontend/src/hooks/`)

### `useSupabaseAuth.js`
*   Initializes the `@supabase/supabase-js` client.
*   Listens for `onAuthStateChange`.
*   Updates local storage token snapshots for `api.js` interceptor usage.

### `useAdminRole.js`
*   Decodes the JWT or fetches the user's role from Supabase to gate access to `/admin`.
*   **Consumers**: `AdminDashboard.js`, `ProtectedRoute.js`.

### `useModels.js`
*   Fetches the model registry from `api.js` (`fetchMCOModels()`).
*   Manages the state of the model selection dropdowns.
*   **Consumers**: `ModelsPage.js`, `InputArea.js`, `FigmaChatShell.js`.

### `useAuthContext.js`
*   React Context wrapper hook for consuming the current user session seamlessly in functional components.

## Dependency Graph Example

```text
ChatEngineV5.js
 ├── useStore (Global UI state)
 │    └── chat_id, messages
 ├── useSupabaseAuth (Auth state)
 │    └── user
 └── sessionManager.js
      └── api.js
           ├── POST /api/mco/run
           └── GET /api/chat/{id}/messages
```
