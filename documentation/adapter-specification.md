# Sentinel-E EVO — Adapter Specifications

Because we are explicitly *not* modifying the backend, several newly styled Figma UI components require lightweight frontend adapters (data mappers or wrapper components) to successfully consume existing state and API contracts.

## 1. Chat Engine Adapter
**Target Component:** `ChatPage.tsx`
**Current Driver:** `ChatEngineV5.js`

**Specification:**
We must build a wrapper or hook within `ChatPage.tsx` to handle the transition from simple local state (used in Figma mocks) to the robust `useStore` architecture.
*   **Prop Transformation:** Replace local `messages` arrays in `ChatPage.tsx` with `useStore(state => state.messages)`.
*   **Action Transformation:** When the composer submits text, intercept it and route to `sendMCOQuery()` (imported from `api.js`) instead of a local mock function.
*   **State Transformation:** Listen to the response from `sendMCOQuery()`, handle the success payload, and dispatch updates to both `useStore` (for the primary chat history) and `cognitiveStore` (for any visualizer metadata).

## 2. Debate Metadata Adapter
**Target Component:** `CinematicDebatePanel.tsx`
**Source Data:** `machine_metadata.rounds` (from `POST /api/mco/run`)

**Specification:**
The backend `StructuredDebateEngine` outputs raw JSON representing agent rounds.
*   **State Transformation:** Read `cognitiveStore.debate_rounds`.
*   **Prop Transformation:** `CinematicDebatePanel.tsx` likely expects an array of distinct "Agent" objects with `name`, `stance`, `avatar_url`, and `argument_text`. The adapter must safely iterate over the backend's `machine_metadata.rounds` and map `agent_id` -> `name`, `content` -> `argument_text`, and compute layout alignments (left/right positioning based on pro/con stances).

## 3. Evidence Metadata Adapter
**Target Component:** `CinematicEvidencePanel.tsx`
**Source Data:** `reasoning_json` (from `EvidenceEngine`)

**Specification:**
*   **State Transformation:** Read `cognitiveStore.evidence_chain`.
*   **Prop Transformation:** Extract source citations, confidence scores, and synthesis logic from the raw `reasoning_json` blob. Format this into a chronological or source-based array of `{ title, url, snippet, confidence }` objects that the Cinematic component can safely map over without throwing `undefined` errors.

## 4. Glass / Insight Adapter
**Target Component:** `OmegaInsightPanel.tsx`
**Source Data:** `machine_metadata` (from Omega / Deep Probe modes)

**Specification:**
*   **Prop Transformation:** Extract system trust metrics, internal boundary warnings, and safety checks from the backend envelope. Convert numeric scores (e.g., `0.87` confidence) into visual prop percentages expected by the Glass UI progress bars or radar charts.

## 5. Admin Dashboard Adapter
**Target Component:** `AdminPage.tsx`
**Source Data:** `/api/admin/*` endpoints

**Specification:**
*   **State Transformation:** Wrap `AdminPage.tsx` in a generic data-fetching hook that calls `api.get('/api/admin/system/stats')` via `api.js`.
*   **Prop Transformation:** The backend returns structured dictionaries (e.g., `{ total_users: 150, active_chats: 45 }`). The Recharts components in `AdminPage.tsx` require specific array formats (e.g., `[{ name: 'Users', value: 150 }]`). The adapter must map the raw API JSON to these array structures before passing them to the UI.
