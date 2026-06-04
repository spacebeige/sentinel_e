# Sentinel-E EVO — Master Architecture Context

## Section 1 — Project Overview

Project Name:
Sentinel-E EVO

Working Branch:
integrationv0

Backend Source of Truth:
latest main

Frontend Restoration Source:
main @ 248ef3c

UI Source:
integration-ui-backend

This project is NOT a branch merge.
This project is a UI transplantation effort.
Backend remains authoritative.
Frontend backend wiring remains authoritative.
UI components are transplanted onto existing architecture.

## Section 2 — Current Frontend Architecture

### Directory Structure
```text
frontend/src/
├── components/
├── pages/
├── hooks/
├── stores/
├── services/
├── layouts/
├── figma_features/
├── figma_shell/
├── utils/
└── styles/
```

### Major Files
*   **Components & Shells:** `ChatEngineV5.js`, `ChatThread.js`, `InputArea.js`, `Sidebar.js`, `SessionSidebar.js`, `AdminDashboard.js`, `FigmaChatShell.js`
*   **Services:** `api.js`, `sessionManager.js`, `sessionPersistence.js`, `supabaseSessionManager.js`
*   **State Management:** `useStore.js`, `cognitiveStore.js`
*   **Hooks:** `useSupabaseAuth.js`, `useAdminRole.js`, `useModels.js`

## Section 3 — Route Map

*   `/` -> `LandingPage.js` (Landing / marketing)
*   `/chat` -> `ChatPage.js` (Uses `ChatEngineV5.js` and `FigmaChatShell.js`)
*   `/admin` -> `AdminDashboard.js`
*   `/models` -> `ModelsPage.js`
*   `/pricing` -> `PricingPage.js`

## Section 4 — Frontend → Backend API Mapping

### `api.js` (Authoritative HTTP Layer)

Mappings:
*   `sendStandard()` → `POST /run/standard`
*   `sendExperimental()` → `POST /run/experimental`
*   `sendKill()` → `POST /run/omega/kill`
*   `sendFeedback()` → `POST /feedback`
*   `createSession()` → `POST /api/session`
*   `getHistory()` → `GET /api/history`
*   `createChat()` → `POST /api/chat`
*   `getChatMessages()` → `GET /api/chat/{chat_id}/messages`
*   `sendMCOQuery()` → `POST /api/mco/run`
*   `editMessage()` → `PUT /api/messages/{id}`
*   `regenerateMessage()` → `POST /api/messages/{id}/regenerate`

> **CRITICAL RULE:** `api.js` is authoritative. Do not replace `api.js`. New UI components must consume existing `api.js` functions.

## Section 5 — State Architecture

### `useStore.js`
*   **State:** `session_id`, `chat_id`, `messages`, `chats`, `mode`
*   **Consumers:** `ChatPage`, `ChatEngineV5`, `Sidebar`

### `cognitiveStore.js`
*   **State:** `active_models`, `debate_rounds`, `evidence_chain`
*   **Consumers:** `DebateArena`, `GlassConsole`, `CognitionStreamPanel`

> **CRITICAL RULE:** New Figma visualizers must consume these stores. Do not replace these stores.

## Section 6 — Authentication Architecture

### Flow
```text
Supabase
↓
useSupabaseAuth
↓
supabaseSessionManager
↓
api.js interceptor
↓
Authorization Bearer Token
↓
backend auth_v2
↓
ensure_user_exists()
```

### Critical Files
*   `useSupabaseAuth.js`
*   `supabaseSessionManager.js`
*   `api.js`

> **CRITICAL RULE:** Authentication logic must remain intact. Only UI may change.

## Section 7 — Chat Architecture

### Flow
```text
ChatPage
↓
FigmaChatShell
↓
ChatEngineV5
↓
api.js
↓
POST /api/mco/run
```

### Additional Routes
*   `POST /api/session`
*   `GET /api/history`
*   `POST /api/chat`
*   `GET /api/chat/{id}/messages`
*   `PUT /api/messages/{id}`
*   `POST /api/messages/{id}/regenerate`

## Section 8 — Orchestrator Architecture

### Flow
```text
POST /api/mco/run
↓
AgenticOrchestrator
↓
StructuredDebateEngine | EvidenceEngine | OmegaKernel
↓
CognitiveGateway
```

### Expected Response
```json
{
  "success": true,
  "data": {
    "response": "...",
    "machine_metadata": {}
  }
}
```

## Section 9 — Debate / Evidence / Glass Mapping

### Debate
*   **Current:** `DebateArena.js`
*   **Future:** `CinematicDebatePanel.tsx`
*   **Map:** `machine_metadata.rounds`

### Evidence
*   **Current:** `EvidenceConsole.js`
*   **Future:** `CinematicEvidencePanel.tsx`
*   **Map:** `reasoning_json`, `evidence metadata`

### Glass
*   **Current:** `GlassConsole.js`
*   **Future:** Glass UI
*   **Map:** `machine_metadata`, trust metrics, transparency payload

## Section 10 — Admin Architecture

*   **Component:** `AdminDashboard.js`
*   **Endpoints:**
    *   `/api/admin/system/stats`
    *   `/api/admin/system/architecture`
    *   `/api/admin/web-analytics`
    *   `/api/admin/feedback-summary`
    *   `/api/admin/models/performance`

> **CRITICAL RULE:** Admin APIs remain authoritative. UI only changes presentation.

## Section 11 — Figma UI Inventory

Source: `integration-ui-backend` (Specifically `figma_ui/`)

### Inventory
*   **Pages:** `HomePage.tsx`, `LoginPage.tsx`, `SignupPage.tsx`, `ChatPage.tsx`, `ProfilePage.tsx`, `SettingsPage.tsx`, `AdminPage.tsx`, `ExplorePage.tsx`
*   **Components:** `AdminModal.tsx`, `CinematicDebatePanel.tsx`, `CinematicEvidencePanel.tsx`, `OmegaInsightPanel.tsx`, `SessionAnalyticsPanel.tsx`, `CrossAnalysisPanel.tsx`, `FeaturesSection.tsx`, `HeroSection.tsx`
*   **Layouts:** `Layout.tsx`, `Navbar.tsx`, `Footer.tsx`
*   **Design System:** `tailwind.css`, `theme.css`, `fonts.css`, Tailwind Configuration
*   **Gamified/Visual Assets:** `Tree.tsx`, `WaterTile.tsx`, `House.tsx`, `Fence.tsx`, `Lamp.tsx`, `Rock.tsx`

Reusable components will be surgically transplanted to replace legacy React `.js` files.

## Section 12 — Migration Rules

**Never Replace:**
*   `api.js`
*   `sessionManager.js`
*   `sessionPersistence.js`
*   `supabaseSessionManager.js`
*   `useStore.js`
*   `cognitiveStore.js`
*   `useSupabaseAuth.js`
*   `backend/`
*   `database/`
*   `memory/`
*   `gateway/`
*   `api/`
*   `orchestration/`

**Safe To Replace:**
*   `pages/`
*   `components/`
*   `layouts/`
*   `styles/`
*   Presentation layer

## Section 13 — Final Goal

```text
Latest Main Backend
+
Frontend Wiring From Main@248ef3c
+
integration-ui-backend Design System
+
integration-ui-backend Components
+
integration-ui-backend Layouts
=
Sentinel-E EVO Production Candidate
```
