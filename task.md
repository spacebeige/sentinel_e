# Sentinel-E EVO — Figma UI Migration Task List

## Setup
- `[x]` Configure Vite aliases (`@services`, `@stores`, `@hooks`, `@utils`) in `figma_ui/vite.config.ts`.
- `[x]` Configure TypeScript path mappings in `figma_ui/tsconfig.json`.
- `[x]` Ensure Vite handles `.js` files containing JSX from `frontend/src/` and shims `process.env` if necessary.

## Phase 1: Landing & Shell
- `[x]` Migrate routing in `figma_ui/src/app/routes.tsx` to match legacy routes.
- `[x]` Update Landing Page & Navbar to use `useSupabaseAuth` for context.

## Phase 2: Chat & Messaging Loop
- `[x]` Strip local state/mocks from `ChatPage.tsx`, `Sidebar`, and `Composer`.
- `[x]` Integrate `useStore` (`messages`, `chat_id`, `mode`, etc.) into Chat UI.
- `[x]` Integrate `sendMCOQuery` and `getHistory` into `ChatPage` via `api.js`.

## Phase 3: Cognitive Visualizers
- `[x]` Adapt `CinematicDebatePanel.tsx` to consume `cognitiveStore.debate_rounds`.
- `[x]` Adapt `CinematicEvidencePanel.tsx` to consume `cognitiveStore.evidence_chain`.
- `[x]` Adapt `OmegaInsightPanel.tsx` to consume `machine_metadata`.

## Phase 4: Utilities & Dashboards
- `[/]` Migrate `LoginPage.tsx` (Remove guest, bind to `useSupabaseAuth`)
- `[ ]` Migrate `SignupPage.tsx` (Bind to `useSupabaseAuth`)
- `[ ]` Migrate `ProfilePage.tsx` (Bind to `api.get('/api/user')`, disable avatar upload)
- `[ ]` Migrate `SettingsPage.tsx` (Bind to `/api/user/settings`, remove unsupported fields)
- `[ ]` Migrate Navbar/Sidebar (Remove guest mode, enforce `useAdminRole`)
- `[ ]` Validate Chat loop and Visualizers.

## Phase 5: Verification
- `[ ]` Build `figma_ui` successfully.
- `[ ]` Validate Chat loop and Visualizers.
