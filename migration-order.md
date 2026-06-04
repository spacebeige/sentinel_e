# Sentinel-E EVO — Migration Order

The migration of UI components from `figma_ui` to the live `integrationv0` environment must be handled iteratively to isolate bugs and prevent regressions in the core messaging loop. 

Here is the recommended order, ranked by difficulty (Easiest to Hardest).

## 1. Landing (Lowest Difficulty)
*   **Target Components:** `HomePage.tsx`, `HeroSection.tsx`, `Navbar.tsx`, `Footer.tsx`
*   **Reasoning:** The landing pages are largely static and visual. They require minimal state dependency (only simple AuthContext to swap 'Login' for 'Dashboard' buttons). Migrating this first establishes that the Vite build, Tailwind CSS tokens, and static assets from the Figma design system are functioning correctly in the live repository.

## 2. Profile & Settings (Low Difficulty)
*   **Target Components:** `ProfilePage.tsx`, `SettingsPage.tsx`
*   **Reasoning:** These are standalone pages with simple, distinct CRUD operations. The settings page only needs to map UI toggles to existing `api.js` GET/PUT methods. The profile page only reads from the `useSupabaseAuth` context. If these break, they do not block the core product experience.

## 3. Admin (Medium Difficulty)
*   **Target Components:** `AdminPage.tsx`
*   **Reasoning:** The admin page requires the implementation of the `AdminDashboardAdapter`. The visual components (cards, charts) are complex, but the data fetching logic (`GET /api/admin/*`) is simple and unidirectional. It introduces the concept of data mapping adapters without the risk of breaking real-time chat interactions.

## 4. Chat Core (High Difficulty)
*   **Target Components:** `ChatPage.tsx`, Composer, Sidebar
*   **Reasoning:** This is the heart of Sentinel-E EVO. We must replace the legacy `FigmaChatShell.js` and `ChatEngineV5.js` with the unified `ChatPage.tsx`. This requires wiring `useStore` to handle session IDs, mapping the message array precisely, and routing all submissions through `sendMCOQuery()` in `api.js`. This step carries high regression risk and must be tested extensively.

## 5. Debate / Evidence / Glass Panels (Highest Difficulty)
*   **Target Components:** `CinematicDebatePanel.tsx`, `CinematicEvidencePanel.tsx`, `OmegaInsightPanel.tsx`
*   **Reasoning:** These orchestrator visualizers depend on complex, nested JSON objects returned inside `machine_metadata` or `reasoning_json` from the backend's cognitive engines. The adapters required to safely parse, validate, and render this data into the high-fidelity Figma components are intricate. Furthermore, they are highly dynamic based on real-time streaming states. They must only be attempted after the fundamental Chat Core is completely stable.
