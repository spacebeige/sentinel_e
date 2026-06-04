# Sentinel-E EVO — UI Map (integration-ui-backend)

## Overview
This document inventories the UI assets, components, layouts, and design system elements present in the `integration-ui-backend` branch. This branch serves as the authoritative source for UI design and user experience.

## 1. Figma UI Structure (`figma_ui/`)
The frontend is encapsulated in the `figma_ui` directory. It uses Vite, React, and TailwindCSS.

### Core Directories
*   `src/app/components/`: Reusable React components and page assemblies.
*   `src/app/components/figma/`: Specific components exported or mapped from Figma designs.
*   `src/app/config/`: Frontend configuration (routing names, runtime environment variables).
*   `src/app/context/`: React context providers (Themes, Chat state).
*   `src/app/hooks/`: Custom React hooks (debounce, Lenis smooth scrolling, session logic).
*   `src/app/orchestration/`: UI representations of backend orchestration modes (games, visual renders).
*   `src/app/services/`: API client abstractions.
*   `src/styles/`: Global stylesheets (Tailwind, typography, custom theme variables).

## 2. Page Inventory
*   `HomePage.tsx`: Landing page with Hero Section and Feature overviews.
*   `LoginPage.tsx` & `SignupPage.tsx`: Authentication flows.
*   `AuthCallbackPage.tsx`: Supabase OAuth callback handler.
*   `ChatPage.tsx`: The primary conversational interface. Extensively overhauled in this branch to include dynamic panels and new layout structures.
*   `ExplorePage.tsx` & `ModelsPage.tsx` & `EnginesPage.tsx`: Overviews of system capabilities.
*   `PricingPage.tsx`: Tier selection for Pro modes.
*   `ProfilePage.tsx` & `CompleteProfilePage.tsx`: User identity and onboarding.
*   `SettingsPage.tsx`: User configuration (theme, models, notifications).
*   `AdminPage.tsx` & `AdminRequestsPage.tsx`: Administrator dashboards.

## 3. Component Inventory
### Core Chat Components
*   `CinematicDebatePanel.tsx`: Visualizer for multi-agent debate mode.
*   `CinematicEvidencePanel.tsx`: Display for forensic search and evidence synthesis.
*   `CinematicOrchestratorLoader.tsx`: Loading state for complex orchestrations.
*   `CrossAnalysisPanel.tsx`: UI for comparing multiple model outputs.
*   `OmegaInsightPanel.tsx`: Dedicated insight renderer for Omega mode.
*   `SessionAnalyticsPanel.tsx`: Telemetry visualization.

### Layout & Navigation
*   `Layout.tsx`: Main application shell.
*   `Navbar.tsx`: Top navigation bar.
*   `Footer.tsx`: Application footer.
*   `Fence.tsx`, `House.tsx`, `Lamp.tsx`, `Rock.tsx`, `Tree.tsx`, `WaterTile.tsx`, `PathTile.tsx`: Micro-components for isometric/gamified visual flair.

### Utility Components
*   `ErrorBoundary.tsx` & `CinematicErrorBoundary.tsx`: Fallback UI for React crashes.
*   `ImageWithFallback.tsx` (in `figma/`): Resilient image loading.
*   `ModeIconRenderer.tsx`: Dynamic icon selection based on chat mode.
*   `ProtectedRoute.tsx`: Route guard enforcing authentication.

## 4. Design System Inventory
*   **TailwindCSS**: The primary styling engine, configured in `tailwind.config.js`.
*   **Theme Tokens (`theme.css`)**: CSS variables defining the color palette (vibrant/dark modes), spacing, and border radii.
*   **Typography (`fonts.css`)**: Custom font imports and baseline mappings.
*   **Global Styles (`index.css`)**: Resets and global utility classes.

## 5. Responsive Framework
*   Utilizes Tailwind's mobile-first breakpoints (`sm:`, `md:`, `lg:`, `xl:`).
*   Components like `Navbar` and `ChatPage` conditionally render sidebars and panels based on screen width.
*   `useLenis.ts` integrates smooth scrolling for non-touch devices, degrading gracefully on mobile.

## 6. Reusable UI Assets
*   **Icons**: Rendered via `lucide-react` and `ModeIconRenderer.tsx`.
*   **Gamified Elements**: 3D-styled or pixel-art components in the root components directory (e.g., `Tree.tsx`, `WaterTile.tsx`).
*   **Modals**: `AdminModal.tsx`, `ProfileModal.tsx` for overlay interactions.
