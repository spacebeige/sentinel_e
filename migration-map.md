# Sentinel-E EVO — Migration Map

## Overview
This document maps the current `integrationv0` application state (routing, APIs, and state dependencies) to the target Figma components and layouts found in the `integration-ui-backend` branch. It also assesses the complexity of transplanting these UI elements into the clean `integrationv0` baseline.

## 1. Authentication & Onboarding
| Current Route / Page | Current APIs | Target Figma Component | Complexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `/login` | N/A (Supabase Client) | `LoginPage.tsx` | Low | Minimal logic changes. Ensure Supabase Auth context remains wired. |
| `/signup` | N/A (Supabase Client) | `SignupPage.tsx` | Low | UI styling update. |
| `/auth/callback` | Supabase Session Exchange | `AuthCallbackPage.tsx` | Medium | Must preserve `ensure_user_exists` sync logic with backend. |
| `/profile` | `GET /api/user` | `ProfilePage.tsx`, `CompleteProfilePage.tsx` | Medium | Requires mapping existing user state to the new visual layout. |

## 2. Core Navigation & Layout
| Current Element | Current State Dependencies | Target Figma Component | Complexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| App Shell | ThemeContext, AuthContext | `Layout.tsx`, `Navbar.tsx`, `Footer.tsx` | High | The new `Layout` structure might break existing CSS scoping. Must merge carefully. |
| Routing Guard | Session State | `ProtectedRoute.tsx` | Low | Logic is likely identical; update visual loading state if changed. |

## 3. Conversational Interface (The Core)
| Current Route / Page | Current APIs | Target Figma Component | Complexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `/chat` | `GET /api/history`, `POST /api/chat` | `ChatPage.tsx` (Major Overhaul) | Very High | `ChatPage.tsx` has massive changes (+958 lines). Need to map V2 endpoints (`/api/chat/{id}/message` and `/api/mco/run`) strictly without overwriting existing backend integrations. |
| Debate Mode Rendering | `POST /api/mco/run` | `CinematicDebatePanel.tsx` | High | New visualizer component. Must map `machine_metadata.rounds` to the visual timeline. |
| Evidence/Glass Mode | `POST /api/mco/run` | `CinematicEvidencePanel.tsx` | High | Must parse `reasoning_json` or metadata accurately to populate the evidence list. |
| Omega / Deep Probe | `POST /api/mco/run` | `OmegaInsightPanel.tsx` | High | Relies on deep tree traversal of model outputs. |

## 4. Settings & Configuration
| Current Route / Page | Current APIs | Target Figma Component | Complexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `/settings` | `GET /api/user/settings`, `PUT /api/user/settings` | `SettingsPage.tsx` | Medium | Connect the styled toggles/forms to the existing Axios PUT requests. |
| `/models`, `/engines` | Static config | `ModelsPage.tsx`, `EnginesPage.tsx` | Low | Mostly static informational pages. |

## 5. Administration
| Current Route / Page | Current APIs | Target Figma Component | Complexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `/admin` | `GET /api/admin/system/stats` | `AdminPage.tsx`, `AdminModal.tsx` | Medium | Re-link the newly styled admin cards to the V2 `system_statistics` endpoint. |
| `/admin/requests` | N/A (V2 uses direct DB or placeholder) | `AdminRequestsPage.tsx` | Low | Ensure role-based access control (`role === 'admin'`) is preserved. |

## 6. Services & State Transplantation Strategy
*   **API Client**: Do NOT overwrite `integrationv0`'s `apiClient.ts` completely. Inspect the diff and cherry-pick UI-specific interceptors or error handling.
*   **Analytics & Session**: Keep the latest `integrationv0` versions of `sessionManager.ts` and `analyticsService.ts`, as they are tightly coupled to the backend logic.
*   **CSS / Styling**: Safely merge `theme.css` and `tailwind.config.js` to ensure the design system tokens are available without breaking global resets.

## Next Steps
Awaiting explicit approval to begin the UI transplantation process from `integration-ui-backend` to `integrationv0` according to this migration map.
