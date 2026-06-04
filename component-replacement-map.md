# Sentinel-E EVO — Component Replacement Map

## 1. Landing & Shell Components
| Current Component | Replacement Component | State Dependencies | Service Dependencies | Adapter Required? |
| :--- | :--- | :--- | :--- | :--- |
| `LandingPage.js`, `HomePage.js` | `HomePage.tsx`, `HeroSection.tsx` | None | None | No. Direct mount. |
| `Navbar.js` | `Navbar.tsx` | Auth Context | `useSupabaseAuth` | No. Drop-in replacement. |
| `Layout.js` | `Layout.tsx` | Theme Context | None | No. Drop-in replacement. |

## 2. Chat Core Components
| Current Component | Replacement Component | State Dependencies | Service Dependencies | Adapter Required? |
| :--- | :--- | :--- | :--- | :--- |
| `ChatPage.js` / `FigmaChatShell.js` | `ChatPage.tsx` | `useStore` (`chat_id`, `chats`) | `api.js` | Yes. Must wire routing params. |
| `ChatEngineV5.js` | `ChatPage.tsx` (Internal logic) | `useStore` (`messages`, `mode`) | `api.js` (`sendMCOQuery`) | Yes. `ChatEngineAdapter` required to bind UI handlers to `api.js`. |
| `InputArea.js` | Component within `ChatPage.tsx` | `useModels` | `api.js` | Yes. Ensure `mode` and `model` selection flow correctly. |
| `SessionSidebar.js` | Component within `ChatPage.tsx` | `useStore` | `api.js` (`getHistory`) | Yes. Must wire history load action. |

## 3. Orchestration Visualizers
| Current Component | Replacement Component | State Dependencies | Service Dependencies | Adapter Required? |
| :--- | :--- | :--- | :--- | :--- |
| `DebateArena.js` | `CinematicDebatePanel.tsx` | `cognitiveStore.debate_rounds` | None | Yes. `DebateMetadataAdapter`. |
| `EvidenceConsole.js` | `CinematicEvidencePanel.tsx` | `cognitiveStore.evidence_chain`| None | Yes. `EvidenceMetadataAdapter`. |
| `GlassConsole.js` | `OmegaInsightPanel.tsx` | `cognitiveStore` | None | Yes. `GlassAdapter`. |

## 4. Administration & Utilities
| Current Component | Replacement Component | State Dependencies | Service Dependencies | Adapter Required? |
| :--- | :--- | :--- | :--- | :--- |
| `AdminDashboard.js` | `AdminPage.tsx` | `useAdminRole` | `api.js` (`/api/admin/*`) | Yes. `AdminDashboardAdapter`. |
| `themeManager.js` logic | `SettingsPage.tsx` | LocalStorage, Theme Context | `api.js` | No. |
| N/A | `ProfilePage.tsx` | Auth Context | `api.js` | No. |
