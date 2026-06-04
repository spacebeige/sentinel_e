# Legacy Dependency Audit

This report documents all external NPM packages imported by the authoritative Sentinel-E architecture components now being consumed by the `figma_ui` application via path aliases. 

## Packages Imported by Authoritative Legacy Layers

| Package | Imported From | Used By | Present In `figma_ui/package.json`? |
|---------|---------------|---------|--------------------------------------|
| `axios` | `frontend/src/services/api.js` | `@services/api` alias calls | **No** |
| `react` | `frontend/src/hooks/useModels.js` | `@hooks/useModels` | **Yes** (via `peerDependencies`) |
| `react` | `frontend/src/hooks/useSupabaseAuth.js` | `@hooks/useSupabaseAuth` | **Yes** (via `peerDependencies`) |
| `react` | `frontend/src/stores/cognitiveStore.js` | `@stores/cognitiveStore` | **Yes** (via `peerDependencies`) |
| `react-router-dom` | `frontend/src/hooks/useAuthContext.js` | `@hooks/useAuthContext` | **No** (Vite uses `react-router` v7 instead) |
| `zustand` | `frontend/src/stores/useStore.js` | `@stores/useStore` | **Yes** (`"zustand": "latest"`) |

## Summary
The UI alias resolution correctly maps the Figma components to the legacy code. However, the legacy components introduce two new implicit dependency chains (`axios`, `react-router-dom`) that the `figma_ui` environment lacks. 

Furthermore, even for the packages that *are* present (`zustand`, `react`), Node Module resolution paths differ structurally, leading to downstream build failures on Vercel.
