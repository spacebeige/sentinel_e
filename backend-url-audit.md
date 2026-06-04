# Backend URL Audit

## Search Scope
- `figma_ui/` (including `src/legacy/`)
- `frontend/`

## Objective
Identify all remaining hardcoded legacy backend references and migrate them to `https://sentinel-e-evo.onrender.com` or dynamically inject them via `import.meta.env.VITE_API_URL`.

## Modified Files

| File | Old Value | New Value |
|------|-----------|-----------|
| `figma_ui/src/app/services/config.ts` | `API_BASE: import.meta.env.VITE_API_URL || "https://sentinel-e.onrender.com"` | `API_BASE: import.meta.env.VITE_API_URL` (Strict checking added) |
| `figma_ui/src/legacy/config.js` | `export const API_BASE = process.env.REACT_APP_API_URL || "https://sentinel-e.onrender.com"` | `export const API_BASE = import.meta.env.VITE_API_URL` (Strict checking added) |

## Results
- `sentinel-e.onrender.com` yields **0** remaining hits within the UI codebase.
- The UI layer correctly and safely sources its API endpoint solely from the Vercel production environment variable without fallback risks.
