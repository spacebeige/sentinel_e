# Supabase Config Audit

## Objective
The `figma_ui` is a Vite-based application, which requires environment variables to be accessed via `import.meta.env.VITE_*`. The legacy frontend logic heavily utilized Webpack's `process.env.REACT_APP_*` pattern. This audit details the successful migration of all initialization secrets to ensure compatibility with Vercel deployment targets.

## Environment Variable Replacement

| File | Old Variable | Replacement Variable |
|------|--------------|----------------------|
| `figma_ui/src/legacy/config.js` | `process.env.REACT_APP_API_URL` | `import.meta.env.VITE_API_URL` |
| `figma_ui/src/legacy/lib/supabase.js` | `process.env.REACT_APP_SUPABASE_URL` | `import.meta.env.VITE_SUPABASE_URL` |
| `figma_ui/src/legacy/lib/supabase.js` | `process.env.REACT_APP_SUPABASE_ANON_KEY` | `import.meta.env.VITE_SUPABASE_ANON_KEY` |
| `figma_ui/src/legacy/services/supabaseSessionManager.js` | `process.env.REACT_APP_RUNTIME_ADMIN_EMAILS` | `import.meta.env.VITE_RUNTIME_ADMIN_EMAILS` |

## Results
- `REACT_APP` yields **0** remaining hits within the UI codebase.
- The UI layer strictly consumes native `VITE_` prefixed environment variables.
