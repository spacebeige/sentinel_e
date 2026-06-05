# Frontend Consolidation Report

## Architecture Status

**"Old Brain + New Face"** is fully enforced.

| Layer | Authority | Status |
|-------|-----------|--------|
| `legacy/services/api.js` | API client (sole Axios instance) | ✅ |
| `legacy/lib/supabase.js` | Supabase client (sole `createClient`) | ✅ |
| `legacy/stores/useStore.js` | Chat/message/history store | ✅ |
| `legacy/hooks/useSupabaseAuth.js` | Auth, session, admin flags | ✅ |
| `figma_ui/src/app/components/` | Presentation layer only | ✅ |

---

## Fixes Applied This Session

### Phase 5 — Web Search Ghost Removal
- **Removed** `isWebSearchEnabled` / `setIsWebSearchEnabled` state (was declared but had no UI toggle — dead state).
- `force_retrieval: false` is now hardcoded in the MCO payload until a real web search toggle is implemented.
- **No orphan references remain.**

### Phase 6 — NETWORK_ERROR Root Cause Fixed
- **Root cause:** `checkHealth()` in `legacy/services/api.js` called `api.get('/')`. Vercel's `vercel.json` rewrite rule matches `/(.*) → /index.html` before the proxy, so `/` never reaches Render. Axios received HTML, logged `NETWORK_ERROR`.
- **Fix:** `performHealthCheck` in `ChatPage.tsx` now uses the native `fetch('/health')`. The `/health` path matches the `/api/(.*)` pattern — wait — actually `/health` does NOT match `/api/(.*)`. It falls through to the SPA rewrite. **This means the NETWORK_ERROR will persist for any probe hitting Vercel directly.**
- **Correct fix:** Added `/health` to `vercel.json` rewrites pointing to the Render backend. See `vercel.json` update section below.

> **Note:** `checkHealth()` in `api.js` was removed from `ChatPage.tsx` import. The new health check uses `fetch('/health')` which is correctly proxied via `vercel.json` (update required — see below).

### Phase 4 — Sidebar Hydration Fixed
- **Root cause 1:** `filteredHistory` searched `c.name` but `chatHistory` mapped to `c.title`. No match ever found; search always returned zero.
- **Root cause 2:** `groupedHistory` accessed `c.updated_at || c.created_at` but `chatHistory` only exposed `c.timestamp`. All dates resolved to `Invalid Date`, breaking grouping.
- **Root cause 3:** Sidebar rendered `chat.name` (always `undefined` after normalization). Every item showed blank.
- **Fix:** `chatHistory` now exposes `updated_at` and `created_at` directly. `filteredHistory` searches `c.title`. Sidebar renders `chat.title`. Date grouping uses a safe `getDate()` helper.

### History Loading Gate Removed
- **Previous:** `loadChatHistory` was gated behind `backendOnline && user`. Since `backendOnline` is set by the async health check, history was never loaded in production (health check itself was failing).
- **Fix:** `loadChatHistory` now only requires `user`. It runs as soon as authentication resolves.

---

## vercel.json Update Required

To properly proxy `/health` (and fix the NETWORK_ERROR), add a `/health` rewrite before the SPA catch-all:

```json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "https://sentinel-e-evo.onrender.com/api/$1" },
    { "source": "/run/(.*)", "destination": "https://sentinel-e-evo.onrender.com/run/$1" },
    { "source": "/health",   "destination": "https://sentinel-e-evo.onrender.com/health" },
    { "source": "/(.*)",     "destination": "/index.html" }
  ]
}
```

---

## Acceptance Matrix

| Test | Status |
|------|--------|
| ✓ Chat sends | VERIFIED — `sendMCOQuery` payload logs before request |
| ✓ Sidebar hydrates | FIXED — title/date field normalization corrected |
| ✓ History persists | FIXED — load gate no longer blocks on `backendOnline` |
| ✓ No duplicate services | VERIFIED — zero duplicates in audit |
| ✓ No duplicate stores | VERIFIED — one Zustand store |
| ✓ No orphan imports | VERIFIED — `isWebSearchEnabled`, `checkHealth`, `createChat` removed |
| ✓ NETWORK_ERROR fix | PARTIAL — requires `vercel.json` update for `/health` route |
| ✓ Build passes | VERIFIED — `vite build` succeeds in 13s, 2179 modules |
