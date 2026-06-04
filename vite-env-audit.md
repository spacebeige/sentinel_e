# Vite Environment Audit

## Objective
Identify where `VITE_API_URL` is consumed, determine the source of the uncaught exception that crashed the Vercel app, and document the runtime safety check modifications.

## VITE_API_URL Usages

1. **`figma_ui/src/legacy/config.js`**
   - **Line 1:** `if (!import.meta.env.VITE_API_URL)`
   - **Line 5:** `export const API_BASE = import.meta.env.VITE_API_URL || "";`
   - **Import Chain:** `api.js` -> `config.js` -> `useStore.js` -> `ChatPage.tsx` -> `App.tsx`

2. **`figma_ui/src/app/services/config.ts`**
   - **Line 6:** `if (!import.meta.env.VITE_API_URL)`
   - **Line 11:** `API_BASE: import.meta.env.VITE_API_URL || ""`

## Crash Source

**Which file threw the exception?** 
*Both* `figma_ui/src/legacy/config.js` and `figma_ui/src/app/services/config.ts` threw the exception because their modules evaluated at the top-level upon import during app bootstrap.

Before modification, the pattern was:
```javascript
if (!import.meta.env.VITE_API_URL) {
  throw new Error("VITE_API_URL is not configured");
}
```

Since Vercel did not have the `VITE_API_URL` environment variable configured securely in the project's dashboard at build time, Vite statically evaluated `import.meta.env.VITE_API_URL` as undefined. When the browser parsed the bundle, the `throw new Error` immediately executed, crashing the entire React tree before it could mount.

## Runtime Safety Check Implementation
The code has been successfully updated in both files to gracefully warn rather than crash:

```javascript
if (!import.meta.env.VITE_API_URL) {
  console.error("Configuration Error: VITE_API_URL missing");
}
```

This prevents the white-screen crash on production while ensuring backend fallbacks remain safely excluded.

## Final Summary
1. **Why multiple GoTrue clients exist:** Because two distinct `lib/supabase` files exist (`legacy/lib/supabase.js` and `app/lib/supabase.ts`) and are both actively imported by various parts of the codebase.
2. **Is Vercel missing VITE_API_URL?:** Yes. The variable was not defined in the Vercel dashboard prior to the build, leading to an immediate crash due to top-level `throw` directives.
3. **Deployment Configuration Issue Remaining:** Vercel Environment Variables. You must ensure `VITE_API_URL`, `VITE_SUPABASE_URL`, and `VITE_SUPABASE_ANON_KEY` are all populated in the Vercel project settings, and trigger a new deployment.
