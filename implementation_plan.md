# Production Deployment Integrity Fix Plan

This plan details the steps required to eliminate old references, fix the authentication initialization on Vite, route fallback paths correctly for SPA on Vercel, and address the missing assets in the `figma_ui` environment.

## Open Questions

> [!WARNING]
> **Missing Assets:** The files `favicon.ico` and `sentinel-e(1).png` do not exist anywhere in the repository tree. I can either:
> 1. Remove the `<img src="/sentinel-e(1).png" />` references from the UI components.
> 2. Create blank/transparent placeholder files in `figma_ui/public/` to prevent 404s.
> 
> *Which approach do you prefer for the missing assets?*

## Proposed Changes

### 1. Backend URL Remediation
We will remove all fallback instances of `https://sentinel-e.onrender.com` to prevent silent misrouting.

#### [MODIFY] `figma_ui/src/app/services/config.ts`
- Change fallback to `https://sentinel-e-evo.onrender.com`.

#### [MODIFY] `figma_ui/src/legacy/config.js`
- Change `process.env.REACT_APP_API_URL` to `import.meta.env.VITE_API_URL`.
- Change fallback URL to `https://sentinel-e-evo.onrender.com`.

### 2. Environment Variable Migration (Auth Fix)
The Supabase initialization fails because Vite uses `import.meta.env.VITE_*` rather than `process.env.REACT_APP_*`. We will migrate the legacy files inside `figma_ui/src/legacy` to use the Vite standard.

#### [MODIFY] `figma_ui/src/legacy/lib/supabase.js`
- Replace `process.env.REACT_APP_SUPABASE_URL` with `import.meta.env.VITE_SUPABASE_URL`.
- Replace `process.env.REACT_APP_SUPABASE_ANON_KEY` with `import.meta.env.VITE_SUPABASE_ANON_KEY`.

#### [MODIFY] `figma_ui/src/legacy/services/supabaseSessionManager.js`
- Replace `process.env.REACT_APP_RUNTIME_ADMIN_EMAILS` with `import.meta.env.VITE_RUNTIME_ADMIN_EMAILS`.

### 3. Vercel SPA Routing Fix
Vercel is throwing 404s on direct navigation (like `/login`) because the `figma_ui` Root Directory lacks SPA rewrite rules.

#### [NEW] `figma_ui/vercel.json`
- Create a Vercel configuration file with SPA routing:
```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

### 4. Asset Audit & Remediation
Depending on your answer to the open question, we will either remove the missing asset references from components (like `figma_ui/src/app/components/Footer.tsx`) or create placeholder files in `figma_ui/public/`.

## Verification Plan

### Automated Tests
1. Generate the required audit reports:
   - `backend-url-audit.md`
   - `supabase-config-audit.md`
   - `route-audit.md`
   - `asset-audit.md`
   - `auth-initialization-report.md`
2. Run `npm run build` inside `figma_ui` to ensure no build errors are introduced.
3. Validate that a search for `sentinel-e.onrender.com` (without `-evo`) and `REACT_APP` yields zero results in the `figma_ui` tree.

### Manual Verification
- Deploy to Vercel and manually test direct URL navigation (e.g., refresh `/login`).
- Verify that Supabase authentication initializes correctly without console warnings.
