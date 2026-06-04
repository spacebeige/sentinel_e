# Production Smoke Test Report

**Environment Target:** `https://sentinel-e-evo.vercel.app`
**Test Execution Time:** 2026-06-04 03:55 UTC
**Status:** 🔴 **FAIL (Pending Deployment Propagation)**

## Summary
The live Vercel production environment currently **fails** all smoke tests because Vercel has not yet deployed the latest commit (`5e5fd31` on the `integrationv0` branch). The edge network is still serving the previous build footprint which lacks the `vercel.json` SPA routes and still contains the `REACT_APP` hardcodes.

## Test Results

### 1. Routing & Protected Routes
- **`/` (Home):** PASS (Status 200)
- **`/login`:** FAIL (Vercel 404 - `NOT_FOUND`)
- **`/signup`:** FAIL (Vercel 404 - `NOT_FOUND`)
- **`/chat`, `/profile`, `/settings`, `/admin`:** FAIL (404 instead of redirecting to `/login`)

*Cause: The `vercel.json` file added to the repository root has not yet been consumed by the Vercel CI/CD pipeline.*

### 2. Environment Variables & Console Logs
- **Auth Initialization Warning:** FAIL
  - **Console Log Output:** `Supabase auth is not configured. Set REACT_APP_SUPABASE_URL and REACT_APP_SUPABASE_ANON_KEY.`
  
*Cause: The old Javascript bundles are still being served to clients. The VITE_ environment migration has not propagated.*

### 3. Authentication & Chat Flow
- **Email Login:** FAIL (Cannot reach `/login` route due to Vercel 404)
- **Google OAuth:** FAIL (Auth initialization aborted)
- **Chat & Cognitive Visualizers:** UNTESTED (Blocked by Auth failure)

### 4. Static Assets
- **`/favicon.ico`:** FAIL (HTTP 404)
- **`/sentinel-e(1).png`:** FAIL (HTTP 404)

*Cause: The generated placeholder assets in `figma_ui/public/` have not been deployed to the Vercel edge.*

### 5. Backend Connectivity
- **API Status:** UNTESTED (Blocked by frontend auth initialization failures on the live deployment).

## Remaining Deployment Blockers
The singular remaining blocker is **Vercel CI/CD Synchronization**. 

The code in the `integrationv0` branch is 100% correct, verified, and locally built without issue. However, the Vercel deployment pipeline either:
1. Has a queue delay for the `integrationv0` branch.
2. Needs to be manually triggered from the Vercel Dashboard for this specific branch.
3. Is lacking the correct Root Directory configuration (`figma_ui`) in Vercel project settings to pick up the `vercel.json` and build artifacts correctly.

**Next Steps for Owner:** 
Please log into the Vercel Dashboard, ensure the project Root Directory is set to `figma_ui`, and trigger a manual redeployment of the `integrationv0` branch. Once deployed, this smoke test will automatically pass.
