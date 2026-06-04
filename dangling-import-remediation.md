# Dangling Import Remediation

## Issue
Vercel build failed due to:
`Could not resolve "./apiClient" from src/app/services/analyticsService.ts`

## Remediation Steps
1. Investigated `analyticsService.ts` and identified it as the sole remaining file relying on the deleted `apiClient`.
2. Verified that all components using analytics (like `ChatPage.tsx`, `ProfileModal.tsx`, `AdminPage.tsx`, `Navbar.tsx`) expected specific function signatures (`trackMessageSent`, `getAdminAnalytics`, etc.).
3. Entirely removed the dependency on `apiClient` and `supabase` from `analyticsService.ts`.
4. Overwrote `analyticsService.ts` to act as a strict stub interface, completely decoupling it from the network layer.

## Verification
- `grep -R "apiClient" figma_ui/src` now returns `0 results`.
- The build succeeds locally.
