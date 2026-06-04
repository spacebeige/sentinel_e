# Auth Provider Trace

## Objective
Trace the import dependency chain connecting `AuthProvider.tsx`, `useSupabaseAuth`, and the underlying `supabase` initialization client.

## Import Chain

1. **`figma_ui/src/app/providers/AuthProvider.tsx`**
   - **Imports:** `useSupabaseAuth` from `@hooks/useSupabaseAuth`
   - **Notes:** The Vite alias `@hooks` maps to `figma_ui/src/legacy/hooks`.

2. **`figma_ui/src/legacy/hooks/useSupabaseAuth.js`**
   - **Imports:** `isSupabaseConfigured`, `supabase`, `getSupabaseClient` from `../lib/supabase`
   - **Notes:** Because of its location in `src/legacy/hooks/`, the relative path `../lib/supabase` resolves to the legacy folder.

3. **`figma_ui/src/legacy/lib/supabase.js`**
   - **Action:** Initializes the Supabase client using `import.meta.env.VITE_SUPABASE_URL`.
   - **Exports:** `supabase` (the active `GoTrueClient` instance).

## Disconnect
While the global `AuthProvider` correctly utilizes the legacy `useSupabaseAuth` (which uses the legacy `supabase.js`), various newer React components in `figma_ui/src/app/components` (e.g., `ChatPage.tsx`) directly import `useSupabaseAuth` from `@hooks/useSupabaseAuth` BUT ALSO manually import `supabase` via relative paths (`import { supabase } from '../lib/supabase'`), resolving to `figma_ui/src/app/lib/supabase.ts` instead. 

This fractured import chain forces both the `legacy` and `app` versions of the client to boot, confirming the dual-instance issue found in the Singleton Audit.
