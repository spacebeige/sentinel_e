# Supabase Singleton Audit

## Objective
Identify the cause of the `Multiple GoTrueClient instances detected` warning in production.

## Findings
A repository-wide search for `createClient` revealed that the application is currently maintaining two separate Supabase client initialization files which are both being imported by the active bundle.

1. **`figma_ui/src/app/lib/supabase.ts`**
2. **`figma_ui/src/legacy/lib/supabase.js`**

Both of these files independently invoke `createClient(supabaseUrl, supabaseAnonKey)`.

### Import Consumers
Because Vercel bundling resolves aliases statically:
- Consumers utilizing `@hooks/useSupabaseAuth` (which resolves to `legacy/hooks`) eventually import the client from `legacy/lib/supabase.js`.
- Consumers importing `../lib/supabase` directly from within `src/app` (like `ChatPage.tsx`, `apiClient.ts`) import the client from `app/lib/supabase.ts`.

## Conclusion
**How many Supabase clients exist:** `2`
**Expected:** `1`

The presence of two independent `supabase.ts/js` initialization files creates a dual-singleton scenario. Since both clients attempt to read and write to the same `localStorage` session keys simultaneously and manage their own token refresh intervals, they trigger the Supabase SDK's multiple instance warning.

To fix this, one of the `lib/supabase` instances must be strictly designated as the sole initialization source, and the other must be deleted, with all imports routed to the surviving singleton.
