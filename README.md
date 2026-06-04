# Sentinel-E

Sentinel-E is a multi-model reasoning platform with orchestration, debate, analytics, tactical mapping, and persistent conversation memory.

This repository now uses **Supabase authentication as the primary frontend auth layer** with mandatory sign-in before protected app access.

## Migration status

- ✅ Supabase auth client + provider wrapper added
- ✅ GitHub OAuth implemented as the primary login flow
- ✅ Mandatory authentication enabled for protected routes (`/chat`, `/models`, `/admin`)
- ✅ Persistent authenticated session restore on refresh
- ✅ User-scoped conversation/session persistence utilities
- ✅ Firebase auth code preserved (not deleted), isolated for rollback/reference

> Preserved Firebase code includes TODO markers like:
>
> `// TODO: Remove or fully restore Firebase auth after Supabase migration stabilizes`

## Current auth architecture

### Frontend

- `frontend/src/lib/supabase.js`  
  Centralized Supabase client (PKCE, persistent sessions, token refresh).

- `frontend/src/services/supabaseSessionManager.js`  
  Session restore, OAuth sign-in/out, auth snapshot handling, auth subscription.

- `frontend/src/hooks/useSupabaseAuth.js`  
  Auth hydration hook used by the app auth provider.

- `frontend/src/hooks/useAuthContext.js`  
  Provider wrapper used by the app UI/routing (`AuthProvider` preserved).

- `frontend/src/components/ProtectedRoute.js`  
  Blocks protected routes until authenticated and hydrated.

### Session/conversation persistence

- `frontend/src/services/sessionPersistence.js` provides centralized persistence utilities:
  - `restoreUserSession()`
  - `saveConversationHistory()`
  - `loadConversationHistory()`
  - `persistSessionState()`
  - `switchConversation()`
  - `createNewConversation()`

Persistence uses:

- `localStorage` for long-lived user-scoped conversation/session memory
- `sessionStorage` for hydration race guards
- Supabase auth session persistence for authenticated continuity

## Mandatory authentication flow

1. User opens Sentinel-E.
2. Protected pages require auth and trigger login modal if unauthenticated.
3. Login uses Supabase GitHub OAuth (`signInWithOAuth({ provider: 'github' })`).
4. On callback, session is restored automatically.
5. User-scoped session + conversation history are hydrated without wiping orchestration state.

## Supabase setup (required)

### 1. Create Supabase project

https://supabase.com/dashboard

### 2. Enable GitHub provider in Supabase

1. Open **Authentication** → **Providers** → **GitHub**
2. Enable provider
3. Add GitHub OAuth client ID + secret

### 3. Create GitHub OAuth app

https://github.com/settings/developers

Recommended values:

- **Homepage URL**: your frontend URL (or `https://sentinel-e-evo.vercel.app` for local)
- **Authorization callback URL**:  
  `https://<your-project-ref>.supabase.co/auth/v1/callback`

### 4. Configure Supabase auth URLs

In Supabase (**Authentication** → **URL Configuration**):

- Site URL: your frontend base URL
- Redirect URLs:
  - `https://sentinel-e-evo.vercel.app/chat`
  - your production chat route URL

## Environment variables

### Frontend (`frontend/.env.local`, Vercel env)

```env
REACT_APP_API_URL=
REACT_APP_SUPABASE_URL=
REACT_APP_SUPABASE_ANON_KEY=
REACT_APP_GUEST_MODE=false
```

### Backend (`backend/.env`, Render backend env)

```env
SUPABASE_SERVICE_ROLE_KEY=
```

### Vercel (Frontend) — required keys

Set these in **Vercel Project → Settings → Environment Variables** for Production/Preview:

1. `REACT_APP_API_URL` (your backend URL)
2. `REACT_APP_SUPABASE_URL`
3. `REACT_APP_SUPABASE_ANON_KEY`
4. `REACT_APP_GUEST_MODE=false`

Do **not** add `SUPABASE_SERVICE_ROLE_KEY` to Vercel frontend env.

### Render/Backend — optional Supabase server key

Only add this to backend if you implement server-side Supabase admin operations:

1. `SUPABASE_SERVICE_ROLE_KEY`

### Security rules

- Never expose `SUPABASE_SERVICE_ROLE_KEY` in frontend bundles.
- Only `REACT_APP_SUPABASE_URL` and `REACT_APP_SUPABASE_ANON_KEY` belong in frontend env.
- Do not expose Firebase Admin credentials in frontend.
- Production default must keep guest fallback disabled: `REACT_APP_GUEST_MODE=false`.

## Running locally

### Backend

```bash
cd /Users/ashwinagarkhed/sentinel_e
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd /Users/ashwinagarkhed/sentinel_e/frontend
npm install
npm start
```

## Sentinel-E compatibility guarantees during migration

The migration keeps existing core systems intact:

- ensemble orchestration
- debate engine
- tactical mapping
- analytics/telemetry panels
- issue extraction
- visualization payloads
- conversation rendering
- model routing

## Notes on Firebase preservation

- Firebase files are intentionally retained.
- Unstable Firebase execution paths stay isolated/commented.
- Auth UI structure is preserved while login execution is routed through Supabase.
- Firebase restoration remains possible from preserved code paths.
