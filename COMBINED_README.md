# Sentinel‑E — Combined Integration & Deployment README

This document combines the core project documentation (implementation summary, integration guide, migration notes, and stabilization checklist) and adds explicit environment variable guidance for Neon/Postgres, Clerk auth (current production), and optional Firebase configuration (legacy/alternate flows). It also summarizes the small runtime fixes applied to align the frontend and backend identity flows.

---

## Project Overview

Sentinel‑E is a chat and memory platform refactored to use Neon (Postgres) as the single source of truth for users, sessions, chats, messages, memory, and metadata. The recent v5 refactor introduced a deterministic persistence model and API contract that guarantees never-null responses and idempotent transactional writes.

Key highlights:
- Neon PostgreSQL as SOT (serverless-optimized connection pooling)
- FastAPI backend with SQLAlchemy async + Alembic migrations
- Deterministic API envelope: { success, data, error }
- Clerk for auth and immutable provider user_id mapping (default)
- Frontend defensive state handling to avoid wiping UI on transient failures

---

## What changed (high level)

- Database: normalized schema for users, sessions, chats, messages, memory, settings, embeddings. No nullable core identity fields.
- Auth: primary integration is Clerk (Clerk JWT verification in `backend/gateway/auth_v2.py`). The backend extracts the provider user id (Clerk `sub`) and uses it as `users.id`.
- API: new `backend/api/endpoints_v2.py` exposing deterministic endpoints (session, history, chat, message, memory, context, user settings).
- Frontend: centralized API client `frontend/src/services/api.js` that unwraps the envelope and injects Clerk tokens. App initialization now creates a backend session and then fetches history.
- Runtime fixes applied: removed legacy Firebase usage from the UI, fixed `SessionSidebar` to call backend API, guarded chat overwrites, added logs to trace user_id flow.

---

## Files of interest (condensed)
- `backend/database/models_v2.py` — normalized schema
- `backend/database/connection_v2.py` — async engine + Neon tuning
- `backend/database/crud_v2.py` — idempotent transactional CRUD
- `backend/gateway/auth_v2.py` — auth dependency + token verification
- `backend/api/endpoints_v2.py` — v2 API router for session/history/chat/messages
- `backend/utils/safe_responses.py` — envelope builders & fallbacks
- `frontend/src/services/api.js` — axios wrapper, token interceptor, API calls
- `frontend/src/App.js` — app init: create session → load history
- `frontend/src/components/ChatEngineV5.js` — chat UI logic, send guard
- `frontend/src/components/SessionSidebar.js` — now uses API for history

---

## ENVIRONMENT VARIABLES — What to set and where

Two locations: backend (server) and frontend (client). Use a `.env` file at the repository root or per-service `.env` as appropriate. Do NOT commit secrets.

### Backend (`backend/.env` or environment at Render)

Required for Neon + Clerk (current default):

- DATABASE_URL
  - Example: postgresql+asyncpg://<user>:<pass>@<host>:<port>/<db>
  - Use Neon connection string (ensure asyncpg driver). Example for Neon serverless: `postgresql+asyncpg://<user>:<password>@<project>.db.neon.tech/<db>?sslmode=require`
- CLERK_SECRET_KEY
  - Clerk backend secret for JWT verification (set from Clerk dashboard).
  - Example: `sk_live_...`
- REDIS_URL (optional)
  - Example: `redis://:password@hostname:6379/0`
- DB_POOL_SIZE (optional)
  - Example: `20`
- DB_POOL_RECYCLE (optional)
  - Example: `3600`
- SENTRY_DSN (optional)

If you use Firebase token verification on the backend (alternate flow) you would instead add:
- FIREBASE_SERVICE_ACCOUNT_JSON
  - Either the JSON string (not recommended) or a path via `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account JSON file with `firebaseadmin` privileges.
  - Example: `GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/firebase-service-account.json`

Notes:
- The backend currently expects a Clerk token; do not mix provider tokens unless you extend `auth_v2.verify_token_any_provider`.

### Frontend (`frontend/.env` or Vercel env vars)

- REACT_APP_API_URL
  - Example: `https://api.yourdomain.com` (used by `API_BASE` in the client)
- REACT_APP_CLERK_PUBLISHABLE_KEY
  - The Clerk publishable key for the frontend (exposed to browser). Example: `pk_live_...`

Optional for Firebase (if you re-enable Firebase auth):
- REACT_APP_FIREBASE_API_KEY
- REACT_APP_FIREBASE_AUTH_DOMAIN
- REACT_APP_FIREBASE_PROJECT_ID
- REACT_APP_FIREBASE_STORAGE_BUCKET
- REACT_APP_FIREBASE_MESSAGING_SENDER_ID
- REACT_APP_FIREBASE_APP_ID
- REACT_APP_FIREBASE_MEASUREMENT_ID

Notes:
- For Clerk the client uses the publishable key via `ClerkProvider` in `frontend/src/index.js`.
- If you switch from Clerk → Firebase auth on the frontend, you must update the backend verification flow accordingly (see below).

---

## How auth flows are expected to align (Clerk default)

1. Frontend obtains Clerk session token (`getToken()`), and sends requests with header:
   Authorization: Bearer <token>
   Optional debug header: `X-Debug-User: <user_id>` (client extracts `sub` from token payload for tracing only)
2. Backend (`auth_v2.get_current_user`) extracts token from Authorization header, verifies with `CLERK_SECRET_KEY`, decodes claims and reads `sub` as `user_id`.
3. Backend upserts/ensures the user via `database/crud_v2.upsert_user` using `user_id` (string). All chats/messages reference `user_id` in the DB.
4. `POST /api/session` uses the resolved `user_id` to create a session row (UUID primary key + `user_id` FK).
5. `GET /api/history` queries `chats` WHERE `chats.user_id = current_user_id` and returns all chats + messages.

This creates a consistent chain: Clerk `sub` → backend `user_id` → Neon `users.id` and foreign keys in `chats.messages`.

---

## Alternate auth: If you need Firebase instead of Clerk (minimal changes)

If you must use Firebase instead of Clerk, make these minimal adjustments:

1. FRONTEND
   - Initialize Firebase JS SDK and call `await firebase.auth().currentUser.getIdToken()` to obtain the ID token.
   - Send the token in the same header format:
     Authorization: Bearer <firebase_id_token>
   - Provide Firebase SDK config via `REACT_APP_FIREBASE_*` env vars listed above.

2. BACKEND
   - Install `firebase-admin` (python: `firebase-admin` package) and load service account credentials via `GOOGLE_APPLICATION_CREDENTIALS` or `FIREBASE_SERVICE_ACCOUNT_JSON`.
   - Replace or extend `auth_v2.verify_clerk_token` with a `verify_firebase_token` implementation:
     ```py
     import firebase_admin
     from firebase_admin import auth as firebase_auth

     cred = firebase_admin.credentials.Certificate('/path/to/serviceAccount.json')
     firebase_admin.initialize_app(cred)

     def verify_firebase_token(token):
         decoded = firebase_auth.verify_id_token(token)
         return decoded  # contains 'uid'
     ```
   - Then in `get_current_user` use `user_id = decoded['uid']` and ensure `user_id` is stored in `users.id`.

3. DB
   - Ensure `users.id` values are Firebase `uid` strings. No schema changes required.

**Caveat**: Don't mix Clerk and Firebase tokens unless you intentionally support multiple providers and map provider + id to `users.id` consistently. The repo includes a multi-provider placeholder in `auth_v2.verify_token_any_provider` — extend that if needed.

---

## ENV VARIABLES Summary (copyable)

Backend `.env` (example):

```
DATABASE_URL=postgresql+asyncpg://sentinel_user:strongpass@neon-host:5432/sentinel_db
CLERK_SECRET_KEY=sk_live_xxx
CLERK_PUBLISHABLE_KEY=pk_live_xxx
REDIS_URL=redis://:redispass@redis-host:6379/0
DB_POOL_SIZE=20
DB_POOL_RECYCLE=3600
SENTRY_DSN=
GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/firebase-service-account.json   # only if using Firebase
```

Frontend `.env` (example for Clerk):

```
REACT_APP_API_URL=https://api.sentinel.example
REACT_APP_CLERK_PUBLISHABLE_KEY=pk_live_xxx
```

Frontend `.env` (example for Firebase):

```
REACT_APP_API_URL=https://api.sentinel.example
REACT_APP_FIREBASE_API_KEY=...
REACT_APP_FIREBASE_AUTH_DOMAIN=...
REACT_APP_FIREBASE_PROJECT_ID=...
REACT_APP_FIREBASE_APP_ID=...
```

---

## Runtime fixes applied in this branch (summary)

- Added instrumentation logs to trace `user_id` throughout requests (backend prints `BACKEND USER_ID: ...`, history/session endpoints log the same).
- Enforced app init order in `frontend/src/App.js`: `createSession()` → `reloadHistory()` to avoid race conditions.
- Hardened frontend state handling in `frontend/src/stores/useStore.js` to avoid overwriting cached chats/messages with empty responses.
- Removed legacy Firebase dependency usage in `frontend/src/components/SessionSidebar.js` and replaced with calls to backend `GET /api/history` and `POST /api/chat`.
- Guarded `ChatEngineV5` send path to block message sends if user id not present.

These changes were intentionally small and additive to stabilize runtime flows without touching DB schema or core CRUD logic.

---

## Testing checklist (quick)

1. Start backend (ensure `CLERK_SECRET_KEY` and `DATABASE_URL` set)
2. Start frontend (ensure `REACT_APP_CLERK_PUBLISHABLE_KEY` and `REACT_APP_API_URL` set)
3. Login from the frontend (Clerk sign-in)
4. Observe frontend console: `INIT FLOW USER: <id>`
5. Observe backend logs: `BACKEND USER_ID: <id>` and `HISTORY REQUEST USER_ID: <id>`
6. Create a chat and send a message
7. Refresh page and verify chat/message persists
8. Verify DB rows: `SELECT * FROM messages WHERE user_id = '<id>'` and `SELECT * FROM chats WHERE user_id = '<id>'`

---

## Troubleshooting

- If `GET /api/history` returns empty but DB has data: ensure the backend is reading the same `user_id` value from the token (check logs) and `chats.user_id` matches exactly.
- If frontend shows blank UI on transient error: check `useStore` persisted state and `setHistory` guard.
- If token verification fails: confirm `CLERK_SECRET_KEY` is correct and tokens are sent as `Authorization: Bearer <token>`.

---

## Next steps & recommendations

- If you plan to migrate auth providers, add explicit provider prefixing (e.g., `clerk:user_123` or `firebase:uid_xxx`) into `users.id` so multiple providers can coexist without collision.
- Add a small integration test that automates: login, create chat, send message, refresh, validate DB rows.
- Setup Sentry/Log streaming on Render to capture `BACKEND USER_ID` events for easier debugging during rollout.

---

If you want, I can now:
- write this file into the repository (it will be saved as `COMBINED_README.md`),
- or additionally update `backend/.env.example` and `frontend/.env.example` with the variable list above.

Which would you like me to do? If yes, I will create the combined README and optional env example files. 