# SuperTokens + Neon Production Authentication System

This folder contains a full open-source, self-hostable authentication stack using:

- **Backend:** Node.js + Express
- **Frontend:** Next.js (App Router)
- **Auth:** SuperTokens (ThirdParty + Session recipes)
- **Data:** Neon PostgreSQL (user metadata, roles, app data)
- **Auth Core:** SuperTokens Core (self-hosted via Docker)

## Project Structure

```text
supertokens-auth/
├── backend/                  # Express API + SuperTokens Node SDK
│   ├── src/
│   │   ├── auth/             # SuperTokens initialization
│   │   ├── db/               # PostgreSQL access + migrations runner
│   │   ├── middleware/       # Session validation middleware
│   │   ├── routes/           # Auth and protected routes
│   │   └── server.js
│   └── .env.example
├── frontend/                 # Next.js App Router app
│   ├── app/
│   │   ├── auth/[[...path]]/ # SuperTokens prebuilt auth UI route
│   │   ├── dashboard/        # Protected page
│   │   └── page.js
│   ├── lib/
│   └── .env.example
├── db/
│   ├── schema.sql
│   └── migrations/001_init.sql
├── docker-compose.yml        # SuperTokens Core (self-hosted)
└── .env.example
```

## 1) Prerequisites

- Node.js 20+
- Docker + Docker Compose
- Neon PostgreSQL project
- Google OAuth app
- GitHub OAuth app

## 2) Environment Setup

1. Copy root env template:

```bash
cp .env.example .env
```

1. Copy backend env template:

```bash
cp backend/.env.example backend/.env
```

1. Copy frontend env template:

```bash
cp frontend/.env.example frontend/.env.local
```

1. Fill all variables in these files.

> Important: `SUPERTOKENS_API_KEY` in backend and root must match if you enable Core API key protection.

## 3) Neon PostgreSQL Setup

1. Create a Neon project and database.
1. Copy the connection string with SSL:

```text
postgresql://USER:PASSWORD@HOST/DB_NAME?sslmode=require
```

1. Set:
   - `NEON_DATABASE_URL` in `backend/.env`
   - `SUPERTOKENS_POSTGRESQL_CONNECTION_URI` in root `.env`

1. Run schema migration:

```bash
cd backend
npm install
npm run db:migrate
```

## 4) Start SuperTokens Core (Self-hosted)

From `supertokens-auth/`:

```bash
docker compose up -d
```

Core endpoint defaults to `http://localhost:3567`.

## 5) OAuth Provider Setup

### Google OAuth

1. Open Google Cloud Console.
1. Create OAuth Client ID (Web application).
1. Add authorized redirect URI:

```text
http://localhost:4000/auth/callback/google
```

1. Put credentials in `backend/.env`:
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`

### GitHub OAuth

1. Open GitHub Settings → Developer settings → OAuth Apps.
1. Create a new OAuth App.
1. Set authorization callback URL:

```text
http://localhost:4000/auth/callback/github
```

1. Put credentials in `backend/.env`:
   - `GITHUB_CLIENT_ID`
   - `GITHUB_CLIENT_SECRET`

## 6) Run Backend and Frontend

### Backend

```bash
cd backend
npm install
npm run dev
```

Runs on `http://localhost:4000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:3000`.

## 7) Core Features Implemented

### Authentication

- Google OAuth login
- GitHub OAuth login
- Secure session cookies (HTTPOnly via SuperTokens)
- Logout API + frontend logout flow
- Session verification middleware (`requireSession`)

### Backend APIs

- `GET /health` → health check
- `GET /api/auth/me` → current session + app user metadata (protected)
- `POST /api/auth/sync-user` → upsert user into Neon after login (protected)
- `POST /api/auth/logout` → revoke session (protected)
- `GET /api/protected/dashboard` → sample protected route

### Database

- `users` table (`id`, `supertokens_user_id`, `email`, `provider`, timestamps)
- `roles` table (optional)
- `user_roles` table (optional)

## 8) Security Notes

- All secrets are environment variables.
- Session cookies are managed by SuperTokens and HTTPOnly.
- CSRF protection enabled through SuperTokens defaults (`antiCsrf: "VIA_TOKEN"`).
- Production cookie strategy uses `cookieSecure=true` and `SameSite=None`.
- CORS uses explicit origin allowlist + `credentials: true`.

## 9) Production Deployment Checklist

- [ ] Set HTTPS domains in `API_DOMAIN` and `WEBSITE_DOMAIN`
- [ ] Keep `NODE_ENV=production`
- [ ] Ensure reverse proxy forwards `X-Forwarded-Proto`
- [ ] Use strong `SUPERTOKENS_API_KEY`
- [ ] Restrict CORS to real frontend domains
- [ ] Rotate OAuth secrets periodically

## 10) Example Protected Request

After signing in from frontend:

```bash
curl -i http://localhost:4000/api/protected/dashboard \
  -H "Cookie: sAccessToken=...; sRefreshToken=..."
```

You should receive protected JSON payload with `userId`.

---

This stack is fully open-source and self-hostable. No Firebase, Clerk, or proprietary auth providers are used.
