# Sentinel-E — Multi-Model Cognitive Reasoning Engine

**Production-Ready AI Orchestration System**  
Version: 6.0.0-ensemble | Status: ✅ Production Ready | Last Updated: April 25, 2026

---

## What's Included

Sentinel-E is a complete system for multi-model AI reasoning:

- **9 Language Models** across 4 providers (Groq, Google, NVIDIA, Anthropic)
- **6 Operating Modes** (Standard, Debate, Evidence, Glass, Synthesis, Experimental)
- **10-Phase Orchestrator** for intelligent query routing and execution
- **Admin Dashboard** with 5-tab interface for system monitoring
- **Boundary Detection** with severity-driven safety policies
- **SuperTokens + Neon Authentication** with Google/GitHub login and secure sessions
- **Multi-Tier Memory** (session, short-term, long-term)
- **Real-Time Evidence Verification** with Tavily and SerperAPI
- **Confidence Calibration** from model agreement, not self-reported certainty

---

## Quick Start (2 Minutes)

```bash
# Terminal 1: Backend
cd /Users/ashwinagarkhed/sentinel_e
source .venv/bin/activate
python -m uvicorn backend.main:app --port 8000 --reload

# Terminal 2: Frontend
cd /Users/ashwinagarkhed/sentinel_e/frontend
npm start

# Access: http://localhost:3000
```

---

## How It Works

### The Process Flow

When you submit a query:

1. **Query Validation** → Check length, sanitize content
2. **Session Initialization** → Create or retrieve chat
3. **Memory Loading** → Inject conversation history
4. **Evidence Retrieval** → Optional external search
5. **Model Selection** → Choose based on complexity
6. **Parallel Execution** → Run models simultaneously
7. **Cross-Model Analysis** → Compare outputs, compute agreement
8. **Debate Rounds** → Optional adversarial reasoning (3+ rounds)
9. **Confidence Calibration** → Score from metrics, not models
10. **Response Synthesis** → Generate final answer with transparency

### Core Principle

**Agreement between models does not equal truth.** Sentinel-E stress-tests consensus to identify whether agreement is structurally stable or superficially coincidental.

---

## System Architecture

```
FRONTEND (React)
  ↓ HTTP + SuperTokens session cookies
BACKEND (FastAPI)
  ├─ API Gateway (auth, admin, middleware)
  ├─ Cognitive Orchestrator (10-phase pipeline)
  ├─ Execution Engines
  │  ├─ Standard (simple/parallel)
  │  ├─ Debate (3+ rounds adversarial)
  │  ├─ Synthesis (collaborative)
  │  ├─ Evidence (fact-checking + search)
  │  ├─ Glass (transparency/audit)
  │  └─ Sigma (experimental)
  ├─ Cognitive Gateway (9 models, 4 providers)
  ├─ Governance (boundary detection, refusal, policies)
  └─ Memory Engine (3-tier caching)
  ↓
PERSISTENCE
  ├─ PostgreSQL (chats, messages, users)
  ├─ SQLite (vision cache, sessions)
  └─ Redis (optional session cache)
```

---

## Setup Guide

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL (or SQLite for dev)
- npm

### Backend Setup

1. Activate environment:

```bash
cd /Users/ashwinagarkhed/sentinel_e
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2. Create `backend/.env`:

```env
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
API_DOMAIN=http://localhost:8000
WEBSITE_DOMAIN=http://localhost:3000

SUPERTOKENS_CONNECTION_URI=your_supertokens_core_or_managed_uri
SUPERTOKENS_API_KEY=
SUPERTOKENS_API_BASE_PATH=/auth
SUPERTOKENS_WEBSITE_BASE_PATH=/auth

GOOGLE_OAUTH_CLIENT_ID=your_google_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_google_client_secret
GITHUB_OAUTH_CLIENT_ID=your_github_client_id
GITHUB_OAUTH_CLIENT_SECRET=your_github_client_secret

GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
NVIDIA_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key
SERPER_API_KEY=your_key
DATABASE_URL=postgresql://user:pass@your-neon-host/sentinel_e?sslmode=require
```

3. Initialize database:

```bash
psql "$DATABASE_URL" -f backend/storage/schema.sql
```

4. Start backend:

```bash
python -m uvicorn backend.main:app --port 8000 --reload
```

### Frontend Setup

1. Install and start:

```bash
cd frontend
npm install
npm start
```

2. Create `frontend/.env.local`:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_AUTH_API_BASE_PATH=/auth
REACT_APP_AUTH_WEBSITE_BASE_PATH=/auth
```

Access: `http://localhost:3000`

---

## Auth Setup

### Neon PostgreSQL

1. Create a Neon Postgres database.
2. Copy the pooled connection string into `DATABASE_URL`.
3. Keep `sslmode=require` in the URL; Sentinel-E converts it for `asyncpg` automatically.
4. On first backend startup, the existing SQLAlchemy init extends the `users` table with:
   - `id`
   - `user_id`
   - `email`
   - `name`
   - `provider`
   - `role`
   - `created_at`

### SuperTokens

1. Create or self-host a SuperTokens Core instance.
2. Set `SUPERTOKENS_CONNECTION_URI` and `SUPERTOKENS_API_KEY` if your core requires one.
3. Keep `API_DOMAIN` pointed at Render/FastAPI and `WEBSITE_DOMAIN` pointed at Vercel/React.
4. `ALLOWED_ORIGINS` must include every frontend origin that will send cookies.

### Google OAuth

1. Create an OAuth client in Google Cloud.
2. Add your frontend callback origin and the SuperTokens backend callback URL from `/auth/callback/google`.
3. Put the client ID/secret into `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET`.

### GitHub OAuth

1. Create an OAuth app in GitHub Developer Settings.
2. Set the callback URL to your backend auth callback at `/auth/callback/github`.
3. Put the client ID/secret into `GITHUB_OAUTH_CLIENT_ID` and `GITHUB_OAUTH_CLIENT_SECRET`.

### Frontend Auth Flow

- `Login / Sign Up` now opens a modal, not a dedicated auth page.
- Google/GitHub sessions are stored in secure httpOnly cookies.
- Chat and Models routes open the auth modal when the user is not signed in.
- After sign-in, the navbar updates with the synced user profile from Neon.

---

## Roles & Access Control

### User Roles

| Role | Permissions | How Set |
|------|-------------|---------|
| **User** | Chat, feedback, personal sessions | Default on signup |
| **Admin** | System stats, user management, dashboard | Via /api/admin/users/make-admin |
| **Moderator** | Reserved for future use | Database role field |

### How It Works

1. SuperTokens creates the session and stores it in secure cookies
2. Backend verifies the session on protected requests
3. Sentinel-E upserts the authenticated user into Neon without duplicating by email
4. Endpoints check role with `@require_admin()` decorator
5. All data filtered by user_id (session isolation)

### Promote to Admin

```bash
curl -X POST 'http://localhost:8000/api/admin/users/make-admin' \
  -H 'Content-Type: application/json' \
  --cookie "sAccessToken=<your_session_cookie>; sRefreshToken=<your_refresh_cookie>" \
  -d '{"email": "user@example.com"}'
```

---

## Operating Modes

### Standard Mode

Single query → parallel execution → best response

- **Trivial queries**: Single fast model (Llama 8B)
- **Moderate queries**: 2-3 models in parallel
- **Complex queries**: Full ensemble (all models)

### Debate Mode

Multi-round adversarial reasoning

- **Round 1**: Independent positions
- **Round 2**: Rebuttals (stable models only)
- **Round 3+**: Final positions if disagreement > 40%
- **Output**: Position trajectories, agreement matrix, conflict analysis

### Evidence Mode

Fact-checking with web search

- Queries Tavily + SerperAPI
- Extracts claims from responses
- Cross-references web sources
- Shows source reliability
- Detects contradictions

### Glass Mode

Full transparency and audit

Shows: reasoning steps, scoring breakdown, routing decisions, token usage, confidence components, trust metrics

### Synthesis Mode

Collaborative reasoning

- Anchor model produces draft
- Peer review from other models
- Iterative refinement
- Consensus scoring
- Optional Claude enhancement (500 token cap)

### Experimental Mode (Sigma)

Stress testing and boundary analysis

- Hypothesis extraction
- Boundary violation detection
- Safety scenario testing
- Full diagnostic output

---

## Running the System

### Start Services

**Backend Terminal**:
```bash
cd /Users/ashwinagarkhed/sentinel_e
source .venv/bin/activate
python -m uvicorn backend.main:app --port 8000 --reload
```

**Frontend Terminal**:
```bash
cd /Users/ashwinagarkhed/sentinel_e/frontend
npm start
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Auth status
curl -X POST http://localhost:8000/api/auth/session

# List models
curl http://localhost:8000/api/models
```

### Using the System

1. Visit `http://localhost:3000`
2. Open `Login / Sign Up` and authenticate with Google or GitHub
3. Type query and select mode (Standard/Debate/Evidence/Glass/Synthesis)
4. Submit query
5. View response with reasoning transparency
6. Rate with 👍 👎 buttons

---

## API Endpoints

### Core Execution

- `POST /api/mco/run` — Main MCO orchestrator
- `POST /api/run/standard` — Standard mode
- `POST /api/run/experimental` — Experimental mode

### Authentication

- `POST /api/auth/session` — Current auth/session status
- `GET /api/auth/me` — Current signed-in user
- `POST /api/auth/sync-user` — Upsert signed-in user into Neon

### Models

- `GET /api/models` — List registered models
- `GET /api/models/status` — Real-time availability
- `POST /api/models/claude/toggle` — Enable/disable Claude
- `GET /api/models/claude/usage` — Usage statistics

### Chat

- `GET /api/chats` — List chats
- `GET /api/chat/{chat_id}/messages` — Get messages
- `PUT /api/messages/{message_id}` — Edit message
- `POST /api/messages/{message_id}/regenerate` — Regenerate response

### Feedback & Admin

- `POST /feedback` — Record feedback (👍 👎)
- `GET /feedback/stats` — Aggregate statistics
- `GET /api/admin/system/stats` — System stats (admin only)
- `GET /api/admin/system/architecture` — System architecture
- `GET /api/admin/web-analytics?days=7` — Engagement breakdown
- `POST /api/admin/users/make-admin` — Promote user

### Health

- `GET /health` — Health check
- `GET /api/optimization/stats` — Performance metrics

---

## Database Schema

### PostgreSQL (Main)

**chats**: Sessions (chat_id, session_id, user_id, created_at)

**messages**: Chat messages (message_id, chat_id, role, content, image_b64)

**users**: Accounts (`user_id`, `email`, `name`, `provider`, `role`, `active`)

**uploaded_assets**: Files (asset_id, session_id, file_type, file_hash)

### SQLite (Session Cache)

**vision_cache**: Image processing (image_hash → summary)

**context_assets**: Per-message assets

**session_cache**: Session state with TTL

### Redis (Optional)

Best-effort session mirror with configurable TTL

---

## Admin Dashboard

### Access

1. Sign in with Google or GitHub
2. Promote user: `/api/admin/users/make-admin`
3. Refresh the session or sign in again if role changed
4. Visit `/admin` route

### 5 Tabs

**Overview**: Key metrics (users, chats, messages, feedback)

**Analytics**: 7-day engagement breakdown, feedback distribution

**Architecture**: System layers, reasoning pipeline, model registry

**Feedback**: Feedback by mode, sentiment analysis, recent items

**Users**: Admin promotion form, user management

---

## Safety & Governance

### Boundary System

Detects claims exceeding epistemic boundaries:

- **Minimal (10)**: Fully grounded
- **Low (30)**: Minor gaps
- **Medium (50)**: Substantial gaps
- **High (70)**: Significant gaps → refusal triggered
- **Critical (90)**: Purely speculative

### Refusal Logic

```
if boundary_severity >= threshold (default: 70):
    → Show refusal message with reason
else:
    → Execute request
```

### Safety Policies

Configurable policies that override execution:
- Confidence floor (minimum 0.65)
- Topic boundaries
- Model disagreement thresholds
- Content policy checks

### Feedback System

Users rate responses with 👍 👎:
- Optional reason field for negative feedback
- Read-only telemetry (never affects response)
- Used for system improvement

---

## Deployment & Troubleshooting

### Quick Deployment

```bash
# Backend
python -m uvicorn backend.main:app --port 8000

# Frontend
npm start
```

### Docker

```bash
docker build -f backend/Dockerfile -t sentinel-e .
docker run -p 8000:8000 -e GROQ_API_KEY=$GROQ_API_KEY sentinel-e
```

### Common Issues

**Backend won't start**
- Check Python: `python --version` (need 3.11+)
- Verify env vars: `echo $GROQ_API_KEY`
- Clear cache: `rm -rf __pycache__`

**Frontend won't load**
- Check Node: `node --version` (need 20+)
- Clear cache: `npm cache clean --force`
- Rebuild: `npm install && npm start`

**Database errors**
- Check PostgreSQL: `psql -U postgres`
- Verify schema: `psql -d sentinel_e -c "\dt"`

**Models not responding**
- Verify API keys in `.env`
- Check rate limits
- Test: `curl http://localhost:8000/api/models/status`

---

## Project Structure

```
sentinel_e/
├── backend/
│   ├── main.py (FastAPI app)
│   ├── core/ (orchestration, debate, synthesis)
│   ├── gateway/ (auth, admin, middleware)
│   ├── database/ (models, CRUD)
│   ├── memory/ (3-tier system)
│   ├── optimization/ (token, cost, cache)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/ (Chat, Mode views, Dashboard)
│   │   ├── services/ (API, auth, session)
│   │   ├── pages/ (AdminDashboard)
│   │   └── App.js
│   ├── package.json
│   └── .env.local
├── data/ (session storage)
└── README.md (this file)
```

---

## Model Registry

| Model | Provider | Tier | Role |
|-------|----------|------|------|
| Llama 70B | Groq | 1 | Analysis |
| Qwen 32B | Groq | 2 | Cross-analysis |
| Llama Scout 17B | Groq | 2 | Critique |
| Llama 8B | Groq | 3 | Verification |
| Gemini Flash 2.0 | Google | 3 | Synthesis |
| Mistral Large | NVIDIA | 1 | Deep analysis |
| Kimi K2 | NVIDIA | 2 | Extended critique |
| Qwen VL 7B | Qwen | 2 | Vision input |
| Claude Sonnet | Anthropic | — | Optional |

**Debate Tiers**: Tier 1 generates analysis, Tier 2 rebuts, Tier 3 synthesizes

---

## Environment Checklist

- [ ] API keys in `.env`
- [ ] Database URL configured
- [ ] SuperTokens connection URI configured
- [ ] Google + GitHub OAuth credentials set
- [ ] Node 20+ installed
- [ ] Python 3.11+ installed
- [ ] PostgreSQL running
- [ ] Frontend `.env.local` configured
- [ ] Admin user promoted
- [ ] Health checks pass

---

## Key Features

✅ **Multi-Model Orchestration** — 9 models across 4 providers  
✅ **Debate Mode** — 3+ round adversarial reasoning  
✅ **Evidence Verification** — Real-time web search integration  
✅ **Transparency** — Full reasoning visibility (Glass mode)  
✅ **Admin Dashboard** — System monitoring and management  
✅ **Safety Policies** — Severity-driven refusal system  
✅ **SuperTokens + Neon Auth** — Social login and session management  
✅ **Confidence Calibration** — From model agreement, not self-reported  
✅ **Memory Engine** — 3-tier caching system  
✅ **Production Ready** — Tested and deployed

---

## Next Steps

1. Complete Setup Guide above
2. Start backend and frontend
3. Promote admin user
4. Access admin dashboard at `/admin`
5. Start asking questions!

**Support**: Check logs with `tail -f /var/log/sentinel_e.log` or test endpoints with curl

---


---

## 🔐 Legacy SuperTokens Reference App

> **Note:** Sentinel-E now uses the integrated FastAPI + React auth flow documented above. The `/supertokens-auth` directory remains as a separate reference implementation only.

### What's Included

- **Backend:** Express.js + SuperTokens Node SDK (ThirdParty + Session recipes)
- **Frontend:** Next.js App Router + SuperTokens React SDK (prebuilt UI, protected dashboard)
- **Database:** Neon PostgreSQL (schema + migrations)
- **Core:** Self-hosted SuperTokens Core (Docker Compose)
- **Deliverables:**
  - `/supertokens-auth/backend` — Express API, session middleware, Neon integration
  - `/supertokens-auth/frontend` — Next.js App Router, prebuilt auth UI, protected dashboard
  - `/supertokens-auth/db` — SQL schema, migration scripts
  - `/supertokens-auth/docker-compose.yml` — SuperTokens Core
  - `/supertokens-auth/README.md` — Step-by-step setup, OAuth config, deployment
  - `.env.example` files for all components

### Key Features

- Google & GitHub OAuth (ThirdParty recipe)
- Secure HTTPOnly cookie sessions (production-ready)
- CORS with credentials, CSRF protection, secure cookie defaults
- Protected API routes (`/api/auth/me`, `/api/protected/dashboard`, etc.)
- Neon upsert on login, role-ready schema
- Next.js protected dashboard, prebuilt auth UI, logout/session flows
- All code and config isolated from main Sentinel-E app

### Quick Start

1. Copy `.env.example` files in `/supertokens-auth/backend`, `/frontend`, and `/db` to `.env` and fill secrets (see `/supertokens-auth/README.md` for details)
2. Start SuperTokens Core: `docker-compose up -d` in `/supertokens-auth`
3. Run backend: `cd backend && npm install && npm start`
4. Run frontend: `cd frontend && npm install && npm run dev`
5. Open `http://localhost:3000` and sign in with Google/GitHub

See `/supertokens-auth/README.md` for full setup, OAuth callback config, and deployment notes.

---

_Sentinel-E is a production-grade multi-model reasoning engine built with FastAPI, React, and modern AI orchestration._
