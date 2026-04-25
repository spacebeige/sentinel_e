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
- **Firebase Integration** for authentication and user management
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
  ↓ HTTP + JWT
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

2. Create `.env` file:

```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
NVIDIA_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key
SERPER_API_KEY=your_key
FIREBASE_PROJECT_ID=sentinel-c69c7
FIREBASE_PRIVATE_KEY=your_key
JWT_SECRET_KEY=your_secret
DATABASE_URL=postgresql://user:pass@localhost/sentinel_e
```

3. Initialize database:

```bash
psql -U postgres -d sentinel_e -f backend/storage/schema.sql
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
REACT_APP_FIREBASE_API_KEY=AIzaSyAeqmYqh_18lyXmhPyVMbWKUcmJ07QNzEI
REACT_APP_FIREBASE_PROJECT_ID=sentinel-c69c7
REACT_APP_API_URL=http://localhost:8000
```

Access: `http://localhost:3000`

---

## Roles & Access Control

### User Roles

| Role | Permissions | How Set |
|------|-------------|---------|
| **User** | Chat, feedback, personal sessions | Default on signup |
| **Admin** | System stats, user management, dashboard | Via /api/admin/users/make-admin |
| **Moderator** | Reserved for future use | Database role field |

### How It Works

1. JWT token issued on session creation
2. Backend queries User table for role on each request
3. Endpoints check role with `@require_admin()` decorator
4. All data filtered by user_id (session isolation)

### Promote to Admin

```bash
TOKEN=$(curl -X POST http://localhost:8000/api/auth/session | jq -r '.access_token')

curl -X POST 'http://localhost:8000/api/admin/users/make-admin' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
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

# Create session
curl -X POST http://localhost:8000/api/auth/session

# List models
curl http://localhost:8000/api/models
```

### Using the System

1. Visit `http://localhost:3000`
2. System creates anonymous session automatically
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

- `POST /api/auth/session` — Create session + JWT
- `POST /api/auth/refresh` — Refresh token

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

**users**: Accounts (user_id, email, role, active)

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

1. Create session and get JWT token
2. Promote user: `/api/admin/users/make-admin`
3. Login as admin user
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
│   │   ├── services/ (API, Firebase, session)
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
- [ ] Firebase credentials set
- [ ] JWT secret generated
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
✅ **Firebase Integration** — Auth and session management  
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

_Sentinel-E is a production-grade multi-model reasoning engine built with FastAPI, React, and modern AI orchestration._
