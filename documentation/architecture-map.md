# Sentinel-E EVO — Architecture Map (integrationv0)

## Overview
This document outlines the complete architectural structure of the `integrationv0` branch, which represents the clean baseline imported from the latest `main`.

## 1. Backend Structure
The backend is a FastAPI application driven by a cognitive architecture and multiple execution engines.

*   **API Layer (`backend/api/`)**
    *   `endpoints_v2.py`: The Phase 2-10 API contracts (Session, Chat, History, Memory, Config).
    *   `orchestration_routes.py`: Endpoints for model inference and multi-agent routing.
    *   `persistence_routes.py`: Additional persistence layers.
*   **Core Cognitive Runtime (`backend/core/`)**
    *   `agentic_orchestrator.py`: Orchestrates multi-step reasoning.
    *   `deliberative_orchestrator.py`: Manages the debate mode workflow.
    *   `structured_debate_engine.py`: Handles structured multi-agent debates.
    *   `knowledge_memory.py`: User-specific memory extraction and injection.
    *   `glass_pipeline.py` & `evidence_engine.py`: Evidence retrieval and synthesis.
*   **Data Access Layer (`backend/database/`)**
    *   `models_v2.py`: The deterministic, normalized Neon PostgreSQL schema.
    *   `crud_v2.py`: V2 CRUD operations mapped to `models_v2.py`.
    *   `connection_v2.py`: Asyncpg database connection and pooling factory.
*   **Gateway & Auth (`backend/gateway/`)**
    *   `auth_v2.py`: Supabase authentication and user initialization middleware.
    *   `chat_routes.py`: Legacy routes (to be mapped or deprecated).
    *   `admin_routes.py`: Admin dashboard routing and system statistics.
*   **Metacognitive Layer (`backend/metacognitive/`)**
    *   `cognitive_gateway.py`: Unified LLM model interface (Groq, Anthropic, Qwen, Gemini).

## 2. Frontend Structure
The frontend is a React application built with Vite and TailwindCSS, residing in `figma_ui/`.

*   **App Root (`src/app/`)**
    *   `App.tsx`: Main application wrapper and provider setup.
    *   `main.tsx`: DOM entry point.
*   **Routing (`src/app/routes.tsx`)**
    *   Maps endpoints to top-level page components.
    *   Protected by `ProtectedRoute.tsx`.
*   **Context & State (`src/app/context/`, `src/app/hooks/`)**
    *   `AuthContext` (via Supabase).
    *   `ChatInteractionContext` (chat UI state).
    *   `useSessionPersistence.ts` (syncing local state with backend session).
*   **Services (`src/app/services/`)**
    *   `apiClient.ts`: Core Axios instance with auth interceptors.
    *   `sessionManager.ts`: API calls for session lifecycle.
    *   `analyticsService.ts`: Telemetry tracking.

## 3. Route Tree (Frontend)
*   `/` -> `HomePage.tsx`
*   `/login` -> `LoginPage.tsx`
*   `/signup` -> `SignupPage.tsx`
*   `/auth/callback` -> `AuthCallbackPage.tsx`
*   `/chat` (Protected) -> `ChatPage.tsx`
*   `/profile` (Protected) -> `ProfilePage.tsx`
*   `/settings` (Protected) -> `SettingsPage.tsx`
*   `/admin` (Protected) -> `AdminPage.tsx`

## 4. Auth Flow
1.  User authenticates via Supabase on the frontend (Google OAuth or Email).
2.  Supabase handles redirect to `/auth/callback` and sets local JWT tokens.
3.  Frontend makes authenticated requests via `apiClient` using `Authorization: Bearer <token>`.
4.  Backend `auth_v2.py` verifies the JWT via Supabase or decodes it locally.
5.  Backend calls `ensure_user_exists` to sync the auth provider ID with the deterministic `users` table.

## 5. API Contracts (Backend V2)
*   `POST /api/session`: Initialize device session.
*   `GET /api/history`: Load unified chat history list.
*   `POST /api/chat`: Create a new conversational container.
*   `GET /api/chat/{id}`: Fetch chat by ID.
*   `POST /api/chat/{id}/message`: Append user message and trigger model execution.
*   `GET /api/chat/{id}/messages`: Fetch full message history for a specific chat.
*   `GET /api/user/settings`: Fetch user configuration.
*   `POST /api/mco/run`: (Orchestration) Trigger model response based on mode.

## 6. Persistence Layer
*   **Primary Database**: Neon Serverless PostgreSQL.
*   **ORM**: SQLAlchemy `2.0+` with `asyncpg`.
*   **Schema**: Completely normalized (UUIDs for primary keys, strict foreign keys).
*   **Migrations**: Managed via `alembic`.
*   **Vector/Search**: `pgvector` or external Pinecone connection for semantic evidence.

## 7. Chat System
*   Frontend `ChatPage.tsx` maintains local message history.
*   Upon submission, sends payload to `/api/chat/{id}/message` or `/api/mco/run`.
*   Backend resolves the `QueryMode` (Conversational, Debate, Deep Probe).
*   Orchestrator invokes the `CognitiveGateway` and returns the generated tokens.
*   Frontend receives and renders the response, including any `reasoning_json` or `machine_metadata`.

## 8. Admin System
*   `/api/admin/system/stats`: Aggregates DB statistics (total users, chats, tokens).
*   `/api/admin/requests`: Handles Pro account upgrade requests.
*   Admin dashboard polls these endpoints to display live telemetry.
