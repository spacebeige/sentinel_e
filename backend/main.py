"""
============================================================
Sentinel-E v5.0 — Production API Gateway
============================================================
Layered Architecture:
  L1: API Gateway (this file) — routing, auth, middleware
  L2: Orchestrator           — request coordination
  L3: Model Interface        — provider abstraction
  L4: Cognitive Engine       — Omega kernel
  L5: Memory Engine          — 3-tier memory
  L6: Retrieval Engine       — cognitive RAG
  L7: Presentation Layer     — response formatting

Security:
  - JWT authentication
  - CSP headers
  - Rate limiting
  - Input validation
  - Prompt firewall
  - Centralized error handling
  - No credentials in logs
  - Strict CORS
"""

import sys
import os
import json
import logging
import base64
import asyncio
import sqlite3
import uuid as uuid_lib
import re
from collections import Counter
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
from uuid import UUID

# Add project root to path
# In the Docker container (Render root = backend/) main.py is at /app/main.py,
# so all sub-packages (gateway/, evaluation/, etc.) sit alongside it at /app/.
# Insert /app so they're importable regardless of whether PYTHONPATH is set.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Gateway Layer Imports ────────────────────────────────────
from gateway.config import get_settings
from gateway.auth import (
    get_current_user,
    get_optional_user,
)
from gateway.firebase_service import get_firebase_service, firebase_is_enabled
from gateway.middleware import (
    RateLimitMiddleware, SecurityHeadersMiddleware,
    RequestTrackingMiddleware, ErrorHandlerMiddleware,
    InputValidationMiddleware,
)
from gateway.prompt_firewall import get_firewall

# ── Database ─────────────────────────────────────────────────
from database.connection import get_db, init_db, check_redis, redis_client
from database.crud import (
    create_chat, get_chat, list_chats, update_chat_metadata,
    add_message, get_chat_messages, update_message, delete_messages_after,
    add_user_memory, get_user_memory, get_user_preferences, upsert_user_preference,
)
from database.models import UserMemory

# ── Core Engine Imports ──────────────────────────────────────
from sentinel.sentinel_sigma_v4 import SentinelSigmaOrchestratorV4
from sentinel.schemas import SentinelRequest
from core.omega_kernel import OmegaCognitiveKernel
from core.mode_config import ModeConfig
from core.knowledge_learner import KnowledgeLearner
from utils.chat_naming import generate_chat_name
from utils.output_sanitizer import sanitize_output
from core.context_builder import get_context_builder
from utils.api_response import api_response, api_error, api_success
from gateway.auth import get_user_id

# ── New Architecture Layers ──────────────────────────────────
from memory.memory_engine import MemoryEngine
from retrieval.cognitive_rag import CognitiveRAG
from core.dynamic_analytics import DynamicAnalyticsEngine

# ── Cognitive Core Engine v7.0 ────────────────────────────────
from core.cognitive_orchestrator import CognitiveCoreEngine

# ── Multimodal Capability Auditor ─────────────────────────────
from core.multimodal_auditor import MultimodalAuditor

# ── Sub-Mode Pipelines (Glass / Evidence / Synthesis) ─────────
from core.glass_pipeline import build_glass_result
from core.evidence_pipeline import build_evidence_result
from core.synthesis_engine import build_synthesis_result

# ── Optimization Layer ───────────────────────────────────────
from optimization import (
    get_token_optimizer,
    get_response_cache,
    get_fallback_router,
    get_cost_governor,
    get_observability_hub,
)

# ── Meta-Cognitive Orchestrator ──────────────────────────────
from metacognitive.orchestrator import MetaCognitiveOrchestrator
from metacognitive.background_daemon import BackgroundDaemon
from metacognitive.routes import router as mco_router, set_orchestrator as mco_set_orchestrator, set_daemon as mco_set_daemon, set_cognitive_engine as mco_set_cognitive_engine

# ── Battle Platform v2 ────────────────────────────────────────
from evaluation.routes import router as battle_router

# ── Standard Mode (direct model routing) ─────────────────────
from gateway.chat_routes import router as chat_router

# ── Admin Routes ──────────────────────────────────────────────
from gateway.admin_routes import router as admin_router
from gateway.workflow_admin_routes import router as workflow_admin_router

# ── Critical Security & Stability Fixes ───────────────────
from fixes.session_cache_manager import SessionCacheManager
from fixes.exception_handling import log_unhandled_exceptions, safe_execute
from fixes.config_validation import validate_production_config
from fixes.session_ownership_validation import validate_kernel_before_use

# ── Logging ──────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Sentinel-API")

# ── Environment Detection ───────────────────────────────────
ENV = os.getenv("ENV", "development").lower()

# ── Global State ─────────────────────────────────────────────
orchestrator: Optional[SentinelSigmaOrchestratorV4] = None
omega_kernel: Optional[OmegaCognitiveKernel] = None
knowledge_learner: Optional[KnowledgeLearner] = None
cognitive_rag: Optional[CognitiveRAG] = None
analytics_engine: Optional[DynamicAnalyticsEngine] = None
mco_orchestrator: Optional[MetaCognitiveOrchestrator] = None
mco_daemon: Optional[BackgroundDaemon] = None
mco_bridge = None  # MCOModelBridge — unified model client
cognitive_orchestrator_engine: Optional[CognitiveCoreEngine] = None  # Cognitive Engine v7.0
multimodal_auditor: Optional[MultimodalAuditor] = None  # Multimodal Capability Auditor

# FIX #1: Replace unbounded dictionaries with SessionCacheManager (thread-safe LRU with TTL)
# Prevents memory leak crashes; auto-evicts expired sessions
omega_sessions: SessionCacheManager = SessionCacheManager(max_sessions=500, ttl_minutes=60)
memory_sessions: SessionCacheManager = SessionCacheManager(max_sessions=500, ttl_minutes=60)
SESSION_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "data", "session_cache.db")
_SQLITE_SESSION_TABLE_READY = False


def _ensure_session_sqlite_table():
    """Ensure SQLite session cache table exists."""
    global _SQLITE_SESSION_TABLE_READY
    if _SQLITE_SESSION_TABLE_READY:
        return

    os.makedirs(os.path.dirname(SESSION_SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SESSION_SQLITE_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_cache (
                chat_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        _SQLITE_SESSION_TABLE_READY = True
    finally:
        conn.close()


def _sqlite_write_session(chat_id: str, payload: str):
    _ensure_session_sqlite_table()
    conn = sqlite3.connect(SESSION_SQLITE_PATH)
    try:
        conn.execute(
            """
            INSERT INTO session_cache (chat_id, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (chat_id, payload, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _sqlite_read_session(chat_id: str) -> Optional[str]:
    _ensure_session_sqlite_table()
    conn = sqlite3.connect(SESSION_SQLITE_PATH)
    try:
        row = conn.execute(
            "SELECT payload FROM session_cache WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ── Adapter: StructuredModelOutput → Pipeline-compatible ─────
# The sub-mode pipelines (glass, evidence, synthesis) expect objects
# with .output.success, .output.raw_output, .output.model_name,
# .score.final_score — matching MCO's ScoredModelResult interface.
# This adapter bridges StructuredModelOutput from the ensemble path.

class _PipelineScore:
    """Adapter mimicking MCO ModelScore for pipeline functions."""
    __slots__ = ("model_name", "topic_alignment", "knowledge_grounding",
                 "specificity", "confidence_calibration", "drift_penalty", "final_score")

    def __init__(self, model_name: str, confidence: float):
        self.model_name = model_name
        self.final_score = confidence
        self.topic_alignment = confidence
        self.knowledge_grounding = confidence * 0.9
        self.specificity = confidence * 0.85
        self.confidence_calibration = confidence
        self.drift_penalty = 0.0


class _PipelineOutput:
    """Adapter mimicking MCO ModelOutput for pipeline functions."""
    __slots__ = ("success", "raw_output", "model_name")

    def __init__(self, succeeded: bool, raw_output: str, model_name: str):
        self.success = succeeded
        self.raw_output = raw_output
        self.model_name = model_name


class _PipelineResult:
    """Adapter wrapping StructuredModelOutput for pipeline functions."""
    __slots__ = ("output", "score")

    def __init__(self, smo):
        self.output = _PipelineOutput(smo.succeeded, smo.raw_output, smo.model_name)
        self.score = _PipelineScore(smo.model_name, smo.confidence)


def _adapt_ensemble_for_pipelines(ensemble_response):
    """Convert EnsembleResponse model_outputs to pipeline-compatible lists."""
    adapted_results = [_PipelineResult(m) for m in ensemble_response.model_outputs]
    scoring_breakdown = [_PipelineScore(m.model_name, m.confidence) for m in ensemble_response.model_outputs]
    divergence_metrics = {
        "max_divergence": ensemble_response.ensemble_metrics.contradiction_density,
        "mean_divergence": ensemble_response.ensemble_metrics.disagreement_entropy,
    }
    return adapted_results, scoring_breakdown, divergence_metrics


STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "i", "you", "we", "they", "this", "that"
}


def extract_topics(text: str, top_n: int = 5) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", str(text or "").lower())
    filtered = [w for w in words if w not in STOPWORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


async def update_memory_bg(user_id: str, content: str, db: Optional[AsyncSession] = None):
    """Adaptive memory graph updater (non-fatal)."""
    from database.connection import get_db

    async def _update_with_session(session: AsyncSession):
        topics = extract_topics(content)
        now = datetime.utcnow()

        for topic in topics:
            result = await session.execute(
                select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.key == topic,
                )
            )
            existing = result.scalars().first()

            if existing:
                existing.weight = min(float(existing.weight or 1.0) + 1.0, 20.0)
                existing.last_used = now
                existing.recency_score = 1.0
                existing.value = (existing.value or content)[:300]
            else:
                session.add(UserMemory(
                    user_id=user_id,
                    key=topic,
                    value=str(content or "")[:300],
                    weight=1.0,
                    last_used=now,
                    recency_score=1.0,
                    confidence=1,
                ))

        cutoff = now - timedelta(days=7)
        stale_result = await session.execute(
            select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.last_used < cutoff,
            )
        )
        stale_entries = stale_result.scalars().all()
        for entry in stale_entries:
            entry.recency_score = max(float(entry.recency_score or 1.0) * 0.85, 0.05)
            if float(entry.recency_score or 0.0) < 0.1 and float(entry.weight or 1.0) < 2.0:
                await session.delete(entry)

        await session.commit()

    try:
        if db is not None:
            try:
                await _update_with_session(db)
                return
            except Exception:
                try:
                    await db.rollback()
                except Exception:
                    pass

        async for fresh_db in get_db():
            try:
                await _update_with_session(fresh_db)
            except Exception:
                await fresh_db.rollback()
                raise
            break
    except Exception as e:
        logger.warning(f"[memory_bg] non-fatal error: {e}")


async def safe_pinecone_query(index, **kwargs) -> list:
    try:
        result = await asyncio.to_thread(index.query, **kwargs)
        return list(getattr(result, "matches", []) or [])
    except Exception as e:
        logger.warning(f"[pinecone] non-fatal: {e}")
        return []


async def safe_pinecone_upsert(index, vectors: list) -> None:
    try:
        await asyncio.to_thread(index.upsert, vectors=vectors)
    except Exception as e:
        logger.warning(f"[pinecone] upsert non-fatal: {e}")


async def safe_build_context(db: AsyncSession, user_id: str, chat_id: Any, query: str) -> Dict[str, Any]:
    try:
        builder = get_context_builder()
        return await builder.build_context(db, user_id, chat_id, query)
    except Exception as e:
        logger.warning(f"[context] non-fatal: {e}")
        return {
            "context_str": "",
            "system_instructions": "",
            "recent_history": [],
            "context": "",
            "timestamp": datetime.utcnow().isoformat(),
        }


async def safe_memory_update(user_id: str, content: str, db: Optional[AsyncSession] = None) -> None:
    try:
        await update_memory_bg(user_id, content, db)
    except Exception as e:
        logger.warning(f"[memory] non-fatal: {e}")


async def _startup_optional_systems_check() -> Dict[str, bool]:
    systems = {"db": False, "pinecone": False, "memory": False}

    # DB is critical.
    try:
        async for db in get_db():
            await db.execute(text("SELECT 1"))
            systems["db"] = True
            break
    except Exception as e:
        logger.error(f"[startup] DB FAILED — this is critical: {e}")
        raise

    # Pinecone/vector service is optional.
    try:
        from utils.vector_service import get_vector_service
        _ = get_vector_service()
        systems["pinecone"] = True
    except Exception as e:
        logger.warning(f"[startup] Pinecone unavailable (non-fatal): {e}")

    # Memory table availability is optional.
    try:
        async for db in get_db():
            await db.execute(select(UserMemory).limit(1))
            systems["memory"] = True
            break
    except Exception as e:
        logger.warning(f"[startup] Memory table unavailable (non-fatal): {e}")

    logger.info(f"[startup] System status: {systems}")
    return systems


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, omega_kernel, knowledge_learner, cognitive_rag, analytics_engine
    global mco_orchestrator, mco_daemon, mco_bridge, cognitive_orchestrator_engine, multimodal_auditor
    logger.info("Initializing Sentinel-E v5.0 Production System...")

    # FIX #5: Validate production configuration on startup (fail fast)
    try:
        validate_production_config()
        logger.info("✅ Production configuration validated")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        if ENV == "production":
            raise

    # FIX #2: Install global exception handler (catch unhandled exceptions)
    log_unhandled_exceptions()

    # Initialize DB with timeout to prevent NeonDB cold start from blocking deploy
    try:
        await asyncio.wait_for(init_db(), timeout=20)
    except asyncio.TimeoutError:
        logger.warning("Database init timed out (NeonDB cold start?) — will retry on first request")
    except Exception as e:
        logger.warning(f"Database init failed (non-fatal): {e}")
    try:
        await asyncio.wait_for(check_redis(), timeout=10)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Redis check timed out or failed (non-fatal): {e}")
    await asyncio.to_thread(_ensure_session_sqlite_table)

    # Optional-system availability check (DB fatal, others non-fatal)
    await _startup_optional_systems_check()

    # Initialize Firebase Admin SDK
    firebase_service = await asyncio.to_thread(get_firebase_service)
    if firebase_is_enabled():
        logger.info("✓ Firebase Admin SDK initialized successfully")
    else:
        logger.warning("Firebase Admin SDK not initialized (optional — backend will work without it)")

    # Initialize core components
    try:
        orchestrator = SentinelSigmaOrchestratorV4()
        logger.info("✓ SentinelSigmaOrchestratorV4 initialized")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        orchestrator = None

    try:
        knowledge_learner = KnowledgeLearner()
    except Exception as e:
        logger.error(f"Failed to initialize knowledge_learner: {e}")
        knowledge_learner = None

    # New architecture layers
    try:
        cognitive_rag = CognitiveRAG()
    except Exception as e:
        logger.error(f"Failed to initialize cognitive_rag: {e}")
        cognitive_rag = None

    try:
        analytics_engine = DynamicAnalyticsEngine()
    except Exception as e:
        logger.error(f"Failed to initialize analytics_engine: {e}")
        analytics_engine = None

    # Optimization layer (lightweight singletons)
    try:
        get_token_optimizer()
        get_response_cache()
        get_fallback_router()
        get_cost_governor()
        get_observability_hub()
    except Exception as e:
        logger.warning(f"Optimization layer init issues (non-fatal): {e}")

    # ── Meta-Cognitive Orchestrator (MUST init before OmegaKernel) ─
    try:
        mco_orchestrator = MetaCognitiveOrchestrator()
        if redis_client:
            mco_orchestrator.set_redis(redis_client)
        mco_set_orchestrator(mco_orchestrator)

        # Create MCO bridge — unified model client that routes through gateway
        from models.mco_bridge import MCOModelBridge
        mco_bridge = MCOModelBridge(mco_orchestrator.cognitive_gateway)
        logger.info("MCO Model Bridge created — all model calls route through CognitiveModelGateway")

        # Log environment validation summary
        enabled = mco_bridge.get_enabled_models_info()
        from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
        total = len(COGNITIVE_MODEL_REGISTRY)
        logger.info(f"Model registry: {len(enabled)}/{total} models enabled")
        for m in enabled:
            logger.info(f"  ✓ {m['name']} ({m['registry_key']}) — provider={m['provider']}, vision={m['supports_vision']}")
        disabled_keys = [k for k, s in COGNITIVE_MODEL_REGISTRY.items() if not s.enabled]
        for dk in disabled_keys:
            logger.warning(f"  ✗ {dk} — DISABLED (missing API key)")

        # ── Cognitive Core Engine v7.0 ────────────────────────
        cognitive_orchestrator_engine = CognitiveCoreEngine(model_bridge=mco_bridge)
        mco_set_cognitive_engine(cognitive_orchestrator_engine)
        logger.info("Cognitive Core Engine v7.0 initialized — ensemble-only, no mode routing")

        # ── Multimodal Auditor ────────────────────────────────
        multimodal_auditor = MultimodalAuditor(
            cognitive_engine=cognitive_orchestrator_engine,
            redis_client=redis_client,
            cognitive_rag=cognitive_rag,
        )
        logger.info("Multimodal Capability Auditor initialized")

        # Background daemon (starts paused — activate via API)
        mco_daemon = BackgroundDaemon(
            cognitive_gateway=mco_orchestrator.cognitive_gateway,
            knowledge_engine=mco_orchestrator.knowledge_engine,
            session_engine=mco_orchestrator.session_engine,
            interval=300,
        )
        mco_set_daemon(mco_daemon)
        logger.info("Meta-Cognitive Orchestrator initialized")
    except Exception as e:
        logger.warning(f"MCO init failed (non-fatal): {e}")

    # ── Battle Platform v2 — ELO, Dataset, Monitoring ────────
    try:
        from ranking.elo_engine import get_elo_engine
        from evaluation.dataset import get_evaluation_dataset
        from evaluation.benchmark_pipeline import get_benchmark_pipeline
        from monitoring.ops_dashboard import get_ops_dashboard
        from evaluation.company_pipeline import get_company_pipeline
        from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY, get_tiered_models_for_debate

        get_elo_engine()         # seed ELO entries for all registry models
        get_evaluation_dataset() # create dataset file if absent
        get_benchmark_pipeline() # load historical reports
        get_ops_dashboard()      # start in-process ring buffer
        get_company_pipeline()   # load persisted company jobs

        n_models = len(COGNITIVE_MODEL_REGISTRY)
        tier_preview = get_tiered_models_for_debate("general", 6)
        logger.info(
            "Battle Platform v2 initialized — %d models in registry, "
            "default debate tier selection: %s",
            n_models, tier_preview,
        )
    except Exception as e:
        logger.warning(f"Battle Platform v2 init non-fatal: {e}")

    # ── Omega Kernel (uses MCO bridge if available, else legacy client) ─
    omega_kernel = OmegaCognitiveKernel(
        sigma_orchestrator=orchestrator,
        knowledge_learner=knowledge_learner,
        cloud_client=mco_bridge,
    )
    omega_kernel.knowledge_learner = knowledge_learner

    logger.info("Sentinel-E v5.0 online. All systems initialized (with optimization layer + MCO).")
    yield
    # Cleanup MCO
    if mco_orchestrator:
        await mco_orchestrator.close()
    if mco_daemon and mco_daemon.is_running:
        mco_daemon.stop()
    logger.info("Shutting down Sentinel-E v5.0...")


app = FastAPI(
    title="Sentinel-E API",
    version="5.0.0",
    lifespan=lifespan,
    docs_url=None if ENV == "production" else "/docs",
    redoc_url=None if ENV == "production" else "/redoc",
    openapi_url=None if ENV == "production" else "/openapi.json",
)

# ── Timeout Middleware ───────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Prevent requests from running indefinitely."""

    TIMEOUT_SECONDS = 180  # 3 minutes for ensemble operations

    async def dispatch(self, request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out. The ensemble analysis took too long. Please try a simpler query or fewer models."},
            )

# ── Middleware Stack (order matters: outermost first) ────────
app.add_middleware(TimeoutMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(InputValidationMiddleware)

# ── CORS (FIXED FOR VERCEL + RENDER) ─────────────────────────
origins = [
    "https://sentinel-e.vercel.app",  # ✅ your frontend
    "http://localhost:3000",          # optional (dev)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Meta-Cognitive Orchestrator Router ──────────────────────
app.include_router(mco_router)

# ── Battle Platform v2 Router ────────────────────────────────
app.include_router(battle_router)

# ── Admin Routes ──────────────────────────────────────────────
app.include_router(admin_router)
app.include_router(workflow_admin_router)

# ── Standard Mode Router (POST /chat/{model_id}) ──────────────
app.include_router(chat_router)


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def _evict_sessions():
    """
    DEPRECATED: Session eviction now handled by SessionCacheManager.
    This function is kept for backward compatibility but is no longer needed.
    SessionCacheManager automatically evicts oldest sessions when max capacity reached.
    """
    pass  # No-op; SessionCacheManager handles this safely


async def _persist_session(chat_id: str, kernel: OmegaCognitiveKernel, memory: MemoryEngine):
    """Persist session to SQLite (primary) with Redis as optional best-effort mirror.
    
    ✅ Includes ownership metadata for security validation on restore.
    """
    try:
        session_data = {
            "owner_user_id": getattr(kernel, '_owner_user_id', None),  # ✅ Store owner
            "omega": kernel.serialize_session(),
            "memory": memory.serialize(),
        }
        payload = json.dumps(session_data, default=str)
        await asyncio.to_thread(_sqlite_write_session, chat_id, payload)

        # Legacy mirror (optional)
        await redis_client.setex(
            f"session:{chat_id}",
            settings.REDIS_SESSION_TTL,
            payload,
        )
        logger.debug(f"✓ Persisted session {chat_id} for user {session_data.get('owner_user_id')}")
    except Exception as e:
        logger.warning(f"Session persist failed: {e}")


async def _restore_session(chat_id: str, user_id: str = ""):
    """Restore session from SQLite first, then Redis fallback for backward compatibility.
    
    ✅ Validates user ownership before restoring.
    """
    try:
        cached = await asyncio.to_thread(_sqlite_read_session, chat_id)
        if not cached:
            cached = await redis_client.get(f"session:{chat_id}")

        if not cached:
            return None, None

        data = json.loads(cached)
        
        # ✅ Verify ownership metadata
        owner_user_id = data.get("owner_user_id")
        if owner_user_id and user_id and owner_user_id != user_id:
            logger.warning(f"🔒 SECURITY: Session restore ownership mismatch - chat_id={chat_id}, attempted_by={user_id}, owner={owner_user_id}")
            raise HTTPException(status_code=403, detail="Unauthorized: Cannot restore session owned by another user")
        
        kernel = OmegaCognitiveKernel.restore_from_session(
            data.get("omega", {}),
            sigma_orchestrator=orchestrator,
            knowledge_learner=knowledge_learner,
            cloud_client=mco_bridge,
        )
        # ✅ Set owner on restored kernel
        kernel._owner_user_id = user_id or owner_user_id
        
        memory = MemoryEngine.deserialize(data.get("memory", {}))
        memory.user_id = user_id  # ✅ Update to current user
        
        return kernel, memory
    except HTTPException:
        raise  # Re-raise authorization errors
    except Exception as e:
        logger.warning(f"Session restore failed: {e}")
    return None, None


async def _get_session(chat_id: str, user_id: str = ""):
    """Get or create session pair with user ownership validation."""
    global omega_sessions, memory_sessions

    # ✅ Check cache ownership - verify user owns this session
    if chat_id in omega_sessions:
        cached_kernel = omega_sessions[chat_id]
        cached_memory = memory_sessions.get(chat_id, None)
        
        # Verify ownership if metadata exists
        if hasattr(cached_kernel, '_owner_user_id') and cached_kernel._owner_user_id:
            if cached_kernel._owner_user_id != user_id:
                logger.warning(f"🔒 SECURITY: Attempted unauthorized session access - chat_id={chat_id}, attempted_by={user_id}, owner={cached_kernel._owner_user_id}")
                raise HTTPException(status_code=403, detail="Unauthorized: This session belongs to another user")
        
        # Return cached session if ownership verified
        if not cached_memory:
            cached_memory = MemoryEngine(user_id=user_id)
            memory_sessions[chat_id] = cached_memory
        return cached_kernel, cached_memory

    # Try persisted cache (SQLite first, Redis fallback) with user validation
    kernel, memory = await _restore_session(chat_id, user_id)
    if kernel:
        omega_sessions[chat_id] = kernel
        memory_sessions[chat_id] = memory
        logger.debug(f"✓ Restored session {chat_id} for user {user_id}")
        return kernel, memory

    # Create new session with ownership metadata
    # Note: Session eviction is now handled automatically by SessionCacheManager (LRU+TTL)
    kernel = OmegaCognitiveKernel(
        sigma_orchestrator=orchestrator,
        knowledge_learner=knowledge_learner,
        cloud_client=mco_bridge,
    )
    # ✅ FIX #5: Store ownership metadata for session hijacking prevention
    kernel._owner_user_id = user_id
    kernel._session_id = chat_id  # Session ownership validation requires this
    
    memory = MemoryEngine(user_id=user_id)
    omega_sessions[chat_id] = kernel
    memory_sessions[chat_id] = memory
    logger.debug(f"✓ Created new session {chat_id} for user {user_id}")
    return kernel, memory


# ============================================================
# HEALTH & STATUS
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Sentinel-E",
        "version": "5.0.0",
    }


@app.get("/health")
async def health_check():
    """Production health check."""
    health = {
        "status": "healthy",
        "version": "5.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if redis_client:
            await redis_client.ping()
            if getattr(redis_client, '_is_stub', False):
                health["redis"] = "in_memory_fallback"
            else:
                health["redis"] = "connected"
        else:
            health["redis"] = "not_configured"
    except Exception:
        health["redis"] = "disconnected"
    return health


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Prevent noisy 404s for automatic browser favicon requests."""
    return Response(status_code=204)


# ============================================================
# IMAGE UPLOAD HELPER
# ============================================================

async def _read_upload_as_b64(file) -> tuple:
    """Read UploadFile to base64 with size validation for 512MB Render safety."""
    from sentinel.schemas import MAX_IMAGE_BYTES
    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_IMAGE_BYTES // (1024*1024)}MB limit.",
        )
    mime = file.content_type or "image/png"
    return base64.b64encode(contents).decode("utf-8"), mime


# ============================================================
# MEMORY EXTRACTION
# ============================================================

async def _extract_memory_bg(user_id: str, user_text: str):
    """Asynchronously extract user preferences/facts and save to DB."""
    try:
        from database.connection import get_db
        from database.crud import add_user_memory
        import json
        import re
        
        prompt = (
            "Analyze the following user input and extract any explicit personal facts or preferences. "
            "Return only valid JSON in the format: {\"key\": \"value\", \"confidence\": 90}. "
            "If no clear fact or preference exists, return an empty JSON object {}. "
            f"Input: {user_text}"
        )
        
        # Use a fast model for extraction
        fast_model = "llama31-8b"
        res = await mco_bridge.call_model(
            fast_model, 
            prompt, 
            "You are a memory extractor. Output only raw JSON. No markdown, no tags."
        )
        
        # Clean potential markdown
        res = re.sub(r'```json\n|\n```|```', '', res).strip()
        data = json.loads(res)
        
        if data and "key" in data and "value" in data:
            async for db in get_db():
                await add_user_memory(
                    db, 
                    user_id, 
                    str(data["key"]).lower().replace(' ', '_'), 
                    str(data["value"]), 
                    int(data.get("confidence", 75))
                )
                break
    except Exception as e:
        logger.debug(f"Background memory extraction skipped/failed: {e}")


async def _evolve_memory_graph_bg(user_id: str, user_text: str, assistant_text: str = ""):
    """Lightweight adaptive memory evolution: topics, preferences, corrections."""
    try:
        from database.connection import get_db
        from database.crud import add_user_memory

        def _topics(text: str):
            stop = {
                "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
                "this", "that", "it", "as", "at", "by", "from", "i", "you", "we", "they",
            }
            parts = [w.strip(".,:;!?()[]{}\"'").lower() for w in str(text or "").split()]
            return [w for w in parts if len(w) > 2 and w not in stop][:10]

        merged_text = f"{user_text}\n{assistant_text}".strip()
        topics = _topics(merged_text)

        preference_pairs = []
        lowered = str(user_text or "").lower()
        if "please" in lowered and ("concise" in lowered or "brief" in lowered):
            preference_pairs.append(("response_style", "concise"))
        if "step by step" in lowered:
            preference_pairs.append(("response_format", "step_by_step"))
        if "instead" in lowered or "correct" in lowered:
            preference_pairs.append(("correction_pattern", str(user_text)[:200]))

        async for db in get_db():
            for topic in topics:
                await add_user_memory(
                    db,
                    user_id,
                    key=f"topic:{topic}",
                    value=topic,
                    confidence=70,
                    metadata_json={"type": "topic", "source": "adaptive_learning"},
                )

            for k, v in preference_pairs:
                await add_user_memory(
                    db,
                    user_id,
                    key=f"pref:{k}",
                    value=v,
                    confidence=80,
                    metadata_json={"type": "preference", "source": "adaptive_learning"},
                )
            break
    except Exception as e:
        logger.debug(f"Adaptive memory evolution skipped/failed: {e}")

SAFE_FALLBACK_ANSWER = (
    "I’m here and ready to help. I hit a temporary processing issue, "
    "so I’m returning a safe response while preserving your session history. "
    "Please retry your request."
)


def _dedupe_and_cap_history(history: list, limit: int) -> list:
    """Deterministic history cleanup: drop empties/dupes and cap size."""
    seen = set()
    cleaned = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        key = (role, content)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"role": role, "content": content})
    return cleaned[-max(limit, 1):]


async def _safe_add_message(
    db: AsyncSession,
    chat_id,
    user_id: str,
    role: str,
    content: str,
    **kwargs,
):
    """Best-effort persistence wrapper: never raises outward."""
    try:
        return await add_message(db, chat_id, user_id, role, content, **kwargs)
    except Exception as e:
        logger.warning(f"Safe add_message fallback triggered for chat {chat_id}: {e}")
        return None


def _build_safe_api_payload(chat_id: str, mode: str, sub_mode: Optional[str], message: str, error_code: str):
    safe_text = sanitize_output(message)
    return {
        "chat_id": chat_id,
        "mode": mode,
        "sub_mode": sub_mode,
        "formatted_output": safe_text,
        "data": {"priority_answer": safe_text},
        "confidence": 0.2,
        "session_state": {},
        "boundary_result": {"risk_level": "LOW", "severity_score": 5},
        "omega_metadata": {
            "fallback": True,
            "error": error_code,
        },
    }


# ── Shared core — called by /api/run route and all form shims ────────────────
async def _run_sentinel_core(
    request: "SentinelRequest",
    db: AsyncSession,
    user_id: str,
    frontend_context: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
):
    """
    Core execution logic for all Sentinel run endpoints.
    Extracted so both the JSON route and form-data shims can call it.
    """
    try:
        if not user_id:
            return api_error("Authentication required", status_code=401)
        logger.info("auth user_id=%s", user_id)
        if not orchestrator:
            return api_error("System initializing. Please retry.", status_code=503)

        # ── Input Validation ─────────────────────────────────────
        if len(request.text) > settings.MAX_INPUT_LENGTH:
            return api_error("Input too long.", status_code=400)

        request.rounds = min(request.rounds, settings.MAX_ROUNDS)

        # ── Prompt Firewall ──────────────────────────────────────
        firewall = get_firewall()
        verdict = firewall.analyze(request.text)

        if verdict.blocked:
            logger.warning(f"Firewall blocked input from {user_id}: {verdict.violations}")
            return api_error("Your input could not be processed. Please rephrase.", status_code=400)
        
        effective_text = verdict.sanitized_text or request.text

        # ── Chat Resolution ──────────────────────────────────────
        chat = None
        if request.chat_id:
            chat = await get_chat(db, request.chat_id, user_id=user_id)
            if not chat:
                return api_error("Chat not found or unauthorized access", status_code=403)
        
        if not chat:
            chat_name = generate_chat_name(effective_text, request.mode)
            chat = await create_chat(db, chat_name, request.mode, user_id=user_id)

        await _safe_add_message(
            db,
            chat.id,
            user_id,
            "user",
            effective_text,
            image_b64=request.image_b64,
            image_mime=request.image_mime,
        )

        # ── Session & Memory ─────────────────────────────────────
        kernel, memory = await _get_session(str(chat.id), user_id)
        
        try:
            await validate_kernel_before_use(kernel, user_id, str(chat.id))
        except ValueError as e:
            return api_error(f"Session validation failed: {e}", status_code=403)
    
        memory.add_message("user", effective_text)

        # Trigger optional memory updates without blocking the response.
        if background_tasks is not None:
            background_tasks.add_task(safe_memory_update, user_id, effective_text, db)
        else:
            asyncio.create_task(safe_memory_update(user_id, effective_text, None))
    
        # ── Conversation History ─────────────────────────────────
        history = []
        try:
            # ✅ Pass user_id to verify ownership
            stored = await get_chat_messages(db, chat.id, user_id=user_id)
            if len(stored) > 1:
                for msg in stored[-settings.SHORT_TERM_MEMORY_SIZE:]:
                    history.append({"role": msg.role, "content": msg.content})
        except Exception as e:
            logger.warning(f"History retrieval failed: {e}")
    
        # ── Frontend Context (Sanitized) ─────────────────────────
        if frontend_context:
            try:
                ctx = json.loads(frontend_context)
                safe_ctx = firewall.validate_context_injection(ctx)
                st = safe_ctx.get("shortTerm", {})
    
                if st.get("isFollowUp") and st.get("resolvedQuery"):
                    effective_text = st["resolvedQuery"]
    
                active_entity = st.get("activeEntity")
                active_topic = st.get("activeTopic")
                if active_entity or active_topic:
                    ctx_hint = "Context: "
                    if active_topic:
                        ctx_hint += f"topic is '{active_topic}'"
                    if active_entity:
                        ctx_hint += f"{', ' if active_topic else ''}subject is '{active_entity}'"
                    history.insert(0, {"role": "system", "content": ctx_hint})
            except Exception as e:
                logger.debug(f"Context injection skipped: {e}")
    
        # ── Memory Context Injection ─────────────────────────────
        memory_ctx = memory.build_prompt_context()
        if memory_ctx:
            history.insert(0, {"role": "system", "content": memory_ctx})
    
        # ── Context Builder Injection ─────────────────────────────
        try:
            context_bundle = await safe_build_context(db, user_id, chat.id, effective_text)

            history = context_bundle.get("recent_history", [])
            system_instructions = context_bundle.get("system_instructions", "")
            
            if system_instructions:
                history.insert(0, {"role": "system", "content": system_instructions})
        except Exception as e:
            logger.warning(f"Context builder failed: {e} - falling back to recent history")
            # history already contains recent messages from line 791
            context_bundle = {"context_str": "", "system_instructions": "", "recent_history": history}

        history = _dedupe_and_cap_history(history, settings.SHORT_TERM_MEMORY_SIZE)

        # ── Agentic Layer ────────────────────────────────────────
        if getattr(request, "agentic", False):
            try:
                from core.agentic_orchestrator import AgenticOrchestrator
                orchestrator_agent = AgenticOrchestrator(mco_bridge)
                agent_res = await orchestrator_agent.run(user_id, str(chat.id), effective_text, context_bundle.get("context_str", ""))
                
                formatted_output = sanitize_output(agent_res["formatted_output"])
                await _safe_add_message(db, chat.id, user_id, "assistant", formatted_output)
                return api_success({
                    "chat_id": str(chat.id),
                    "formatted_output": formatted_output,
                    "agent_logs": agent_res.get("step_logs", [])
                })
            except Exception as e:
                logger.error(f"Agentic execution failed: {e}")
                # Fall through to standard pipeline
    
        # ── Cognitive RAG ────────────────────────────────────────
        rag_result = None
        if cognitive_rag:
            try:
                rag_result = await cognitive_rag.process(effective_text)
                if rag_result and rag_result.retrieval_executed and rag_result.sources:
                    rag_context = "External evidence:\n" + "\n".join(
                        [f"- [{s.title}]({s.url}): {s.content[:200]}" for s in rag_result.sources[:3]]
                    )
                    history.append({"role": "system", "content": rag_context})
            except Exception as e:
                logger.warning(f"RAG failed: {e}")
    
        # ── Mode Resolution ──────────────────────────────────────
        omega_mode = request.mode
        mode_map = {"conversational": "standard", "forensic": "standard", "experimental": "research"}
        omega_mode = mode_map.get(omega_mode, omega_mode)
    
        sub_mode = getattr(request, "sub_mode", None) or omega_mode
        kill = getattr(request, "kill", False)
        role_map = getattr(request, "role_map", None) or {}
        cache_mode_key = f"{omega_mode}:{sub_mode or 'standard'}"
    
        # ── Optimization: Observability Tracing ───────────────────
        obs_hub = get_observability_hub()
        request_id = str(uuid_lib.uuid4().hex[:12])
        tracer = obs_hub.start_request(session_id=str(chat.id), request_id=request_id)
        tracer.start_span("total")
    
        # ── Optimization: Response Cache Check ────────────────────
        cache = get_response_cache()
        cache_result = cache.lookup(effective_text, cache_mode_key)
        if cache_result.hit:
            cached_response = cache_result.response or {}
            cache_latency = tracer.end_span("total")
            tier_name = {1: "exact", 2: "lexical", 3: "semantic"}.get(cache_result.tier, "unknown")
            tracer.record_cache_hit(tier=tier_name, latency_ms=cache_latency)
            summary = tracer.finalize()
            obs_hub.record(summary)
    
            # Return cached response (preserving full response contract)
            return {
                "chat_id": str(chat.id),
                "chat_name": cached_response.get("chat_name", ""),
                "mode": cached_response.get("mode", omega_mode),
                "sub_mode": cached_response.get("sub_mode", sub_mode),
                "formatted_output": cached_response.get("formatted_output", ""),
                "data": {"priority_answer": cached_response.get("formatted_output", "")},
                "confidence": cached_response.get("confidence", 0.5),
                "session_state": cached_response.get("session_state", {}),
                "boundary_result": cached_response.get("boundary_result", {}),
                "omega_metadata": {**cached_response.get("omega_metadata", {}), "cache_hit": True, "cache_tier": tier_name},
            }
        tracer.record_cache_miss()
    
        # ── Optimization: Cost Governance ─────────────────────────
        governor = get_cost_governor()
        tier = "premium" if omega_mode in ("research", "experimental") else "standard"
        gov_decision = governor.check_budget(str(chat.id), requested_tier=tier)
        if not gov_decision.allowed:
            from optimization.observability import ObservabilityEvent, EventType
            tracer.record_event(ObservabilityEvent(event_type=EventType.BUDGET_EXCEEDED))
            logger.warning(f"Budget exceeded for chat {chat.id}: {gov_decision.reason}")
            raise HTTPException(status_code=429, detail=f"Session budget exceeded. {gov_decision.reason}")
    
        # Apply cost governor model recommendation
        if gov_decision.downgraded and gov_decision.recommended_model:
            logger.info(f"Cost governor downgraded model for chat {chat.id}: {gov_decision.recommended_model}")
            # Override selected_model on the request so downstream routing uses the cheaper model
            if hasattr(request, 'selected_model'):
                request.selected_model = gov_decision.recommended_model
    
        # ── Optimization: Token Optimization ──────────────────────
        token_optimizer = get_token_optimizer()
    
        # Separate system messages and user/assistant history
        system_msgs = [m for m in history if m.get("role") == "system"]
        conv_history = [m for m in history if m.get("role") != "system"]
        system_prompt = "\n".join(m.get("content", "") for m in system_msgs)
    
        opt_result = token_optimizer.optimize(
            query=effective_text,
            system_prompt=system_prompt,
            history=conv_history,
            context_window=settings.TOKEN_BUDGET_PER_REQUEST,
        )
        depth_assessment = opt_result.get("depth_assessment")
        if opt_result.get("compression_applied") or opt_result.get("deduped_history_count", 0) < len(conv_history):
            original_tokens = sum(len(m.get("content", "")) // 4 for m in history)
            opt_system = opt_result.get("system_prompt", system_prompt)
            opt_history_list = opt_result.get("history", conv_history)
            optimized_tokens = len(opt_system) // 4 + sum(len(m.get("content", "")) // 4 for m in opt_history_list)
            if original_tokens > optimized_tokens:
                tracer.record_token_optimization(
                    original_tokens=original_tokens,
                    optimized_tokens=optimized_tokens,
                )
            # Rebuild history with optimized system prompt + conversation
            history = []
            if opt_system:
                history.append({"role": "system", "content": opt_system})
            history.extend(opt_history_list)
    
        # ══════════════════════════════════════════════════════════
        # QUERY ROUTING — Decide execution path before running
        # ══════════════════════════════════════════════════════════
        from core.query_router import route_query, ExecutionPath
        from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
    
        routing_decision = route_query(
            query=effective_text,
            mode=omega_mode,
            sub_mode=sub_mode,
            selected_model=getattr(request, "selected_model", None),
            model_registry=COGNITIVE_MODEL_REGISTRY,
            image_b64=request.image_b64,
        )
        logger.info(
            f"Routing decision: path={routing_decision.path.value}, "
            f"reason={routing_decision.reason}, "
            f"complexity={routing_decision.query_complexity}"
        )
    
        # ══════════════════════════════════════════════════════════
        # PATH: SINGLE MODEL CHAT — bypass ensemble entirely
        # ══════════════════════════════════════════════════════════
        if routing_decision.path == ExecutionPath.SINGLE_MODEL and routing_decision.selected_model:
            tracer.start_span("kernel")
            single_model_id = routing_decision.selected_model
            try:
                spec = COGNITIVE_MODEL_REGISTRY.get(single_model_id)
                if not spec or not spec.enabled:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{single_model_id}' is not available.",
                    )
    
                raw_output = await mco_bridge.call_model(
                    single_model_id, effective_text, "",
                    image_b64=request.image_b64,
                    image_mime=request.image_mime,
                )
                kernel_latency = tracer.end_span("kernel")
    
                # Check for error responses from the model
                if raw_output.startswith("Error:"):
                    logger.error(f"Model '{single_model_id}' returned error: {raw_output}")
                    raise HTTPException(
                        status_code=502,
                        detail="Provider unavailable. Please try again or select a different model.",
                    )
    
                formatted_output = sanitize_output(raw_output)
                confidence = 0.8  # Single model — no ensemble calibration
    
                await _safe_add_message(db, chat.id, user_id, "assistant", formatted_output)
                memory.add_message("assistant", formatted_output)
                await _persist_session(str(chat.id), kernel, memory)
    
                omega_metadata = {
                    "version": "6.1.0",
                    "mode": "single_model",
                    "sub_mode": None,
                    "selected_model": single_model_id,
                    "model_name": spec.name,
                    "provider": spec.provider,
                    "model_type": getattr(spec, "model_type", "external"),
                    "confidence": confidence,
                    "routing": {
                        "path": routing_decision.path.value,
                        "reason": routing_decision.reason,
                        "query_complexity": routing_decision.query_complexity,
                    },
                }
    
                await update_chat_metadata(
                    db, chat.id,
                    priority_answer=formatted_output,
                    machine_metadata=omega_metadata,
                    rounds=0,
                )
    
                try:
                    total_latency = tracer.end_span("total")
                except Exception:
                    total_latency = kernel_latency
    
                return {
                    "chat_id": str(chat.id),
                    "mode": "single_model",
                    "sub_mode": None,
                    "formatted_output": formatted_output,
                    "data": {"priority_answer": formatted_output},
                    "confidence": confidence,
                    "session_state": {},
                    "boundary_result": {"risk_level": "LOW", "severity_score": 20},
                    "omega_metadata": omega_metadata,
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Single model chat failed for '{single_model_id}': {e}")
                raise HTTPException(
                    status_code=502,
                    detail="Provider unavailable. Please try again or select a different model.",
                )
    
        # ══════════════════════════════════════════════════════════
        # PATH: TRIVIAL QUERY — use fastest available model, skip debate
        # ══════════════════════════════════════════════════════════
        if routing_decision.skip_debate and routing_decision.path == ExecutionPath.STANDARD:
            tracer.start_span("kernel")
            try:
                # Use the fastest model for trivial queries
                fast_model = "llama31-8b"
                spec = COGNITIVE_MODEL_REGISTRY.get(fast_model)
                if not spec or not spec.enabled:
                    # Fallback to any enabled model
                    fast_model = next(
                        (k for k, s in COGNITIVE_MODEL_REGISTRY.items() if s.enabled),
                        None,
                    )
                    if not fast_model:
                        raise HTTPException(status_code=503, detail="No models available.")
                    spec = COGNITIVE_MODEL_REGISTRY[fast_model]
    
                raw_output = await mco_bridge.call_model(
                    fast_model, effective_text, "",
                    image_b64=request.image_b64,
                    image_mime=request.image_mime,
                )
                kernel_latency = tracer.end_span("kernel")
    
                if raw_output.startswith("Error:"):
                    # For trivial queries, try one fallback before failing
                    fallback_model = next(
                        (k for k, s in COGNITIVE_MODEL_REGISTRY.items()
                         if s.enabled and k != fast_model),
                        None,
                    )
                    if fallback_model:
                        raw_output = await mco_bridge.call_model(
                            fallback_model, effective_text, "",
                            image_b64=request.image_b64,
                            image_mime=request.image_mime,
                        )
                        if raw_output.startswith("Error:"):
                            logger.error(f"Fallback model '{fallback_model}' also failed: {raw_output}")
                            raise HTTPException(
                                status_code=502,
                                detail="Provider unavailable. Please try again.",
                            )
                        spec = COGNITIVE_MODEL_REGISTRY[fallback_model]
                        fast_model = fallback_model
                    else:
                        logger.error(f"No fallback available. Original error: {raw_output}")
                        raise HTTPException(
                            status_code=502,
                            detail="Provider unavailable. Please try again.",
                        )
    
                formatted_output = sanitize_output(raw_output)
                confidence = 0.7
    
                await _safe_add_message(db, chat.id, user_id, "assistant", formatted_output)
                memory.add_message("assistant", formatted_output)
                await _persist_session(str(chat.id), kernel, memory)
    
                omega_metadata = {
                    "version": "6.1.0",
                    "mode": omega_mode,
                    "sub_mode": sub_mode,
                    "model_used": fast_model,
                    "model_name": spec.name,
                    "confidence": confidence,
                    "routing": {
                        "path": routing_decision.path.value,
                        "reason": routing_decision.reason,
                        "query_complexity": routing_decision.query_complexity,
                        "debate_skipped": True,
                    },
                }
    
                await update_chat_metadata(
                    db, chat.id,
                    priority_answer=formatted_output,
                    machine_metadata=omega_metadata,
                    rounds=0,
                )
    
                try:
                    total_latency = tracer.end_span("total")
                except Exception:
                    total_latency = kernel_latency
    
                return {
                    "chat_id": str(chat.id),
                    "mode": omega_mode,
                    "sub_mode": sub_mode,
                    "formatted_output": formatted_output,
                    "data": {"priority_answer": formatted_output},
                    "confidence": confidence,
                    "session_state": {},
                    "boundary_result": {"risk_level": "LOW", "severity_score": 15},
                    "omega_metadata": omega_metadata,
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Fast path failed, falling through to ensemble: {e}")
                # Fall through to ensemble if fast path fails
    
        # ══════════════════════════════════════════════════════════
        # COGNITIVE ENSEMBLE — For analytical queries and debate mode
        # ══════════════════════════════════════════════════════════
        use_ensemble = cognitive_orchestrator_engine is not None
    
        # ── Multimodal Capability Audit (pre-execution) ──────────
        audit_report = None
        if use_ensemble and multimodal_auditor:
            try:
                from core.multimodal_auditor import phase1_inspect_input, phase2_capability_check, phase3_model_availability_audit
                inspection = phase1_inspect_input(
                    query=effective_text,
                    image_b64=request.image_b64,
                    image_mime=request.image_mime,
                )
                capability = phase2_capability_check(inspection)
                _, disabled_reports = phase3_model_availability_audit()
    
                audit_report = {
                    "input_type": inspection.input_type.value,
                    "multimodal_required": inspection.multimodal_required,
                    "required_capabilities": capability.required_capabilities,
                    "preferred_models": capability.preferred_models,
                    "disabled_models": [
                        {"reason": d.reason, "provider": d.provider, "env_var": d.required_env_var}
                        for d in disabled_reports
                    ],
                }
                logger.info(
                    f"Pre-execution audit: type={inspection.input_type.value}, "
                    f"multimodal={inspection.multimodal_required}"
                )
            except Exception as audit_err:
                logger.warning(f"Pre-execution audit failed (non-fatal): {audit_err}")
    
        if use_ensemble:
            tracer.start_span("kernel")
            try:
                from core.ensemble_schemas import EnsembleFailure
                ensemble_response = await cognitive_orchestrator_engine.process(
                    query=effective_text,
                    chat_id=str(chat.id),
                    rounds=max(request.rounds, 3),
                    history=history,
                    image_b64=request.image_b64,
                    image_mime=request.image_mime,
                )
            except EnsembleFailure as ef:
                logger.error(f"Ensemble hard failure: {ef}")
                ensemble_response = ef.to_response()
                cognitive_orchestrator_engine_failed = False  # still return structured error
            except Exception as ens_err:
                logger.error(f"Ensemble engine crashed: {ens_err} — falling back to legacy kernel")
                cognitive_orchestrator_engine_failed = True
            else:
                cognitive_orchestrator_engine_failed = False
    
            if not cognitive_orchestrator_engine_failed:
                kernel_latency = tracer.end_span("kernel")
                payload = ensemble_response.to_frontend_payload()
    
                formatted_output = sanitize_output(ensemble_response.formatted_output)
                if rag_result and rag_result.retrieval_executed:
                    if rag_result.no_sources_found:
                        formatted_output += "\n\n*No verified external sources found for this query.*"
                    elif rag_result.citations_text:
                        formatted_output += "\n\n" + rag_result.citations_text
                    payload["formatted_output"] = formatted_output
                    payload["final_answer"] = formatted_output
    
                confidence = ensemble_response.confidence.final_confidence
                ens_entropy = ensemble_response.ensemble_metrics.disagreement_entropy
                ens_fragility = ensemble_response.ensemble_metrics.fragility_score
                ens_debate_rounds = ensemble_response.debate_result.total_rounds
    
                omega_metadata = payload.get("omega_metadata", {})
                omega_metadata.update({
                    "version": "7.1.0-cognitive",
                    "mode": omega_mode,
                    "sub_mode": sub_mode,
                    "confidence": confidence,
                    "entropy": ens_entropy,
                    "fragility": ens_fragility,
                    "fragility_index": ens_fragility,
                    "ensemble_metrics": payload.get("ensemble_metrics", {}),
                    "debate_result": payload.get("debate_result", {}),
                    "debate_rounds": payload.get("debate_rounds", []),
                    "model_outputs": payload.get("model_outputs", []),
                    "agreement_matrix": payload.get("agreement_matrix", {}),
                    "drift_metrics": payload.get("drift_metrics", {}),
                    "tactical_map": payload.get("tactical_map", []),
                    "confidence_graph": payload.get("confidence_graph", payload.get("calibrated_confidence", {})),
                    "session_intelligence": payload.get("session_intelligence", {}),
                    "session_analytics": payload.get("session_analytics", {}),
                    "model_status": payload.get("model_status", []),
                    "reasoning_trace": {
                        "engine": "CognitiveCoreEngine",
                        "pipeline": "cognitive_v7",
                        "models_executed": ensemble_response.models_executed,
                        "models_succeeded": ensemble_response.models_succeeded,
                        "models_failed": ensemble_response.models_failed,
                        "debate_rounds": ens_debate_rounds,
                    },
                    "boundary_result": {
                        "risk_level": (
                            "LOW" if confidence > 0.7
                            else "MEDIUM" if confidence > 0.4
                            else "HIGH"
                        ),
                        "severity_score": int((1 - confidence) * 100),
                        "explanation": (
                            f"Ensemble confidence from {ensemble_response.models_executed} models, "
                            f"{ens_debate_rounds} debate rounds"
                        ),
                    },
                })
    
                # Inject multimodal audit report into metadata
                if audit_report:
                    omega_metadata["multimodal_audit"] = audit_report
    
                if rag_result and rag_result.retrieval_executed:
                    omega_metadata["rag_result"] = {
                        "executed": True,
                        "source_count": rag_result.source_count,
                        "average_reliability": rag_result.average_reliability,
                        "contradictions": len(rag_result.contradictions),
                        "no_sources": rag_result.no_sources_found,
                    }
    
                # ── Sub-Mode Pipeline Execution ──────────────────────
                # Wire Glass / Evidence / Synthesis pipelines so the
                # frontend views receive real analytical data.
                if sub_mode in ("glass", "evidence", "synthesis"):
                    try:
                        adapted_results, scoring_bd, div_metrics = _adapt_ensemble_for_pipelines(ensemble_response)
                        if scoring_bd:
                            winning = max(scoring_bd, key=lambda s: getattr(s, "final_score", 0.0)).model_name
                        elif ensemble_response.model_outputs:
                            winning = ensemble_response.model_outputs[0].model_name
                        else:
                            winning = "unknown"
    
                        if sub_mode == "glass":
                            omega_metadata["audit_result"] = build_glass_result(
                                all_results=adapted_results,
                                scoring_breakdown=scoring_bd,
                                divergence_metrics=div_metrics,
                                aggregated_answer=formatted_output,
                                winning_model=winning,
                                drift_score=ensemble_response.debate_result.drift_index,
                                volatility_score=ensemble_response.ensemble_metrics.fragility_score,
                            )
                        elif sub_mode == "evidence":
                            omega_metadata["forensic_result"] = await build_evidence_result(
                                query=effective_text,
                                all_results=adapted_results,
                                scoring_breakdown=scoring_bd,
                                aggregated_answer=formatted_output,
                                winning_model=winning,
                            )
                        elif sub_mode == "synthesis":
                            omega_metadata["synthesis_result"] = build_synthesis_result(
                                all_results=adapted_results,
                                scoring_breakdown=scoring_bd,
                                divergence_metrics=div_metrics,
                                aggregated_answer=formatted_output,
                                winning_model=winning,
                            )
                    except Exception as pipe_err:
                        logger.error(f"Sub-mode pipeline '{sub_mode}' failed: {pipe_err}", exc_info=True)
    
                await update_chat_metadata(
                    db, chat.id,
                    priority_answer=formatted_output,
                    machine_metadata=omega_metadata,
                    rounds=request.rounds,
                )
                memory.add_message("assistant", formatted_output)
                await _persist_session(str(chat.id), kernel, memory)
    
                # Update MCO session analytics with drift/rift metrics
                if mco_orchestrator and hasattr(mco_orchestrator, 'session_engine'):
                    try:
                        mco_orchestrator.session_engine.update_analytics(
                            session_id=str(chat.id),
                            mode=omega_mode,
                            drift_value=ensemble_response.debate_result.drift_index,
                            rift_value=ensemble_response.debate_result.rift_index,
                            disagreement_value=ensemble_response.ensemble_metrics.disagreement_entropy,
                        )
                        mco_orchestrator.session_engine.add_conversation_message(
                            session_id=str(chat.id),
                            role="assistant",
                            content=formatted_output[:500],
                            confidence=confidence,
                            latency_ms=kernel_latency,
                        )
                    except Exception:
                        pass
    
                try:
                    await redis_client.setex(
                        f"chat:{chat.id}:metadata",
                        settings.REDIS_SESSION_TTL,
                        json.dumps(omega_metadata, default=str),
                    )
                except Exception:
                    pass
    
                await _safe_add_message(db, chat.id, user_id, "assistant", formatted_output)
    
                response_payload = {
                    **payload,
                    "chat_id": str(chat.id),
                    "mode": omega_mode,
                    "sub_mode": sub_mode,
                    "formatted_output": formatted_output,
                    "confidence": confidence,
                    "entropy": ens_entropy,
                    "fragility": ens_fragility,
                    "fragility_index": ens_fragility,
                    "boundary_result": omega_metadata["boundary_result"],
                    "omega_metadata": omega_metadata,
                }
    
                try:
                    cache.store(effective_text, cache_mode_key, response_payload)
                except Exception:
                    pass
    
                try:
                    est_input_tokens = sum(len(m.get("content", "")) // 4 for m in history)
                    est_output_tokens = len(formatted_output) // 4
                    governor.record_usage(
                        session_id=str(chat.id),
                        model_id="ensemble",
                        input_tokens=est_input_tokens,
                        output_tokens=est_output_tokens,
                        latency_ms=kernel_latency,
                    )
                except Exception:
                    pass
    
                try:
                    total_latency = tracer.end_span("total")
                    tracer.record_model_call(
                        model_id="ensemble",
                        input_tokens=sum(len(m.get("content", "")) // 4 for m in history),
                        output_tokens=len(formatted_output) // 4,
                        latency_ms=kernel_latency,
                        cost_estimate=0.0,
                    )
                    summary = tracer.finalize()
                    obs_hub.record(summary)
                except Exception:
                    pass
    
                return response_payload
    
        # ══════════════════════════════════════════════════════════
        # LEGACY KERNEL PATH (fallback if ensemble unavailable)
        # ══════════════════════════════════════════════════════════
        config = ModeConfig.from_legacy(
            text=effective_text,
            mode=omega_mode,
            sub_mode=sub_mode,
            kill_switch=kill,
            enable_shadow=request.enable_shadow,
            rounds=request.rounds,
            chat_id=str(chat.id),
            history=history,
            role_map=role_map,
        )
    
        # ── Execute (Legacy) ─────────────────────────────────────
        tracer.start_span("kernel")
        try:
            result = await kernel.process(config)
            kernel_latency = tracer.end_span("kernel")
        except Exception as kernel_err:
            logger.error(f"Legacy kernel execution failed: {kernel_err}", exc_info=True)
            fallback_payload = _build_safe_api_payload(
                chat_id=str(chat.id),
                mode=omega_mode,
                sub_mode=sub_mode,
                message=SAFE_FALLBACK_ANSWER,
                error_code="kernel_execution_failed",
            )
            await _safe_add_message(db, chat.id, user_id, "assistant", fallback_payload["formatted_output"])
            return api_success(fallback_payload)
    
        # ── Extract & Build Response (Legacy) ────────────────────
        formatted_output = sanitize_output(result.get("formatted_output", ""))
        confidence = result.get("confidence", 0.5)
        session_state = result.get("session_state", {})
        reasoning_trace = result.get("reasoning_trace", {})
        boundary_result = result.get("boundary_result", {})
    
        # ── Inject RAG Citations ─────────────────────────────────
        if rag_result and rag_result.retrieval_executed:
            if rag_result.no_sources_found:
                formatted_output += "\n\n*No verified external sources found for this query.*"
            elif rag_result.citations_text:
                formatted_output += "\n\n" + rag_result.citations_text
    
        # ── Dynamic Analytics ────────────────────────────────────
        analytics = None
        if analytics_engine:
            model_outputs = []
            failed_model_ids = []
            if result.get("omega_metadata", {}).get("aggregation_result"):
                agg = result["omega_metadata"]["aggregation_result"]
                if isinstance(agg, dict):
                    for m in agg.get("model_outputs", []):
                        if isinstance(m, dict):
                            output_text = m.get("output", "")
                            if m.get("error") or m.get("status") == "failed":
                                failed_model_ids.append(m.get("model_id", "unknown"))
                                continue  # don't score failed models, but track them
                            if output_text:
                                model_outputs.append(output_text)
                        else:
                            text = str(m)
                            if text:
                                model_outputs.append(text)
    
            if model_outputs:
                analytics = analytics_engine.compute(
                    model_outputs=model_outputs,
                    evidence_sources=rag_result.source_count if rag_result else 0,
                    contradiction_count=rag_result.contradiction_count if rag_result else 0,
                    evidence_reliability=rag_result.average_reliability if rag_result else 0,
                )
                confidence = analytics.confidence  # Use dynamic confidence
    
        # ── Build Metadata ───────────────────────────────────────
        omega_metadata = {
            "version": "5.0.0",
            "mode": result.get("mode", omega_mode),
            "sub_mode": result.get("sub_mode", sub_mode),
            "confidence": confidence,
            "session_state": session_state,
            "reasoning_trace": reasoning_trace,
            "boundary_result": boundary_result,
        }
    
        # Mode-specific data
        for key in ["confidence_evolution", "fragility_index", "behavioral_risk",
                     "evidence_result", "stress_result", "confidence_components",
                     "debate_result"]:
            if result.get(key) is not None:
                omega_metadata[key] = result[key]
    
        # Dynamic analytics override
        if analytics:
            omega_metadata["confidence_components"] = analytics.confidence_components
            omega_metadata["boundary_result"] = {
                "risk_level": analytics.risk_level,
                "severity_score": int(analytics.boundary_risk * 100),
                "explanation": analytics.explanation,
                "risk_dimensions": analytics.boundary_components,
            }
    
        # Engine metadata passthrough
        if result.get("omega_metadata"):
            engine_meta = result["omega_metadata"]
            for key in ["aggregation_result", "forensic_result", "audit_result", "pipeline_steps"]:
                if engine_meta.get(key):
                    omega_metadata[key] = engine_meta[key]
    
        # RAG metadata
        if rag_result and rag_result.retrieval_executed:
            omega_metadata["rag_result"] = {
                "executed": True,
                "source_count": rag_result.source_count,
                "average_reliability": rag_result.average_reliability,
                "contradictions": len(rag_result.contradictions),
                "no_sources": rag_result.no_sources_found,
            }
    
        # ── Knowledge Learning ───────────────────────────────────
        if knowledge_learner and boundary_result:
            severity = boundary_result.get("severity_score", 0)
            if severity > 40:
                try:
                    knowledge_learner.record_boundary_violation(
                        model_name=omega_metadata.get("mode", "unknown"),
                        severity_score=severity,
                        severity_level=boundary_result.get("risk_level", "LOW"),
                        claim_type=boundary_result.get("claim_type", "unknown"),
                        run_id=str(chat.id),
                    )
                except Exception as e:
                    logger.warning(f"Knowledge learning failed: {e}")
    
        # ── Persist ──────────────────────────────────────────────
        await update_chat_metadata(
            db, chat.id,
            priority_answer=formatted_output,
            machine_metadata=omega_metadata,
            rounds=request.rounds,
        )
        memory.add_message("assistant", formatted_output)
        await _persist_session(str(chat.id), kernel, memory)
    
        try:
            await redis_client.setex(
                f"chat:{chat.id}:metadata",
                settings.REDIS_SESSION_TTL,
                json.dumps(omega_metadata, default=str),
            )
        except Exception:
            pass
    
        await _safe_add_message(db, chat.id, user_id, "assistant", formatted_output)
    
        # ── Rolling Summary Check ────────────────────────────────
        if memory.needs_summarization():
            try:
                summary_prompt = memory.generate_summary_prompt()
                if summary_prompt and mco_orchestrator and mco_orchestrator.cognitive_gateway:
                    from metacognitive.schemas import CognitiveGatewayInput
                    gw_input = CognitiveGatewayInput(
                        user_query=summary_prompt,
                        stabilized_context={},
                        knowledge_bundle=[],
                        session_summary={},
                    )
                    gw_output = await mco_orchestrator.cognitive_gateway.invoke_model(
                        "llama33-70b", gw_input
                    )
                    summary = gw_output.raw_output if gw_output.success else None
                    if summary and not summary.startswith("Error"):
                        memory.rolling_summary.add_summary(summary, settings.ROLLING_SUMMARY_INTERVAL)
                        logger.info(f"Rolling summary generated for chat {chat.id}")
            except Exception as e:
                logger.debug(f"Summary generation skipped: {e}")
    
        # ── Optimization: Cache Store & Observability ─────────────
        response_payload = {
            "chat_id": str(chat.id),
            "chat_name": result.get("chat_name", ""),
            "mode": result.get("mode", omega_mode),
            "sub_mode": result.get("sub_mode", sub_mode),
            "formatted_output": formatted_output,
            "data": {"priority_answer": formatted_output},
            "confidence": confidence,
            "session_state": session_state,
            "boundary_result": omega_metadata.get("boundary_result", boundary_result),
            "omega_metadata": omega_metadata,
        }
    
        # Store in response cache (non-blocking, best-effort)
        try:
            cache.store(effective_text, cache_mode_key, response_payload)
        except Exception:
            pass
    
        # Record usage in cost governor
        try:
            # Estimate tokens from history length + output length
            est_input_tokens = sum(len(m.get("content", "")) // 4 for m in history)
            est_output_tokens = len(formatted_output) // 4
            governor.record_usage(
                session_id=str(chat.id),
                model_id=result.get("omega_metadata", {}).get("primary_model", "groq-small"),
                input_tokens=est_input_tokens,
                output_tokens=est_output_tokens,
                latency_ms=kernel_latency,
            )
        except Exception:
            pass
    
        # Finalize observability
        return api_success(response_payload)

    except Exception as e:
        logger.error(f"CRITICAL ERROR in _run_sentinel_core: {e}", exc_info=True)
        fallback_chat = locals().get("chat")
        fallback_chat_id = str(fallback_chat.id) if fallback_chat else str(getattr(request, "chat_id", "") or "")
        fallback_mode = str(getattr(request, "mode", "standard") or "standard")
        fallback_sub_mode = getattr(request, "sub_mode", None)

        payload = _build_safe_api_payload(
            chat_id=fallback_chat_id,
            mode=fallback_mode,
            sub_mode=fallback_sub_mode,
            message=SAFE_FALLBACK_ANSWER,
            error_code="core_runtime_error",
        )
        if fallback_chat and user_id:
            await _safe_add_message(db, fallback_chat.id, user_id, "assistant", payload["formatted_output"])
        return api_success(payload)


# ============================================================
# MAIN EXECUTION ENDPOINT (JSON body)
# ============================================================

@app.post("/api/run")
async def run_sentinel(
    request: SentinelRequest,
    raw_request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    frontend_context: Optional[str] = None,
):
    """Main entry point for Sentinel execution (JSON body)."""
    user_id = await get_user_id(raw_request)
    return await _run_sentinel_core(request, db, user_id, frontend_context, background_tasks)



# ============================================================
# COMPRESSED PIPELINE ENDPOINT (Sentinel Sigma — LangGraph)
# ============================================================

@app.post("/api/compressed")
async def run_compressed_api(
    request: Request,
    body: SentinelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compressed reasoning pipeline."""
    try:
        user_id = await get_user_id(request)
        if not user_id: return api_error("Auth required", status_code=401)

        firewall = get_firewall()
        verdict = firewall.analyze(body.text)
        if verdict.blocked: return api_error("Input blocked", status_code=400)
        
        effective_text = verdict.sanitized_text or body.text

        # Resolve chat
        chat = None
        if body.chat_id:
            chat = await get_chat(db, body.chat_id, user_id=user_id)
        
        if not chat:
            chat_name = generate_chat_name(effective_text, "compressed")
            chat = await create_chat(db, chat_name, "compressed", user_id=user_id)

        await _safe_add_message(db, chat.id, user_id, "user", effective_text)

        # ── Context Injection ─────────────────────────────────────
        try:
            context_bundle = await safe_build_context(db, user_id, chat.id, effective_text)
            instructions = context_bundle.get("system_instructions", "")
            if instructions:
                effective_text = f"{instructions}\n\nUser: {effective_text}"
        except Exception as e:
            logger.debug(f"Compressed context builder skipped: {e}")

        from compressed.pipeline import run_compressed_pipeline
        result = await run_compressed_pipeline(query=effective_text, session_id=str(chat.id))
        
        formatted = sanitize_output(result.get("formatted_output", ""))
        await _safe_add_message(db, chat.id, user_id, "assistant", formatted)

        return api_success({
            "chat_id": str(chat.id),
            "formatted_output": formatted,
            "mode": "compressed",
            "confidence": result.get("metadata", {}).get("confidence", 0.8)
        })
    except Exception as e:
        logger.error(f"Error in /api/compressed: {e}")
        return api_error(str(e))


# ============================================================
# MULTIMODAL CAPABILITY AUDIT
# ============================================================

@app.post("/api/audit")
async def audit_capabilities(
    text: str = Form(...),
    file: Optional[UploadFile] = File(None),
    user: Dict = Depends(get_current_user),
):
    """
    Run the 8-phase multimodal capability audit without executing the pipeline.

    Returns a structured report:
      SYSTEM_AUDIT  — input classification, model availability
      MODEL_PIPELINE — which models would be assigned
      EXECUTION_STATUS — whether execution would succeed
    """
    from core.multimodal_auditor import MultimodalAuditor, audit_request

    image_b64 = None
    image_mime = None
    file_mime = None

    if file:
        file_mime = file.content_type
        if file_mime and file_mime.startswith("image/"):
            image_b64, image_mime = await _read_upload_as_b64(file)

    try:
        if multimodal_auditor:
            result = await multimodal_auditor.audit_and_route(
                query=text,
                image_b64=image_b64,
                image_mime=image_mime,
                file_mime=file_mime,
                execute=False,
            )
        else:
            result = await audit_request(
                query=text,
                image_b64=image_b64,
                image_mime=image_mime,
                file_mime=file_mime,
            )

        return api_success(result)
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        return api_error(str(e))


@app.post("/api/audit/execute")
async def audit_and_execute(
    text: str = Form(...),
    rounds: int = Form(3),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """
    Run the full 8-phase multimodal audit AND execute the pipeline.

    Returns SYSTEM_AUDIT + MODEL_PIPELINE + FINAL_RESPONSE.
    This is the capability-aware entry point that guarantees:
      - No silent capability skips
      - Vision models used for images
      - Minimum 3 models participate
      - Full audit trail in response
    """
    if not multimodal_auditor:
        raise HTTPException(status_code=503, detail="Multimodal auditor not initialized")

    image_b64 = None
    image_mime = None
    file_mime = None

    if file:
        file_mime = file.content_type
        if file_mime and file_mime.startswith("image/"):
            image_b64, image_mime = await _read_upload_as_b64(file)

    # Build history from context
    history = []
    if context:
        try:
            ctx = json.loads(context)
            if isinstance(ctx, list):
                history = ctx
        except Exception:
            pass

    try:
        result = await multimodal_auditor.audit_and_route(
            query=text,
            chat_id=str(chat_id) if chat_id else "",
            rounds=max(min(rounds, 10), 3),
            history=history,
            image_b64=image_b64,
            image_mime=image_mime,
            file_mime=file_mime,
            execute=True,
        )

        return api_success(result)
    except Exception as e:
        logger.error(f"Audit and execute failed: {e}")
        return api_error(str(e))


# ============================================================
# INDIVIDUAL MODEL MODE (Direct single-model query)
# ============================================================

@app.get("/api/models")
async def list_available_models(user: Dict = Depends(get_current_user)):
    """List all available models from the cognitive model registry."""
    try:
        from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY, MODEL_DEBATE_TIERS
        models = []
        for key, spec in COGNITIVE_MODEL_REGISTRY.items():
            models.append({
                "id": key,
                "name": spec.name,
                "provider": spec.provider,
                "role": spec.role.value,
                "tier": MODEL_DEBATE_TIERS.get(key, 2),
                "enabled": spec.enabled and spec.active,
                "context_window": spec.context_window,
                "max_output_tokens": spec.max_output_tokens,
                "synthesis_only": spec.synthesis_only,
            })
        return api_success({"models": models})
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return api_error(str(e))


@app.post("/api/models/claude/toggle")
async def toggle_claude(user: Dict = Depends(get_current_user)):
    """Toggle Claude on/off. Claude is synthesis-only — this controls whether it participates at all."""
    try:
        from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
        spec = COGNITIVE_MODEL_REGISTRY.get("claude-sonnet-4.6")
        if not spec:
            return api_error("Claude model not found in registry", status_code=404)
        spec.active = not spec.active
        return api_success({
            "model": "claude-sonnet-4.6",
            "active": spec.active,
            "synthesis_only": spec.synthesis_only,
            "message": f"Claude is now {'enabled (synthesis only)' if spec.active else 'disabled'}"
        })
    except Exception as e:
        logger.error(f"Failed to toggle Claude: {e}")
        return api_error(str(e))


@app.get("/api/models/claude/usage")
async def get_claude_usage_stats(user: Dict = Depends(get_current_user)):
    """Get Claude API usage statistics for cost monitoring."""
    from metacognitive.cognitive_gateway import get_claude_usage, COGNITIVE_MODEL_REGISTRY
    usage = get_claude_usage()
    spec = COGNITIVE_MODEL_REGISTRY.get("claude-sonnet-4.6")
    return {
        **usage,
        "budget_usd": 5.0,
        "remaining_usd": round(5.0 - usage["estimated_cost_usd"], 4),
        "active": spec.active if spec else False,
        "enabled": spec.enabled if spec else False,
    }


@app.post("/api/model/{model_id}")
async def query_individual_model(
    model_id: str,
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """
    Query a specific model directly (individual model mode).
    Supported model_ids: llama-3.3-70b, llama-3.1-8b, mixtral-8x7b, llama4-scout, gemini-flash, qwen-2.5-vl
    """
    import time as _time
    from compressed.model_clients import RoleBasedRouter

    user_id = user["user_id"]
    firewall = get_firewall()

    if len(request.text) > settings.MAX_INPUT_LENGTH:
        raise HTTPException(status_code=400, detail="Input too long.")

    verdict = firewall.analyze(request.text)
    if verdict.blocked:
        raise HTTPException(status_code=400, detail="Input blocked by safety filter.")
    effective_text = verdict.sanitized_text or request.text

    router = RoleBasedRouter()
    client = router.get_client_by_id(model_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found. Use GET /api/models to list available models.")
    if not client.available:
        raise HTTPException(status_code=503, detail=f"Model '{model_id}' is not configured (API key missing).")

    t0 = _time.time()
    resp = await client.generate(
        prompt=effective_text,
        system_instruction="You are Sentinel-E, an advanced AI reasoning assistant. Provide clear, well-structured answers.",
        max_tokens=2048,
        temperature=0.3,
    )
    latency_ms = (_time.time() - t0) * 1000

    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Model error: {resp.error}")

    return {
        "model_id": model_id,
        "model": resp.model,
        "response": resp.content,
        "tokens_in": resp.tokens_in,
        "tokens_out": resp.tokens_out,
        "latency_ms": round(latency_ms, 1),
    }


# ============================================================
# FORM-DATA ENDPOINTS (Frontend Compatibility)
# ============================================================

@app.post("/run/standard")
async def run_standard_form(
    text: str = Form(...),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Standard mode via FormData."""
    request = SentinelRequest(text=text, mode="standard", chat_id=chat_id)
    if file:
        img_b64, img_mime = await _read_upload_as_b64(file)
        request.image_b64 = img_b64
        request.image_mime = img_mime
    return await _run_sentinel_core(request, db, user["user_id"], frontend_context=context)



@app.post("/run/experimental")
async def run_experimental_form(
    text: str = Form(...),
    mode: str = Form("experimental"),
    rounds: int = Form(6),
    kill_switch: bool = Form(False),
    sub_mode: str = Form("debate"),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Experimental mode via FormData."""
    if kill_switch:
        request = SentinelRequest(text=text or "kill", mode="kill", rounds=min(rounds, settings.MAX_ROUNDS), chat_id=chat_id, sub_mode="glass")
        return await _run_sentinel_core(request, db, user["user_id"])

    valid_sub_modes = {"debate", "glass", "evidence", "synthesis"}
    sub_mode = sub_mode if sub_mode in valid_sub_modes else "debate"
    valid_modes = {"conversational", "experimental", "forensic", "standard"}
    mode = mode if mode in valid_modes else "experimental"

    request = SentinelRequest(
        text=text, mode=mode, sub_mode=sub_mode,
        rounds=min(rounds, settings.MAX_ROUNDS), chat_id=chat_id,
    )
    if file:
        img_b64, img_mime = await _read_upload_as_b64(file)
        request.image_b64 = img_b64
        request.image_mime = img_mime
    return await _run_sentinel_core(request, db, user["user_id"], frontend_context=context)



@app.post("/run/omega/standard")
async def omega_standard_form(
    text: str = Form(...),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    request = SentinelRequest(text=text, mode="standard", chat_id=chat_id)
    if file:
        img_b64, img_mime = await _read_upload_as_b64(file)
        request.image_b64 = img_b64
        request.image_mime = img_mime
    return await _run_sentinel_core(request, db, user["user_id"])



@app.post("/run/omega/experimental")
async def omega_experimental_form(
    text: str = Form(...),
    rounds: int = Form(3),
    sub_mode: str = Form("debate"),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    valid_sub_modes = {"debate", "glass", "evidence", "synthesis"}
    sub_mode = sub_mode if sub_mode in valid_sub_modes else "debate"
    request = SentinelRequest(text=text, mode="experimental", sub_mode=sub_mode, rounds=min(rounds, settings.MAX_ROUNDS), chat_id=chat_id)
    if file:
        img_b64, img_mime = await _read_upload_as_b64(file)
        request.image_b64 = img_b64
        request.image_mime = img_mime
    return await _run_sentinel_core(request, db, user["user_id"])



@app.post("/run/omega/kill")
async def omega_kill_form(
    text: str = Form(""),
    chat_id: Optional[UUID] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    request = SentinelRequest(text=text or "kill", mode="kill", sub_mode="glass", chat_id=chat_id)
    return await _run_sentinel_core(request, db, user["user_id"])



@app.post("/run/ensemble")
async def run_ensemble_form(
    text: str = Form(...),
    rounds: int = Form(3),
    chat_id: Optional[UUID] = Form(None),
    file: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """
    Ensemble endpoint — always-on multi-model reasoning.
    All requests route through CognitiveOrchestrator.
    Minimum 3 debate rounds enforced. No single-model fallback.
    """
    request = SentinelRequest(
        text=text,
        mode="ensemble",
        sub_mode="full_debate",
        rounds=max(min(rounds, settings.MAX_ROUNDS), 3),
        chat_id=chat_id,
    )
    if file:
        img_b64, img_mime = await _read_upload_as_b64(file)
        request.image_b64 = img_b64
        request.image_mime = img_mime
    return await _run_sentinel_core(request, db, user["user_id"], frontend_context=context)



# ============================================================
# FEEDBACK
# ============================================================

@app.post("/feedback")
async def feedback_endpoint(
    run_id: str = Form(...),
    feedback: str = Form(...),
    rating: Optional[int] = Form(None),
    reason: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    sub_mode: Optional[str] = Form(None),
    boundary_severity: Optional[float] = Form(None),
    fragility_index: Optional[float] = Form(None),
    disagreement_score: Optional[float] = Form(None),
    confidence: Optional[float] = Form(None),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Enhanced feedback with memory learning."""
    user_id = user["user_id"]
    logger.info("auth user_id=%s", user_id)

    try:
        chat_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    chat = await get_chat(db, chat_uuid, user_id=user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found or unauthorized")

    # Update chat metadata
    metadata = chat.machine_metadata or {}
    if "feedback" not in metadata:
        metadata["feedback"] = []

    feedback_entry = {
        "vote": feedback,
        "rating": rating,
        "reason": reason,
        "mode": mode,
        "sub_mode": sub_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(metadata.get("feedback"), list):
        metadata["feedback"].append(feedback_entry)
    else:
        metadata["feedback"] = [feedback_entry]

    await update_chat_metadata(db, chat.id, priority_answer=chat.priority_answer, machine_metadata=metadata, rounds=chat.rounds)

    # Memory learning
    memory = memory_sessions.get(str(chat.id))
    if memory:
        memory.record_feedback(
            vote=feedback,
            rating=rating,
            reason=reason,
            mode=mode,
        )

    # Knowledge learner
    if knowledge_learner:
        try:
            knowledge_learner.record_feedback(
                run_id=run_id, vote=feedback, rating=rating,
                mode=mode or "unknown", sub_mode=sub_mode, reason=reason,
            )
        except Exception as e:
            logger.warning(f"Knowledge learner feedback failed: {e}")

    return {"status": "success", "feedback_id": run_id}


# ============================================================
# CHAT HISTORY
# ============================================================

@app.get("/api/chats")
async def get_chats_list(
    limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    try:
        user_id = user["user_id"]
        chats = await list_chats(db, user_id, limit, offset)
        return api_success(chats)
    except Exception as e:
        logger.error(f"Error in get_chats_list: {e}")
        return api_success([])


@app.get("/api/history")
async def history_alias(
    limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: Optional[Dict] = Depends(get_optional_user),
):
    logger.info("Entering /api/history")
    logger.info("/api/history user authenticated=%s", bool(user))

    try:
        from database.schemas import ChatSchema, MessageSchema
        from database.models import Message

        if not user:
            logger.info("/api/history unauthenticated request — returning empty history")
            return api_success({
                "chats": [],
                "messages": [],
                "metadata": {
                    "total_chats": 0,
                    "total_messages": 0,
                    "fetched_at": datetime.utcnow().isoformat(),
                },
            })

        user_id = user.get("user_id")
        if not user_id:
            logger.warning("/api/history missing user_id in optional user payload")
            return api_success({
                "chats": [],
                "messages": [],
                "metadata": {
                    "total_chats": 0,
                    "total_messages": 0,
                    "fetched_at": datetime.utcnow().isoformat(),
                },
            })
        logger.info("auth user_id=%s", user_id)

        chats = await list_chats(db, user_id, limit, offset)
        logger.info("history status db=success op=list_chats user_id=%s", user_id)
        if chats is None:
            chats = []

        chat_ids = [c.id for c in chats]

        # Fetch all messages for these chats
        messages = []
        if chat_ids:
            try:
                msg_result = await db.execute(
                    select(Message)
                    .where(Message.chat_id.in_(chat_ids))
                    .order_by(Message.created_at.asc())
                )
                messages = msg_result.scalars().all()
                logger.info("history status db=success op=list_messages user_id=%s", user_id)
            except Exception as e:
                logger.warning(f"/api/history message fetch failed: {e}")
                logger.warning("history status db=fail op=list_messages user_id=%s", user_id)
                messages = []

        # Serialize chats safely
        serialized_chats = []
        for c in chats:
            try:
                serialized_chats.append(ChatSchema.model_validate(c).model_dump(mode="json"))
            except Exception as e:
                logger.warning(f"/api/history: failed to serialize chat {getattr(c, 'id', '?')}: {e}")
                # Minimal safe fallback
                try:
                    serialized_chats.append({
                        "id": str(c.id),
                        "chat_name": c.chat_name or "Untitled",
                        "mode": c.mode or "standard",
                        "user_id": c.user_id,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                        "rounds": c.rounds or 0,
                    })
                except Exception:
                    pass

        # Serialize messages safely
        serialized_messages = []
        for m in messages:
            try:
                msg_payload = MessageSchema.model_validate(m).model_dump(mode="json")
                # Never return base64 blobs in history list payloads.
                msg_payload.pop("image_b64", None)
                msg_payload.pop("image_mime", None)
                msg_payload["image_url"] = ((m.metadata_json or {}).get("image_url") if isinstance(m.metadata_json, dict) else None)
                serialized_messages.append(msg_payload)
            except Exception as e:
                logger.warning(f"/api/history: failed to serialize message {getattr(m, 'id', '?')}: {e}")
                try:
                    serialized_messages.append({
                        "id": str(m.id),
                        "chat_id": str(m.chat_id),
                        "user_id": m.user_id,
                        "role": m.role,
                        "content": m.content or "",
                        "image_url": (m.metadata_json or {}).get("image_url") if isinstance(m.metadata_json, dict) else None,
                        "reasoning_json": m.reasoning_json,
                        "metadata_json": m.metadata_json,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    })
                except Exception:
                    pass

        return api_success({
            "chats": serialized_chats,
            "messages": serialized_messages,
            "metadata": {
                "total_chats": len(serialized_chats),
                "total_messages": len(serialized_messages),
                "fetched_at": datetime.utcnow().isoformat(),
            },
        })
    except Exception as exc:
        logger.error(f"Unhandled /api/history error: {exc}", exc_info=True)
        logger.warning("history status db=fail op=history_alias")
        return api_success({
            "chats": [],
            "messages": [],
            "metadata": {
                "total_chats": 0,
                "total_messages": 0,
                "fetched_at": datetime.utcnow().isoformat(),
            },
        })



@app.get("/api/chat/{chat_id}")
async def get_chat_detail(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    try:
        user_id = user["user_id"]
        # ✅ Verify user owns this chat
        chat = await get_chat(db, chat_id, user_id=user_id)
        if not chat:
            return api_error("Chat not found or unauthorized", status_code=403)
        
        messages = await get_chat_messages(db, chat_id, user_id=user_id)
        return api_success({"chat": chat, "messages": messages})
    except Exception as e:
        logger.error(f"Error in get_chat_detail: {e}")
        return api_error(str(e))


@app.get("/api/chat/{chat_id}/messages")
async def get_messages(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    try:
        user_id = user["user_id"]
        # ✅ Verify user owns this chat before returning messages
        msgs = await get_chat_messages(db, chat_id, user_id=user_id)
        if not msgs:
            # Check if chat exists but user doesn't own it
            chat = await get_chat(db, chat_id)
            if chat and chat.user_id != user_id:
                return api_error("Unauthorized: Cannot access this chat", status_code=403)
        
        return api_success([
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "timestamp": m.created_at.isoformat() if m.created_at else None,
                "has_image": bool(m.image_b64),
                "image_b64": m.image_b64 if m.image_b64 else None,
                "image_mime": m.image_mime if m.image_mime else None,
                "reasoning_json": m.reasoning_json,
            }
            for m in msgs
        ])
    except Exception as e:
        logger.error(f"Error in get_messages: {e}")
        return api_success([])


@app.get("/api/history/{chat_id}")
async def history_detail(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    return await get_chat_detail(chat_id, db, user)


@app.put("/api/messages/{message_id}")
async def edit_message(
    message_id: UUID,
    request: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Edit a message's content (ChatGPT-style edit)."""
    try:
        new_content = request.get("content")
        if not new_content:
            return api_error("Content is required", status_code=400)

        msg = await update_message(db, message_id, new_content)
        if not msg:
            return api_error("Message not found", status_code=404)

        return api_success({"status": "updated", "message_id": str(message_id)})
    except Exception as e:
        logger.error(f"Error editing message: {e}")
        return api_error(str(e))


@app.post("/api/messages/{message_id}/regenerate")
async def regenerate_response(
    message_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
):
    """Regenerate the response after a user message."""
    try:
        from database.models import Message as MessageModel

        # Find the target message
        result = await db.execute(
            select(MessageModel).where(MessageModel.id == message_id)
        )
        target_msg = result.scalar_one_or_none()
        if not target_msg:
            return api_error("Message not found", status_code=404)

        if target_msg.role != "user":
            return api_error("Can only regenerate after a user message", status_code=400)

        # Delete all messages after this user message
        await delete_messages_after(db, target_msg.chat_id, message_id)

        # Re-run the query
        regen_request = SentinelRequest(
            text=target_msg.content,
            mode="standard",
            chat_id=str(target_msg.chat_id),
            image_b64=target_msg.image_b64,
            image_mime=target_msg.image_mime,
        )

        # We pass raw_request=None because it's internal (should be handled by standard run_sentinel or similar)
        # Actually, better to just call run_sentinel or a common helper.
        # But run_sentinel expects a Request object.
        # Let's just return a trigger for the frontend to re-send.
        return api_success({"status": "ready_for_regeneration", "chat_id": str(target_msg.chat_id)})
    except Exception as e:
        logger.error(f"Regeneration failed: {e}")
        return api_error(str(e))


# ============================================================
# SESSION & ANALYTICS ENDPOINTS
# ============================================================

@app.get("/api/omega/session/{chat_id}")
async def omega_session_state(
    chat_id: str,
    user: Dict = Depends(get_current_user),
):
    user_id = user["user_id"]
    
    if chat_id in omega_sessions:
        kernel = omega_sessions[chat_id]
        # ✅ FIX #5: Verify ownership before returning session state
        if hasattr(kernel, '_owner_user_id') and kernel._owner_user_id != user_id:
            logger.warning(f"Unauthorized session access: {chat_id} by {user_id}")
            raise HTTPException(status_code=403, detail="Access denied")
        return {
            "chat_id": chat_id,
            "session_state": kernel.get_session_state(),
            "initialized": kernel.is_initialized(),
        }
    kernel, _ = await _restore_session(chat_id, user_id)
    if kernel:
        return {
            "chat_id": chat_id,
            "session_state": kernel.get_session_state(),
            "initialized": kernel.is_initialized(),
        }
    return {"chat_id": chat_id, "session_state": None, "initialized": False}


@app.get("/api/session/{chat_id}/descriptive")
async def session_descriptive(
    chat_id: str,
    user: Dict = Depends(get_current_user),
):
    user_id = user["user_id"]
    
    kernel = omega_sessions.get(chat_id)
    if not kernel:
        kernel, _ = await _restore_session(chat_id, user_id)
    if not kernel:
        return {"error": "Session not found", "chat_id": chat_id}
    
    # ✅ FIX #5: Verify ownership before returning session details
    if hasattr(kernel, '_owner_user_id') and kernel._owner_user_id != user_id:
        logger.warning(f"Unauthorized session access: {chat_id} by {user_id}")
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        return kernel.session.get_descriptive_summary()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/cross-analysis")
async def cross_model_analysis(
    chat_id: Optional[str] = Body(None),
    query: str = Body(""),
    llm_response: str = Body(""),
    user: Dict = Depends(get_current_user),
):
    user_id = user["user_id"]
    
    if not mco_orchestrator or not mco_orchestrator.cognitive_gateway:
        raise HTTPException(status_code=503, detail="System not ready")

    try:
        if not llm_response and chat_id and chat_id in omega_sessions:
            kernel = omega_sessions[chat_id]
            # ✅ FIX #5: Verify kernel ownership
            if hasattr(kernel, '_owner_user_id') and kernel._owner_user_id != user_id:
                logger.warning(f"Unauthorized session access: {chat_id} by {user_id}")
                raise HTTPException(status_code=403, detail="Access denied")
            session_state = kernel.get_session_state()
            llm_response = session_state.get("last_response", "")

        if not llm_response:
            llm_response = "No response available for analysis."

        from metacognitive.schemas import CognitiveGatewayInput
        analysis_prompt = (
            f"Analyze the following AI response for accuracy, completeness, and potential issues.\n\n"
            f"Original query: {query}\n\nAI Response:\n{llm_response}\n\n"
            f"Provide a structured analysis with: factual accuracy, completeness, potential biases, "
            f"confidence assessment, and suggested improvements."
        )
        gw_input = CognitiveGatewayInput(
            user_query=analysis_prompt,
            stabilized_context={},
            knowledge_bundle=[],
            session_summary={},
        )
        # Run cross-analysis through MCO cognitive gateway in parallel
        outputs = await mco_orchestrator.cognitive_gateway.invoke_parallel(gw_input)
        result = {
            "analyses": {
                out.model_name: {
                    "analysis": out.raw_output,
                    "success": out.success,
                    "latency_ms": round(out.latency_ms, 1),
                }
                for out in outputs
                if out.success
            },
            "models_used": [out.model_name for out in outputs if out.success],
            "total_models": len(outputs),
        }
        return result
    except Exception as e:
        logger.error(f"Cross-analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.get("/api/learning")
async def learning_summary(user: Dict = Depends(get_current_user)):
    if not knowledge_learner:
        return {"status": "disabled"}
    try:
        return {
            "status": "active",
            "summary": knowledge_learner.get_learning_summary(),
            "threshold_suggestions": knowledge_learner.suggest_threshold_adjustments(),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# USER MEMORY & PREFERENCES
# ============================================================

@app.get("/api/user/memory")
async def get_memory_api(request: Request, db: AsyncSession = Depends(get_db)):
    """Get all user memory facts."""
    try:
        user_id = await get_user_id(request)
        if not user_id: return api_success([])
        from database.schemas import UserMemorySchema
        memories = await get_user_memory(db, user_id)
        return api_success([UserMemorySchema.model_validate(m).model_dump(mode="json") for m in (memories or [])])
    except Exception as e:
        logger.error(f"Error in /api/user/memory: {e}")
        return api_success([])


@app.post("/api/user/memory")
async def add_memory_api(request: Request, body: dict = Body(...), db: AsyncSession = Depends(get_db)):
    """Add or update a user memory fact."""
    try:
        user_id = await get_user_id(request)
        if not user_id: return api_error("Auth required", status_code=401)
        key, value = body.get("key"), body.get("value")
        if not key or not value: return api_error("key and value required")
        await add_user_memory(db, user_id, key, value, body.get("confidence", 75))
        return api_success({"status": "saved"})
    except Exception as e:
        logger.error(f"Error in POST /api/user/memory: {e}")
        return api_error(str(e))


@app.get("/api/user/preferences")
async def get_preferences_api(request: Request, db: AsyncSession = Depends(get_db)):
    """Get user preferences."""
    try:
        user_id = await get_user_id(request)
        if not user_id: return api_success({})
        prefs = await get_user_preferences(db, user_id)
        return api_success(prefs)
    except Exception as e:
        logger.error(f"Error in /api/user/preferences: {e}")
        return api_success({})


@app.put("/api/user/preferences")
async def update_preferences_api(request: Request, body: dict = Body(...), db: AsyncSession = Depends(get_db)):
    """Update user preferences."""
    try:
        user_id = await get_user_id(request)
        if not user_id: return api_error("Auth required", status_code=401)
        for k, v in body.items():
            await upsert_user_preference(db, user_id, k, str(v))
        return api_success({"status": "updated"})
    except Exception as e:
        logger.error(f"Error in PUT /api/user/preferences: {e}")
        return api_error(str(e))


# ============================================================
# PROVIDER STATUS (Admin Only)
# ============================================================


# Diagnostics endpoint: model registry status with disable reasons
@app.get("/api/models/status")
async def model_registry_status(user: Dict = Depends(get_current_user)):
    """Diagnostics: Full model registry status."""
    try:
        from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY
        return api_success({
            "models": [
                {
                    "id": key,
                    "name": spec.name,
                    "provider": spec.provider,
                    "enabled": spec.enabled,
                    "active": spec.active,
                    "disable_reason": getattr(spec, "disable_reason", None),
                }
                for key, spec in COGNITIVE_MODEL_REGISTRY.items()
            ]
        })
    except Exception as e:
        logger.error(f"Error fetching model status: {e}")
        return api_error(str(e))


# ============================================================
# OPTIMIZATION STATS
# ============================================================

@app.get("/api/optimization/stats")
async def optimization_stats(user: Dict = Depends(get_current_user)):
    """Optimization layer metrics (non-sensitive)."""
    try:
        cache = get_response_cache()
        governor = get_cost_governor()
        obs_hub = get_observability_hub()

        return api_success({
            "cache": cache.stats if cache else {},
            "cost": governor.get_global_stats() if governor else {},
            "observability": obs_hub.get_metrics() if obs_hub else {},
        })
    except Exception as e:
        logger.error(f"Error fetching optimization stats: {e}")
        return api_success({"cache": {}, "cost": {}, "observability": {}})


@app.get("/api/optimization/session/{chat_id}")
async def optimization_session_stats(
    chat_id: str,
    user: Dict = Depends(get_current_user),
):
    """Per-session budget status."""
    governor = get_cost_governor()
    return governor.get_session_budget(chat_id)


@app.get("/api/session/{session_id}/call-graph")
async def get_call_graph(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user)
):
    """GET /api/session/{session_id}/call-graph → returns DAG as JSON"""
    from utils.dag_logger import get_dag_logger
    logger = get_dag_logger()
    graph = await logger.get_session_graph(session_id)
    critical_path = await logger.get_critical_path(session_id)
    return api_success({
        "nodes": graph,
        "critical_path": critical_path
    })

@app.get("/api/session/{session_id}/debug")
async def get_session_debug(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user)
):
    """GET /api/session/{session_id}/debug → full diagnostic payload"""
    from utils.dag_logger import get_dag_logger
    from database.crud import get_user_memory
    
    logger = get_dag_logger()
    graph = await logger.get_session_graph(session_id)
    memories = await get_user_memory(db, user["user_id"])
    
    return api_success({
        "call_graph": graph,
        "memory": memories,
        "token_usage": sum(n.get("input_tokens", 0) + n.get("output_tokens", 0) for n in graph),
        "total_latency": sum(n.get("latency_ms", 0) for n in graph)
    })

# ============================================================
# STARTUP
# ============================================================

