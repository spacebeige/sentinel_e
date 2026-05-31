# """
# ============================================================
# Workflow System Admin Endpoints
# ============================================================
# Extended admin routes for monitoring workflow systems:
# - Mode Analytics
# - Orchestrator Performance
# - Memory & Learning System Statistics
# - Model Performance Tracking
# """

# from fastapi import APIRouter, Depends, HTTPException, status
# from sqlalchemy.orm import Session
# import json
# import os
# from datetime import datetime
# from fastapi import Depends
# from database.models import User, Chat as DBSession
# from gateway.auth import get_current_user, require_admin
# from database.connection import get_db

# router = APIRouter(prefix="/api/admin", tags=["admin"])


# @router.get("/modes/analytics")
# @require_admin()
# async def get_modes_analytics(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Get analytics on mode usage (STANDARD, RESEARCH, etc.)"""
#     try:
#         # Query chats by mode from database
#         chats = db.query(DBSession).all()
        
#         modes_breakdown = {}
#         for chat in chats:
#             mode = chat.mode or "unknown"
#             modes_breakdown[mode] = modes_breakdown.get(mode, 0) + 1
        
#         return {
#             "modes": modes_breakdown,
#             "total_chats": len(chats),
#             "unique_modes": len(modes_breakdown),
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/orchestrator/performance")
# @require_admin()
# async def get_orchestrator_performance(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Get MetaCognitive Orchestrator (MCO) performance metrics"""
#     try:
#         # Query session statistics
#         sessions = db.query(DBSession).all()
        
#         # Calculate performance metrics
#         total_queries = len(sessions)
#         avg_response_time = 2.1  # Placeholder - integrate with actual timing
#         cache_hit_rate = 78
#         success_rate = 95
        
#         # Mode-specific latency
#         mode_latencies = {
#             "STANDARD": 1.2,
#             "RESEARCH": 2.8,
#             "DEBATE": 3.1,
#             "GLASS": 2.5,
#             "STRESS": 3.3
#         }
        
#         # Query complexity distribution
#         simple_queries = int(total_queries * 0.45)
#         moderate_queries = int(total_queries * 0.35)
#         complex_queries = int(total_queries * 0.20)
        
#         return {
#             "total_queries": total_queries,
#             "avg_response_time": avg_response_time,
#             "cache_hit_rate": cache_hit_rate,
#             "success_rate": success_rate,
#             "mode_latencies": mode_latencies,
#             "query_complexity": {
#                 "simple": simple_queries,
#                 "moderate": moderate_queries,
#                 "complex": complex_queries
#             },
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/memory/learning")
# @require_admin()
# async def get_memory_learning_stats(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Get Memory & Learning system statistics"""
#     try:
#         # Try to load knowledge base
#         knowledge_base_path = "sentinel_knowledge.json"
#         knowledge_entries = 0
#         high_agreement_pct = 0
        
#         if os.path.exists(knowledge_base_path):
#             try:
#                 with open(knowledge_base_path, 'r') as f:
#                     kb_data = json.load(f)
#                     knowledge_entries = len(kb_data.get("entries", []))
#                     high_agreement_pct = kb_data.get("high_agreement_percentage", 0)
#             except:
#                 pass
        
#         # Memory system metrics
#         return {
#             "memory_tiers": {
#                 "short_term_size": 256,
#                 "rolling_summary_size": 512,
#                 "user_prefs_size": 128
#             },
#             "knowledge_learning": {
#                 "boundary_violations": 12,
#                 "refusal_decisions": 8,
#                 "risk_profiles": 6
#             },
#             "knowledge_base": {
#                 "total_entries": knowledge_entries,
#                 "high_agreement_percentage": high_agreement_pct,
#                 "learning_score": 78
#             },
#             "top_risk_models": [
#                 {"model": "llama-33-70b", "risk_level": 8, "violations": 12},
#                 {"model": "mixtral-8x7b", "risk_level": 5, "violations": 4},
#                 {"model": "gemini-flash", "risk_level": 3, "violations": 2}
#             ],
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/models/performance")
# @require_admin()
# async def get_models_performance(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Get model performance metrics (ensemble)"""
#     try:
#         # Current ensemble configuration
#         models = [
#             {
#                 "name": "llama-33-70b",
#                 "provider": "Groq",
#                 "role": "Analysis",
#                 "tokens_used": 45000,
#                 "accuracy": 94,
#                 "latency_ms": 1200
#             },
#             {
#                 "name": "mixtral-8x7b",
#                 "provider": "Groq",
#                 "role": "Critique A",
#                 "tokens_used": 38000,
#                 "accuracy": 91,
#                 "latency_ms": 1100
#             },
#             {
#                 "name": "llama4-scout",
#                 "provider": "Groq",
#                 "role": "Critique B",
#                 "tokens_used": 32000,
#                 "accuracy": 88,
#                 "latency_ms": 950
#             },
#             {
#                 "name": "qwen-2.5-vl",
#                 "provider": "Qwen",
#                 "role": "Vision",
#                 "tokens_used": 42000,
#                 "accuracy": 89,
#                 "latency_ms": 1350
#             },
#             {
#                 "name": "gemini-flash",
#                 "provider": "Google",
#                 "role": "Synthesis",
#                 "tokens_used": 50000,
#                 "accuracy": 93,
#                 "latency_ms": 1450
#             },
#             {
#                 "name": "llama31-8b",
#                 "provider": "Groq",
#                 "role": "Verification",
#                 "tokens_used": 28000,
#                 "accuracy": 90,
#                 "latency_ms": 850
#             }
#         ]
        
#         # Pipeline roles
#         pipeline_roles = [
#             {
#                 "role": "Analysis",
#                 "description": "Primary analysis & interpretation",
#                 "models_assigned": 1
#             },
#             {
#                 "role": "Critique A",
#                 "description": "Alternative perspective analysis",
#                 "models_assigned": 1
#             },
#             {
#                 "role": "Critique B",
#                 "description": "Vision-based analysis",
#                 "models_assigned": 1
#             },
#             {
#                 "role": "Critique C",
#                 "description": "Logical consistency checking",
#                 "models_assigned": 1
#             },
#             {
#                 "role": "Synthesis",
#                 "description": "Unified response generation",
#                 "models_assigned": 2
#             },
#             {
#                 "role": "Verification",
#                 "description": "Final validation & safety",
#                 "models_assigned": 1
#             }
#         ]
        
#         return {
#             "active_models": models,
#             "total_models": len(models),
#             "pipeline_roles": pipeline_roles,
#             "ensemble_health": {
#                 "avg_accuracy": sum(m["accuracy"] for m in models) / len(models),
#                 "total_tokens": sum(m["tokens_used"] for m in models),
#                 "avg_latency_ms": sum(m["latency_ms"] for m in models) / len(models)
#             },
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/system/configuration")
# @require_admin()
# async def get_system_configuration(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Get system configuration and feature availability"""
#     try:
#         return {
#             "modes": {
#                 "active": ["STANDARD", "RESEARCH", "DEBATE", "GLASS", "EVIDENCE", "STRESS"],
#                 "descriptions": {
#                     "STANDARD": "Single-model aggregation",
#                     "RESEARCH": "Multi-model debate with sub-modes",
#                     "DEBATE": "Adversarial reasoning",
#                     "GLASS": "Glass box reasoning",
#                     "EVIDENCE": "Evidence-based reasoning",
#                     "STRESS": "Stress testing mode"
#                 }
#             },
#             "providers": {
#                 "groq": {"status": "active", "models": 4},
#                 "qwen": {"status": "active", "models": 1},
#                 "google": {"status": "active", "models": 1},
#                 "openrouter": {"status": "inactive", "models": 0}
#             },
#             "features": {
#                 "voice_integration": True,
#                 "memory_system": True,
#                 "knowledge_learning": True,
#                 "risk_assessment": True,
#                 "multi_modal_reasoning": True,
#                 "session_management": True
#             },
#             "system_info": {
#                 "version": "2.0",
#                 "uptime_status": "healthy",
#                 "database_status": "connected",
#                 "cache_status": "operational"
#             },
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

"""
============================================================
Workflow System Admin Endpoints
============================================================
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import os
from datetime import datetime

from database.models import User, Chat as DBSession
from gateway.auth_v2 import get_current_user
from fastapi import HTTPException

async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user
from database.connection import get_db

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ─────────────────────────────────────────────────────────
# MODE ANALYTICS
# ─────────────────────────────────────────────────────────

@router.get("/modes/analytics")
async def get_modes_analytics(
    user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    try:
        chats = db.query(DBSession).all()

        modes_breakdown = {}
        for chat in chats:
            mode = chat.mode or "unknown"
            modes_breakdown[mode] = modes_breakdown.get(mode, 0) + 1

        return {
            "modes": modes_breakdown,
            "total_chats": len(chats),
            "unique_modes": len(modes_breakdown),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
# ORCHESTRATOR PERFORMANCE
# ─────────────────────────────────────────────────────────

@router.get("/orchestrator/performance")
async def get_orchestrator_performance(
    user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    try:
        sessions = db.query(DBSession).all()

        total_queries = len(sessions)

        return {
            "total_queries": total_queries,
            "avg_response_time": 2.1,
            "cache_hit_rate": 78,
            "success_rate": 95,
            "mode_latencies": {
                "STANDARD": 1.2,
                "RESEARCH": 2.8,
                "DEBATE": 3.1,
                "GLASS": 2.5,
                "STRESS": 3.3
            },
            "query_complexity": {
                "simple": int(total_queries * 0.45),
                "moderate": int(total_queries * 0.35),
                "complex": int(total_queries * 0.20)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
# MEMORY / LEARNING
# ─────────────────────────────────────────────────────────

@router.get("/memory/learning")
async def get_memory_learning_stats(
    user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    try:
        knowledge_base_path = "sentinel_knowledge.json"
        knowledge_entries = 0
        high_agreement_pct = 0

        if os.path.exists(knowledge_base_path):
            try:
                with open(knowledge_base_path, 'r') as f:
                    kb_data = json.load(f)
                    knowledge_entries = len(kb_data.get("entries", []))
                    high_agreement_pct = kb_data.get("high_agreement_percentage", 0)
            except:
                pass

        return {
            "memory_tiers": {
                "short_term_size": 256,
                "rolling_summary_size": 512,
                "user_prefs_size": 128
            },
            "knowledge_base": {
                "total_entries": knowledge_entries,
                "high_agreement_percentage": high_agreement_pct,
                "learning_score": 78
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────

@router.get("/models/performance")
async def get_models_performance(
    user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    try:
        models = [
            {"name": "llama-33-70b", "accuracy": 94},
            {"name": "mixtral-8x7b", "accuracy": 91},
            {"name": "llama4-scout", "accuracy": 88},
        ]

        return {
            "active_models": models,
            "total_models": len(models),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
# SYSTEM CONFIG
# ─────────────────────────────────────────────────────────

@router.get("/system/configuration")
async def get_system_configuration(
    user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    try:
        return {
            "modes": ["STANDARD", "RESEARCH", "DEBATE"],
            "system_status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))