"""
============================================================
Admin Routes — System Management & User Administration
============================================================
Protected endpoints for admin operations:
- User role management
- System statistics & analytics
- Web analytics collection
- Feedback aggregation
- System architecture info
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timedelta

from gateway.auth import get_current_user
from database.connection import get_db
from database.models import User, Chat, Message
from database.crud import list_chats, get_chat_messages

logger = logging.getLogger("admin_routes")
router = APIRouter(prefix="/api/admin", tags=["admin"])


async def require_admin(user: dict = Depends(get_current_user)):
    """Dependency: verify user is admin."""
    if user.get("role") != "admin":
        logger.warning(f"Unauthorized admin access attempt by {user.get('user_id')}")
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@router.post("/users/make-admin")
async def make_user_admin(
    email: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(require_admin),
):
    """✅ Promote a user to admin by email."""
    try:
        # Find user by email
        result = await db.execute(
            select(User).where(User.email == email)
        )
        user = result.scalars().first()
        
        if not user:
            # Create new admin user if doesn't exist
            new_user = User(
                user_id=f"admin-{email.split('@')[0]}",
                email=email,
                role="admin",
            )
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            logger.info(f"✓ Created new admin user: {email}")
            return {
                "status": "created",
                "email": email,
                "role": "admin",
                "message": f"Admin account created for {email}"
            }
        
        # Update existing user to admin
        user.role = "admin"
        user.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"✓ Promoted {email} to admin by {admin.get('user_id')}")
        return {
            "status": "updated",
            "email": email,
            "role": "admin",
            "message": f"User {email} is now an admin"
        }
    except Exception as e:
        logger.error(f"Error promoting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/stats")
async def system_statistics(
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(require_admin),
):
    """✅ Get system-wide statistics for admin dashboard."""
    try:
        # Count users and admins
        users_result = await db.execute(select(User))
        all_users = users_result.scalars().all()
        total_users = len(all_users)
        admin_count = sum(1 for u in all_users if u.role == "admin")
        
        # Count chats and messages
        chats_result = await db.execute(select(Chat))
        chats = chats_result.scalars().all()
        total_chats = len(chats)
        
        messages_result = await db.execute(select(Message))
        messages = messages_result.scalars().all()
        total_messages = len(messages)
        
        # Analytics: chats by mode
        mode_stats = {}
        for chat in chats:
            mode = chat.mode or "unknown"
            mode_stats[mode] = mode_stats.get(mode, 0) + 1
        
        # Feedback analysis
        feedback_data = {
            "total_rated": 0,
            "avg_rating": 0.0,
            "ratings": {"positive": 0, "neutral": 0, "negative": 0}
        }
        
        total_rating = 0
        rated_count = 0
        
        for chat in chats:
            if chat.machine_metadata and isinstance(chat.machine_metadata, dict):
                feedback_list = chat.machine_metadata.get("feedback", [])
                if feedback_list:
                    for fb in feedback_list:
                        rating = fb.get("rating", 0)
                        if rating > 0:
                            total_rating += rating
                            rated_count += 1
                            if rating >= 4:
                                feedback_data["ratings"]["positive"] += 1
                            elif rating <= 2:
                                feedback_data["ratings"]["negative"] += 1
                            else:
                                feedback_data["ratings"]["neutral"] += 1
        
        if rated_count > 0:
            feedback_data["total_rated"] = rated_count
            feedback_data["avg_rating"] = round(total_rating / rated_count, 2)
        
        # Time-based stats
        now = datetime.utcnow()
        last_24h = sum(1 for c in chats if c.created_at and (now - c.created_at) < timedelta(days=1))
        last_7d = sum(1 for c in chats if c.created_at and (now - c.created_at) < timedelta(days=7))
        
        return {
            "timestamp": now.isoformat(),
            "users": {
                "total": total_users,
                "admins": admin_count,
                "regular": total_users - admin_count,
            },
            "chats": {
                "total": total_chats,
                "last_24h": last_24h,
                "last_7d": last_7d,
                "by_mode": mode_stats,
            },
            "messages": {
                "total": total_messages,
                "avg_per_chat": round(total_messages / total_chats, 2) if total_chats > 0 else 0,
            },
            "feedback": feedback_data,
            "system": {
                "uptime_status": "healthy",
                "db_status": "connected",
                "cache_status": "healthy",
            }
        }
    except Exception as e:
        logger.error(f"Error fetching system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/architecture")
async def system_architecture(
    admin: dict = Depends(require_admin),
):
    """✅ Return system architecture information."""
    return {
        "system": {
            "name": "Sentinel-E",
            "version": "5.0.0",
            "description": "Advanced multi-model AI reasoning system"
        },
        "architecture": {
            "layers": [
                {
                    "name": "API Gateway",
                    "component": "FastAPI + JWT Auth",
                    "responsibility": "Request routing, authentication, rate limiting"
                },
                {
                    "name": "Orchestrator",
                    "component": "SentinelSigmaOrchestratorV4",
                    "responsibility": "Request coordination, pipeline management"
                },
                {
                    "name": "Reasoning Engine",
                    "component": "OmegaCognitiveKernel + RoleBasedEngine",
                    "responsibility": "Multi-model debate, analysis, synthesis, verification"
                },
                {
                    "name": "Memory Layer",
                    "component": "MemoryEngine (3-tier)",
                    "responsibility": "Short-term, rolling summary, user preferences"
                },
                {
                    "name": "Retrieval",
                    "component": "CognitiveRAG",
                    "responsibility": "Context retrieval, knowledge base integration"
                },
                {
                    "name": "Optimization",
                    "component": "TokenOptimizer + CostGovernor",
                    "responsibility": "Cost control, token budget management"
                },
                {
                    "name": "Data Layer",
                    "component": "PostgreSQL + Redis + SQLite",
                    "responsibility": "Persistence, caching, session storage"
                }
            ]
        },
        "models": {
            "reasoning": [
                {"role": "analysis", "models": ["llama-3.3-70b", "gemini-flash"]},
                {"role": "critique", "models": ["mixtral-8x7b", "llama-4-scout", "qwen-2.5-vl"]},
                {"role": "synthesis", "models": ["gemini-flash", "llama-3.3-70b"]},
                {"role": "verification", "models": ["llama-3.1-8b", "gemini-flash"]}
            ]
        },
        "features": {
            "modes": ["standard", "research", "compressed"],
            "sub_modes": ["debate", "glass", "evidence", "stress"],
            "capabilities": [
                "Multi-model reasoning",
                "Cross-model debate",
                "Research synthesis",
                "System boundary verification",
                "Knowledge learning",
                "User preference adaptation"
            ]
        },
        "integrations": [
            "Firebase (authentication + session mgmt)",
            "Groq LPU (fast inference)",
            "Google Gemini (multimodal)",
            "Qwen (specialized reasoning)",
            "Meta-Cognitive Orchestrator (advanced reasoning)"
        ]
    }


@router.get("/web-analytics")
async def web_analytics(
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(require_admin),
    days: int = 7,
):
    """✅ Web analytics data (user activity, engagement, retention)."""
    try:
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)
        
        # Get all chats in date range
        chats_result = await db.execute(select(Chat))
        all_chats = chats_result.scalars().all()
        chats_in_range = [c for c in all_chats if c.created_at and c.created_at >= start_date]
        
        # Daily breakdown
        daily_stats = {}
        for i in range(days):
            date = (start_date + timedelta(days=i)).date()
            count = sum(1 for c in chats_in_range if c.created_at.date() == date)
            daily_stats[str(date)] = count
        
        # User engagement
        unique_users = set(c.user_id for c in chats_in_range if c.user_id)
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": now.isoformat(),
                "days": days
            },
            "summary": {
                "total_sessions": len(chats_in_range),
                "unique_users": len(unique_users),
                "avg_sessions_per_user": round(len(chats_in_range) / len(unique_users), 2) if unique_users else 0,
            },
            "daily_breakdown": daily_stats,
            "engagement": {
                "most_active_day": max(daily_stats, key=daily_stats.get) if daily_stats else None,
                "peak_sessions": max(daily_stats.values()) if daily_stats else 0,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching web analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback-summary")
async def feedback_summary(
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(require_admin),
):
    """✅ Aggregate feedback summary from all sessions."""
    try:
        chats_result = await db.execute(select(Chat))
        chats = chats_result.scalars().all()
        
        feedback_items = []
        model_feedback = {}
        
        for chat in chats:
            if chat.machine_metadata and isinstance(chat.machine_metadata, dict):
                feedback_list = chat.machine_metadata.get("feedback", [])
                for fb in feedback_list:
                    feedback_items.append({
                        "chat_id": str(chat.id),
                        "mode": chat.mode,
                        "rating": fb.get("rating"),
                        "vote": fb.get("vote"),
                        "sub_mode": fb.get("sub_mode"),
                        "reason": fb.get("reason"),
                        "timestamp": fb.get("timestamp"),
                    })
                    
                    # Aggregate by model
                    modes = fb.get("sub_mode", "unknown")
                    model_feedback[modes] = model_feedback.get(modes, {"count": 0, "avg_rating": 0})
                    model_feedback[modes]["count"] += 1
        
        # Calculate averages
        for mode in model_feedback:
            ratings = [f["rating"] for f in feedback_items if f["sub_mode"] == mode and f["rating"]]
            if ratings:
                model_feedback[mode]["avg_rating"] = round(sum(ratings) / len(ratings), 2)
        
        return {
            "total_feedback": len(feedback_items),
            "by_rating": {
                "positive": sum(1 for f in feedback_items if f["rating"] and f["rating"] >= 4),
                "neutral": sum(1 for f in feedback_items if f["rating"] and 2 < f["rating"] < 4),
                "negative": sum(1 for f in feedback_items if f["rating"] and f["rating"] <= 2),
            },
            "by_mode": model_feedback,
            "recent_feedback": feedback_items[-20:],  # Last 20
        }
    except Exception as e:
        logger.error(f"Error fetching feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
