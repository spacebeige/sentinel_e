import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from database.crud_v2 import upsert_memory, get_memory_by_key

logger = logging.getLogger("BehavioralMemory")

class BehavioralMemoryManager:
    """
    Manages continuous learning and adaptive cognitive profiling for users.
    Ensures per-user isolation by leveraging the Memory table with a specific key.
    """
    
    PROFILE_KEY = "behavioral_profile"
    
    @staticmethod
    async def get_profile(db: AsyncSession, user_id: str) -> Dict[str, Any]:
        """
        Fetch the user's isolated behavioral profile.
        Returns a default empty profile if none exists.
        """
        memory_entry = await get_memory_by_key(db, user_id=user_id, key=BehavioralMemoryManager.PROFILE_KEY)
        if memory_entry and memory_entry.value:
            if isinstance(memory_entry.value, str):
                try:
                    return json.loads(memory_entry.value)
                except Exception:
                    return {}
            return memory_entry.value
        
        # Default blank slate profile
        return {
            "interaction_patterns": {},
            "preferred_reasoning_depth": "balanced",
            "model_satisfaction": {},
            "workflow_habits": {},
            "correction_frequency": 0
        }

    @staticmethod
    async def update_profile_async(db: AsyncSession, user_id: str, interaction_metrics: Dict[str, Any]) -> None:
        """
        Asynchronously update the user's behavioral profile based on recent interaction.
        This runs as a background task to avoid blocking the main chat response.
        """
        try:
            profile = await BehavioralMemoryManager.get_profile(db, user_id)
            
            # Simple learning heuristics:
            # Update model satisfaction
            model = interaction_metrics.get("model", "unknown")
            latency = interaction_metrics.get("latency_ms", 0)
            user_vote = interaction_metrics.get("vote") # 'up', 'down', or None
            
            if model not in profile["model_satisfaction"]:
                profile["model_satisfaction"][model] = {"uses": 0, "positive": 0, "negative": 0}
            
            profile["model_satisfaction"][model]["uses"] += 1
            if user_vote == "up":
                profile["model_satisfaction"][model]["positive"] += 1
            elif user_vote == "down":
                profile["model_satisfaction"][model]["negative"] += 1
                profile["correction_frequency"] = profile.get("correction_frequency", 0) + 1
            
            # Infer preferred reasoning depth based on query length / complexity (naive proxy for now)
            query_complexity = interaction_metrics.get("query_complexity", "unknown")
            if query_complexity == "complex":
                profile["preferred_reasoning_depth"] = "deep"
            elif query_complexity == "simple" and profile["preferred_reasoning_depth"] != "deep":
                profile["preferred_reasoning_depth"] = "concise"
            
            # Upsert updated profile back to DB
            await upsert_memory(
                db=db,
                user_id=user_id,
                key=BehavioralMemoryManager.PROFILE_KEY,
                value=profile,
                weight=1.0,
                confidence=100,
                tag="system_adaptive"
            )
            logger.info(f"Updated behavioral profile for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update behavioral profile for {user_id}: {str(e)}")

    @staticmethod
    def format_profile_for_prompt(profile: Dict[str, Any]) -> str:
        """
        Convert the raw JSON profile into a semantic string for the orchestrator.
        """
        if not profile:
            return ""
            
        hints = []
        depth = profile.get("preferred_reasoning_depth", "balanced")
        if depth == "deep":
            hints.append("User prefers deep, analytical reasoning.")
        elif depth == "concise":
            hints.append("User prefers concise, direct synthesis.")
            
        corrections = profile.get("correction_frequency", 0)
        if corrections > 5:
            hints.append("User frequently corrects outputs; ensure high accuracy and verify assumptions.")
            
        if not hints:
            return ""
            
        return " ".join(hints)
