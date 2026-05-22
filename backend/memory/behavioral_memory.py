"""
behavioral_memory.py — 7-Layer Adaptive Memory Engine

Layer architecture:
  1. working_memory        — current session context (resets per session)
  2. session_memory        — conversation context & topic continuity
  3. semantic_memory       — learned concepts, domains, expertise signals
  4. behavioral_memory     — interaction patterns & style preferences
  5. tactical_memory       — workflow habits & query strategies
  6. adaptive_preference   — response style, depth, format preferences
  7. runtime_optimization  — latency signals, routing efficiency data

All memory is strictly isolated per user_id. No cross-user data access.
"""

import logging
import json
import math
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from database.crud_v2 import upsert_memory, get_memory_by_key

logger = logging.getLogger("BehavioralMemory")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_profile() -> Dict[str, Any]:
    """Returns a blank-slate 7-layer behavioral profile."""
    return {
        # Layer 1: Working memory (volatile — reset each session by caller)
        "working_memory": {
            "current_topic": None,
            "context_window_tokens": 0,
            "last_reset": _now_iso(),
        },
        # Layer 2: Session memory (per-conversation continuity)
        "session_memory": {
            "total_sessions": 0,
            "avg_session_length": 0,
            "last_session": None,
            "topic_history": [],
        },
        # Layer 3: Semantic memory (domain & knowledge signals)
        "semantic_memory": {
            "inferred_domains": {},     # domain -> frequency
            "expertise_signals": {},    # topic -> confidence (0.0-1.0)
        },
        # Layer 4: Behavioral memory (interaction patterns)
        "behavioral_memory": {
            "interaction_patterns": {},
            "correction_frequency": 0,
            "avg_query_complexity": "balanced",
        },
        # Layer 5: Tactical memory (workflow habits)
        "tactical_memory": {
            "preferred_query_style": "conversational",  # conversational|analytical|direct
            "file_attachment_frequency": 0,
            "follow_up_rate": 0.0,
            "workflow_habits": {},
        },
        # Layer 6: Adaptive preference memory (response style)
        "adaptive_preference": {
            "preferred_reasoning_depth": "balanced",    # concise|balanced|deep
            "preferred_format": "prose",                # prose|structured|mixed
            "preferred_length": "medium",               # brief|medium|detailed
            "model_satisfaction": {},                   # model_id -> {uses, positive, negative}
        },
        # Layer 7: Runtime optimization memory (routing/latency signals)
        "runtime_optimization": {
            "avg_latency_ms": 0,
            "preferred_models": [],     # ranked by satisfaction rate
            "routing_optimizations": 0,
            "last_updated": _now_iso(),
        },
        # Meta
        "_schema_version": 2,
        "_created_at": _now_iso(),
        "_last_updated": _now_iso(),
        "_total_interactions": 0,
    }


class BehavioralMemoryManager:
    """
    Manages continuous adaptive learning for users via 7-layer memory model.
    Per-user isolation guaranteed: all DB operations keyed by user_id.
    """

    PROFILE_KEY = "behavioral_profile_v2"

    @staticmethod
    async def get_profile(db: AsyncSession, user_id: str) -> Dict[str, Any]:
        """Fetch or initialize the user's behavioral profile."""
        memory_entry = await get_memory_by_key(db, user_id=user_id, key=BehavioralMemoryManager.PROFILE_KEY)
        if memory_entry and memory_entry.value:
            raw = memory_entry.value
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    return _default_profile()
            # Migrate old v1 profiles
            if "_schema_version" not in raw:
                return BehavioralMemoryManager._migrate_v1(raw)
            return raw
        return _default_profile()

    @staticmethod
    def _migrate_v1(old: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 flat profile to v2 layered structure."""
        profile = _default_profile()
        if "preferred_reasoning_depth" in old:
            profile["adaptive_preference"]["preferred_reasoning_depth"] = old["preferred_reasoning_depth"]
        if "model_satisfaction" in old:
            profile["adaptive_preference"]["model_satisfaction"] = old["model_satisfaction"]
        if "correction_frequency" in old:
            profile["behavioral_memory"]["correction_frequency"] = old["correction_frequency"]
        if "interaction_patterns" in old:
            profile["behavioral_memory"]["interaction_patterns"] = old["interaction_patterns"]
        profile["_schema_version"] = 2
        return profile

    @staticmethod
    async def update_profile_async(
        db: AsyncSession,
        user_id: str,
        interaction_metrics: Dict[str, Any]
    ) -> None:
        """
        Update the user's profile from a single interaction. Runs as background task.
        interaction_metrics keys:
          model, latency_ms, vote ('up'|'down'|None), query_complexity,
          query_length, domain, sub_mode, session_id, file_attached
        """
        try:
            profile = await BehavioralMemoryManager.get_profile(db, user_id)

            model = interaction_metrics.get("model", "unknown")
            latency = interaction_metrics.get("latency_ms", 0)
            vote = interaction_metrics.get("vote")
            query_complexity = interaction_metrics.get("query_complexity", "balanced")
            domain = interaction_metrics.get("domain")
            file_attached = interaction_metrics.get("file_attached", False)

            profile["_total_interactions"] = profile.get("_total_interactions", 0) + 1

            # — Layer 4: Behavioral —
            bm = profile["behavioral_memory"]
            if vote == "down":
                bm["correction_frequency"] = bm.get("correction_frequency", 0) + 1
            bm["avg_query_complexity"] = query_complexity

            # — Layer 5: Tactical —
            tm = profile["tactical_memory"]
            if file_attached:
                tm["file_attachment_frequency"] = tm.get("file_attachment_frequency", 0) + 1

            # — Layer 6: Adaptive Preference —
            ap = profile["adaptive_preference"]
            if model not in ap["model_satisfaction"]:
                ap["model_satisfaction"][model] = {"uses": 0, "positive": 0, "negative": 0, "avg_latency_ms": 0}

            m_sat = ap["model_satisfaction"][model]
            m_sat["uses"] += 1
            if vote == "up":
                m_sat["positive"] += 1
            elif vote == "down":
                m_sat["negative"] += 1
            # Weighted running average latency
            if latency > 0:
                old_avg = m_sat.get("avg_latency_ms", 0)
                uses = m_sat["uses"]
                m_sat["avg_latency_ms"] = round((old_avg * (uses - 1) + latency) / uses, 1)

            # Update reasoning depth preference
            if query_complexity == "complex":
                ap["preferred_reasoning_depth"] = "deep"
            elif query_complexity == "simple" and ap["preferred_reasoning_depth"] != "deep":
                ap["preferred_reasoning_depth"] = "concise"

            # — Layer 3: Semantic —
            if domain:
                sm = profile["semantic_memory"]
                sm["inferred_domains"][domain] = sm["inferred_domains"].get(domain, 0) + 1

            # — Layer 7: Runtime optimization —
            ro = profile["runtime_optimization"]
            if latency > 0:
                prev_avg = ro.get("avg_latency_ms", 0)
                total = profile["_total_interactions"]
                ro["avg_latency_ms"] = round((prev_avg * (total - 1) + latency) / total, 1)
            # Rebuild preferred_models rank by satisfaction rate
            ro["preferred_models"] = BehavioralMemoryManager._rank_models(ap["model_satisfaction"])
            ro["routing_optimizations"] = ro.get("routing_optimizations", 0) + (1 if vote == "up" else 0)
            ro["last_updated"] = _now_iso()

            profile["_last_updated"] = _now_iso()

            # Apply adaptive decay to de-weight stale preferences
            profile = BehavioralMemoryManager._adaptive_decay(profile)

            await upsert_memory(
                db=db, user_id=user_id,
                key=BehavioralMemoryManager.PROFILE_KEY,
                value=profile, weight=1.0, confidence=100, tag="system_adaptive"
            )
            logger.info(f"Updated 7-layer behavioral profile for user {user_id[:8]}…")

        except Exception as e:
            logger.error(f"Failed to update behavioral profile for {user_id[:8]}: {e}")

    @staticmethod
    def _rank_models(satisfaction: Dict[str, Any]) -> List[str]:
        """Rank models by satisfaction rate (positive / total uses)."""
        ranked = []
        for model_id, data in satisfaction.items():
            uses = data.get("uses", 0)
            if uses == 0:
                continue
            rate = data.get("positive", 0) / uses
            ranked.append((model_id, rate))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in ranked[:5]]

    @staticmethod
    def _adaptive_decay(profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce influence of very stale data. Called on each interaction write.
        Simple: expertise signals decay toward 0.5 (neutral) if not reinforced recently.
        """
        sm = profile.get("semantic_memory", {})
        expertise = sm.get("expertise_signals", {})
        # Decay by 2% per update toward neutral
        for topic in expertise:
            current = expertise[topic]
            expertise[topic] = round(current * 0.98 + 0.5 * 0.02, 4)
        sm["expertise_signals"] = expertise
        return profile

    @staticmethod
    def detect_behavioral_drift(profile: Dict[str, Any]) -> Optional[str]:
        """
        Detect if user behavior has significantly shifted.
        Returns a drift signal string if detected, else None.
        """
        bm = profile.get("behavioral_memory", {})
        correction_freq = bm.get("correction_frequency", 0)
        total = profile.get("_total_interactions", 1)

        # High correction rate = drift signal
        if total > 10 and correction_freq / total > 0.3:
            return "high_correction_rate"
        return None

    @staticmethod
    def format_profile_for_prompt(profile: Dict[str, Any]) -> str:
        """Convert the behavioral profile into a semantic orchestration hint string."""
        if not profile:
            return ""

        hints = []

        # Reasoning depth
        depth = profile.get("adaptive_preference", {}).get("preferred_reasoning_depth", "balanced")
        if depth == "deep":
            hints.append("User prefers deep, analytical reasoning with full explanations.")
        elif depth == "concise":
            hints.append("User prefers concise, direct answers without excess detail.")

        # Correction frequency
        corrections = profile.get("behavioral_memory", {}).get("correction_frequency", 0)
        total = profile.get("_total_interactions", 1)
        if total > 5 and corrections / total > 0.25:
            hints.append("User frequently corrects outputs — prioritize accuracy and verify assumptions.")

        # Format preference
        fmt = profile.get("adaptive_preference", {}).get("preferred_format", "prose")
        if fmt == "structured":
            hints.append("User prefers structured output with headers and bullet points.")

        # Domain expertise signals
        domains = profile.get("semantic_memory", {}).get("inferred_domains", {})
        if domains:
            top_domain = max(domains, key=domains.get)
            if domains[top_domain] > 3:
                hints.append(f"User has demonstrated strong interest in {top_domain}.")

        # Preferred model hint
        preferred = profile.get("runtime_optimization", {}).get("preferred_models", [])
        if preferred:
            hints.append(f"Preferred model routing: {preferred[0]}.")

        return " ".join(hints)
