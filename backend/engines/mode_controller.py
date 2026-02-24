"""
Mode Controller — Sentinel-E Centralized Routing

No internal mode cross-calling allowed.
Each mode executes its own engine exclusively.

Standard → runParallelAggregation (AggregationEngine)
Debate   → runAdversarialDebate (DebateOrchestrator — existing)
Evidence → runForensicPipeline (ForensicEvidenceEngine)
Glass    → runBlindAudit (BlindAuditEngine)
Stress   → runStressTest (StressEngine — existing)

Trigger word detection: if Standard mode input contains evidence triggers,
automatically route to Evidence mode.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("ModeController")

# Evidence trigger words
EVIDENCE_TRIGGERS = [
    "cite", "source", "did that happen", "prove", "evidence",
    "verify", "proof", "citation", "reference", "fact check",
    "is that true", "really", "actually happened", "show me",
    "did that really", "prove it", "is that accurate",
]


def detect_evidence_trigger(text: str) -> bool:
    """Detect if user input contains evidence trigger words."""
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in EVIDENCE_TRIGGERS)


def resolve_execution_mode(
    mode: str, sub_mode: str, text: str, kill_override: bool = False
) -> Dict[str, Any]:
    """
    Resolve the actual execution mode from request parameters.
    
    Handles:
    - Trigger word detection (Standard → Evidence override)
    - Kill override routing
    - Mode validation
    
    Returns:
        {
            "engine": "aggregation" | "debate" | "evidence" | "glass" | "stress",
            "mode": resolved mode string,
            "sub_mode": resolved sub_mode string,
            "trigger_override": bool — whether trigger words caused mode change,
            "kill_active": bool,
        }
    """
    # Kill override — always routes to glass
    if kill_override:
        return {
            "engine": "glass",
            "mode": "research",
            "sub_mode": "glass",
            "trigger_override": False,
            "kill_active": True,
        }

    # Standard mode
    if mode in ("standard", "conversational", "forensic"):
        # Check for evidence triggers
        if detect_evidence_trigger(text):
            logger.info(f"Evidence trigger detected in Standard mode input: '{text[:50]}...'")
            return {
                "engine": "evidence",
                "mode": "research",
                "sub_mode": "evidence",
                "trigger_override": True,
                "kill_active": False,
            }
        return {
            "engine": "aggregation",
            "mode": "standard",
            "sub_mode": None,
            "trigger_override": False,
            "kill_active": False,
        }

    # Research/Experimental mode — route by sub_mode
    if mode in ("research", "experimental"):
        sub = sub_mode.lower() if sub_mode else "debate"
        
        engine_map = {
            "debate": "debate",
            "evidence": "evidence",
            "glass": "glass",
            "stress": "stress",
        }

        return {
            "engine": engine_map.get(sub, "debate"),
            "mode": "research",
            "sub_mode": sub,
            "trigger_override": False,
            "kill_active": False,
        }

    # Fallback — standard aggregation
    return {
        "engine": "aggregation",
        "mode": "standard",
        "sub_mode": None,
        "trigger_override": False,
        "kill_active": False,
    }
