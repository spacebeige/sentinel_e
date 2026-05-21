"""
============================================================
Cognitive Artifact Builder — Sentinel-E v8.0
============================================================
Wraps the existing ensemble response payload and appends a
structured "cognitive artifact" layer.

EXISTING RESPONSE CONTRACTS ARE PRESERVED.
  - formatted_output: unchanged
  - omega_metadata: unchanged structure, only extended
  - All frontend-facing fields: preserved

New field appended: omega_metadata.cognitive_artifact

Cognitive Artifact Structure:
  - primary_conclusion: direct answer
  - reasoning_topology: reasoning chain nodes
  - evidence_matrix: claim → evidence mapping
  - contradiction_analysis: density + unresolved conflicts
  - alternative_perspectives: dissenting model positions
  - verification_results: hallucination gate / verification events
  - confidence_evolution: timeline snapshots
  - reflective_cognition: metacognitive analysis
  - memory_continuity_links: relevant memory retrievals
  - orchestration_identity: OrchestrationRun summary
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CognitiveArtifactBuilder")


def build_cognitive_artifact(
    ensemble_response,           # EnsembleResponse from cognitive_orchestrator.py
    orchestration_run=None,      # OrchestrationRun — optional
    rag_result=None,             # CognitiveRAG result — optional
    reflection: str = "",        # metacognitive reflection text
) -> Dict[str, Any]:
    """
    Build a structured cognitive artifact from an EnsembleResponse.

    This is purely additive: it reads from ensemble_response fields
    that already exist and organizes them into a richer structure.

    Args:
        ensemble_response: The EnsembleResponse from CognitiveCoreEngine.
        orchestration_run: Optional OrchestrationRun for identity context.
        rag_result: Optional RAG result for evidence links.
        reflection: Optional metacognitive reflection string.

    Returns:
        A JSON-safe dict that is appended to omega_metadata['cognitive_artifact'].
    """
    try:
        artifact = {}

        # ── Primary Conclusion ─────────────────────────────────
        artifact["primary_conclusion"] = _extract_primary_conclusion(ensemble_response)

        # ── Reasoning Topology ─────────────────────────────────
        artifact["reasoning_topology"] = _build_reasoning_topology(ensemble_response)

        # ── Evidence Matrix ────────────────────────────────────
        artifact["evidence_matrix"] = _build_evidence_matrix(ensemble_response, rag_result)

        # ── Contradiction Analysis ─────────────────────────────
        artifact["contradiction_analysis"] = _build_contradiction_analysis(ensemble_response)

        # ── Alternative Perspectives ───────────────────────────
        artifact["alternative_perspectives"] = _build_alternative_perspectives(ensemble_response)

        # ── Verification Results ───────────────────────────────
        artifact["verification_results"] = _build_verification_results(ensemble_response)

        # ── Confidence Evolution ───────────────────────────────
        artifact["confidence_evolution"] = _build_confidence_evolution(ensemble_response)

        # ── Reflective Cognition ───────────────────────────────
        artifact["reflective_cognition"] = reflection or _generate_auto_reflection(ensemble_response)

        # ── Memory Continuity Links ────────────────────────────
        artifact["memory_continuity_links"] = _build_memory_links(orchestration_run)

        # ── Orchestration Identity ─────────────────────────────
        if orchestration_run:
            try:
                artifact["orchestration_identity"] = orchestration_run.to_summary()
            except Exception:
                artifact["orchestration_identity"] = {
                    "orchestration_run_id": getattr(orchestration_run, "orchestration_run_id", "unknown")
                }

        # ── Quality Indicators ─────────────────────────────────
        artifact["quality_indicators"] = _build_quality_indicators(ensemble_response)

        return artifact

    except Exception as e:
        logger.warning(f"[CognitiveArtifact] Build failed (non-fatal): {e}")
        return {
            "error": "Cognitive artifact construction failed (non-fatal)",
            "primary_conclusion": getattr(ensemble_response, "formatted_output", "")[:500],
        }


# ── Builder Helpers ────────────────────────────────────────────

def _extract_primary_conclusion(er) -> str:
    """Extract the primary conclusion from the ensemble response."""
    # Try debate consensus first (most authoritative)
    debate = getattr(er, "debate_result", None)
    if debate:
        consensus = getattr(debate, "final_consensus", None)
        if consensus and len(consensus.strip()) > 10:
            return consensus[:500]

    # Fallback: highest confidence model position
    outputs = getattr(er, "model_outputs", []) or []
    successful = [o for o in outputs if getattr(o, "succeeded", False)]
    if successful:
        best = max(successful, key=lambda o: getattr(o, "confidence", 0.0))
        position = getattr(best, "position", "")
        if position:
            return position[:500]

    # Last resort: formatted output
    return (getattr(er, "formatted_output", "") or "")[:500]


def _build_reasoning_topology(er) -> List[Dict[str, Any]]:
    """
    Build reasoning chain nodes from model reasoning fields.
    Each node represents a reasoning step from a model.
    """
    nodes = []
    outputs = getattr(er, "model_outputs", []) or []
    successful = [o for o in outputs if getattr(o, "succeeded", False)]

    for output in successful[:6]:  # Cap at 6 models
        reasoning = getattr(output, "reasoning", "") or ""
        if not reasoning:
            continue
        # Split reasoning into step-like sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reasoning) if len(s.strip()) > 20]
        nodes.append({
            "model_id": getattr(output, "model_id", "unknown"),
            "model_name": getattr(output, "model_name", "unknown"),
            "confidence": round(getattr(output, "confidence", 0.5), 4),
            "reasoning_steps": sentences[:5],  # Top 5 steps per model
            "assumptions": getattr(output, "assumptions", [])[:3],
            "vulnerabilities": getattr(output, "vulnerabilities", [])[:2],
        })

    return nodes


def _build_evidence_matrix(er, rag_result=None) -> List[Dict[str, Any]]:
    """
    Map claims to evidence sources.
    Uses RAG sources if available, otherwise uses model assumptions.
    """
    matrix = []

    # RAG evidence
    if rag_result and getattr(rag_result, "retrieval_executed", False):
        sources = getattr(rag_result, "sources", []) or []
        for source in sources[:5]:
            matrix.append({
                "type": "external_source",
                "title": getattr(source, "title", "Unknown Source"),
                "url": getattr(source, "url", ""),
                "content_preview": getattr(source, "content", "")[:200],
                "reliability": getattr(source, "reliability", 0.5),
            })

    # Model-derived evidence from assumptions
    outputs = getattr(er, "model_outputs", []) or []
    successful = [o for o in outputs if getattr(o, "succeeded", False)]
    for output in successful[:3]:
        for assumption in (getattr(output, "assumptions", []) or [])[:2]:
            if assumption:
                matrix.append({
                    "type": "model_assumption",
                    "model_id": getattr(output, "model_id", "unknown"),
                    "claim": assumption[:200],
                    "confidence": round(getattr(output, "confidence", 0.5), 4),
                })

    return matrix[:10]  # Cap total entries


def _build_contradiction_analysis(er) -> Dict[str, Any]:
    """
    Structured contradiction analysis from ensemble metrics and debate.
    """
    metrics = getattr(er, "ensemble_metrics", None)
    debate = getattr(er, "debate_result", None)

    density = 0.0
    unresolved = []
    resolved = []

    if metrics:
        density = getattr(metrics, "contradiction_density", 0.0) or 0.0

    if debate:
        unresolved = getattr(debate, "unresolved_conflicts", []) or []
        resolved = getattr(debate, "resolved_conflicts", []) if hasattr(debate, "resolved_conflicts") else []

    # Determine contradiction severity label
    if density > 0.7:
        severity = "high"
        summary = "Significant contradictions detected. Multiple models diverge on key claims."
    elif density > 0.4:
        severity = "medium"
        summary = "Moderate contradictions. Debate reduced divergence but some conflicts remain."
    elif density > 0.1:
        severity = "low"
        summary = "Minor contradictions. Models generally converged after debate."
    else:
        severity = "none"
        summary = "High consensus. Models agree on core claims."

    return {
        "density": round(density, 4),
        "severity": severity,
        "summary": summary,
        "unresolved_conflicts": [str(c)[:200] for c in unresolved[:5]],
        "resolved_conflicts": [str(c)[:200] for c in resolved[:3]],
        "requires_verification": density > 0.5,
    }


def _build_alternative_perspectives(er) -> List[Dict[str, Any]]:
    """
    Extract dissenting model positions from the tactical map.
    Shows alternative viewpoints for cognitive transparency.
    """
    perspectives = []
    tactical_map = getattr(er, "tactical_map", None)
    if not tactical_map:
        return perspectives

    entries = getattr(tactical_map, "entries", []) or []
    # Models with low consensus agreement are "alternative" perspectives
    dissenting = sorted(
        [e for e in entries if getattr(e, "agreement_with_consensus", 1.0) < 0.5],
        key=lambda e: getattr(e, "confidence", 0.0),
        reverse=True,
    )
    for entry in dissenting[:3]:
        perspectives.append({
            "model_id": getattr(entry, "model_id", "unknown"),
            "model_name": getattr(entry, "model_name", "unknown"),
            "position": (getattr(entry, "position_summary", "") or "")[:300],
            "confidence": round(getattr(entry, "confidence", 0.0), 4),
            "agreement_with_consensus": round(getattr(entry, "agreement_with_consensus", 0.0), 4),
            "key_differentiator": (getattr(entry, "key_differentiator", "") or "")[:200],
        })
    return perspectives


def _build_verification_results(er) -> Dict[str, Any]:
    """
    Verification summary from ensemble metrics.
    """
    metrics = getattr(er, "ensemble_metrics", None)
    confidence = getattr(er, "confidence", None)

    stability = 0.5
    fragility = 0.5
    calibration_method = "unknown"

    if metrics:
        stability = getattr(metrics, "stability_index", 0.5) or 0.5
        fragility = getattr(metrics, "fragility_score", 0.5) or 0.5

    if confidence:
        calibration_method = getattr(confidence, "calibration_method", "unknown") or "unknown"

    return {
        "stability_index": round(stability, 4),
        "fragility_score": round(fragility, 4),
        "calibration_method": calibration_method,
        "reliability_assessment": (
            "high" if stability > 0.7 and fragility < 0.3 else
            "medium" if stability > 0.4 else
            "low"
        ),
        "verification_passed": stability > 0.5 and fragility < 0.6,
    }


def _build_confidence_evolution(er) -> List[Dict[str, Any]]:
    """
    Format confidence evolution from the ensemble response.
    """
    evolution = []
    conf = getattr(er, "confidence", None)
    if not conf:
        return evolution

    raw_evolution = getattr(conf, "evolution", []) or []
    for snap in raw_evolution[:10]:
        if isinstance(snap, dict):
            evolution.append({
                "phase": snap.get("phase", "unknown"),
                "value": round(float(snap.get("value", 0.0)), 4),
                "method": snap.get("method", ""),
            })

    # If no evolution snapshots, build from conf_evolution dict
    conf_evolution = getattr(er, "confidence_evolution", None)
    if not evolution and isinstance(conf_evolution, dict):
        for phase, value in conf_evolution.items():
            if isinstance(value, (int, float)):
                evolution.append({
                    "phase": phase,
                    "value": round(float(value), 4),
                    "method": "snapshot",
                })

    return evolution


def _generate_auto_reflection(er) -> str:
    """
    Generate a metacognitive reflection from ensemble metrics.
    This is deterministic (no LLM call) — based on measurable outcomes.
    """
    metrics = getattr(er, "ensemble_metrics", None)
    debate = getattr(er, "debate_result", None)
    confidence = getattr(er, "confidence", None)

    parts = []

    if confidence:
        conf_val = getattr(confidence, "final_confidence", 0.5) or 0.5
        if conf_val > 0.8:
            parts.append("The ensemble achieved high confidence, indicating strong cross-model agreement.")
        elif conf_val > 0.5:
            parts.append("The ensemble reached moderate confidence. Some uncertainty remains.")
        else:
            parts.append("Confidence is low. The ensemble encountered significant disagreement or complexity.")

    if metrics:
        density = getattr(metrics, "contradiction_density", 0.0) or 0.0
        stability = getattr(metrics, "stability_index", 0.5) or 0.5
        if density > 0.5:
            parts.append(f"Contradiction density ({density:.2f}) exceeded the stability threshold, suggesting the topic involves genuine epistemic uncertainty.")
        if stability > 0.7:
            parts.append("Debate convergence was strong — positions stabilized across rounds.")

    if debate:
        rounds = getattr(debate, "total_rounds", 0)
        unresolved = getattr(debate, "unresolved_conflicts", []) or []
        if rounds > 0:
            parts.append(f"The {rounds}-round debate process resolved most disagreements.")
        if unresolved:
            parts.append(f"{len(unresolved)} conflict(s) remained unresolved, indicating irreducible uncertainty.")

    return " ".join(parts) if parts else "Ensemble reasoning completed successfully."


def _build_memory_links(orchestration_run=None) -> List[Dict[str, Any]]:
    """Extract memory retrieval records from OrchestrationRun."""
    if not orchestration_run:
        return []
    retrievals = getattr(orchestration_run, "memory_retrievals", []) or []
    return [
        {
            "layer": getattr(r, "layer", "unknown"),
            "key": getattr(r, "key", ""),
            "preview": getattr(r, "content_preview", "")[:100],
            "relevance": round(getattr(r, "relevance_score", 1.0), 4),
        }
        for r in retrievals[:10]
    ]


def _build_quality_indicators(er) -> Dict[str, Any]:
    """High-level quality indicators for the response."""
    models_executed = getattr(er, "models_executed", 0)
    models_succeeded = getattr(er, "models_succeeded", 0)
    debate = getattr(er, "debate_result", None)
    rounds = getattr(debate, "total_rounds", 0) if debate else 0
    metrics = getattr(er, "ensemble_metrics", None)
    stability = getattr(metrics, "stability_index", 0.5) if metrics else 0.5

    # Participation rate
    participation = models_succeeded / max(models_executed, 1)

    return {
        "models_executed": models_executed,
        "models_succeeded": models_succeeded,
        "participation_rate": round(participation, 4),
        "debate_rounds": rounds,
        "stability_index": round(stability, 4),
        "response_grade": (
            "A" if participation > 0.8 and stability > 0.7 and rounds >= 3 else
            "B" if participation > 0.6 and stability > 0.5 else
            "C" if participation > 0.4 else
            "D"
        ),
    }
