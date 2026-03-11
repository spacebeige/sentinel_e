"""
============================================================
Glass Mode Pipeline — Reasoning Transparency Engine
============================================================

Transforms raw MCO ensemble output into Glass Mode data for the
frontend GlassView component.

GlassView expects:
    assessments[]:  Per-model forensic assessments with metric dimensions
    tactical_map:   Model profiles with strengths/weaknesses
    overall_trust:  Aggregated trust score (0-1)
    consensus_risk: 'LOW' | 'MEDIUM' | 'HIGH'
    reasoning_graph: Graph-RAG nodes/edges for visualization

This module bridges the gap between what MCO produces (scoring
breakdowns, divergence metrics) and what GlassView needs.
============================================================
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("GlassPipeline")


def build_glass_result(
    all_results: list,
    scoring_breakdown: list,
    divergence_metrics: dict,
    aggregated_answer: str,
    winning_model: str,
    drift_score: float,
    volatility_score: float,
) -> Dict[str, Any]:
    """
    Build the audit_result payload that GlassView.js consumes.

    Args:
        all_results: List of ScoredModelResult from orchestrator
        scoring_breakdown: List of ModelScore from orchestrator
        divergence_metrics: Dict from orchestrator
        aggregated_answer: Winning answer text
        winning_model: Name of winning model
        drift_score: Session drift
        volatility_score: Volatility across models
    """
    assessments = []
    model_scores_map: Dict[str, dict] = {}

    # Build per-model score lookup
    for s in (scoring_breakdown or []):
        model_scores_map[s.model_name] = {
            "topic_alignment": s.topic_alignment,
            "knowledge_grounding": s.knowledge_grounding,
            "specificity": s.specificity,
            "confidence_calibration": s.confidence_calibration,
            "drift_penalty": s.drift_penalty,
            "final_score": s.final_score,
        }

    # Build assessments — each model is audited
    valid_results = [
        r for r in (all_results or [])
        if r.output.success and r.output.raw_output and r.output.raw_output.strip()
    ]

    for i, r in enumerate(valid_results):
        model_name = r.output.model_name
        scores = model_scores_map.get(model_name, {})
        final_score = scores.get("final_score", 0.5)

        # Map MCO scores to Glass metric dimensions
        topic = scores.get("topic_alignment", 0.5)
        knowledge = scores.get("knowledge_grounding", 0.5)
        specificity = scores.get("specificity", 0.5)
        confidence_cal = scores.get("confidence_calibration", 0.5)
        drift_pen = scores.get("drift_penalty", 0.0)

        # Derive Glass-specific metrics from MCO scores
        logical_coherence = (topic + specificity) / 2
        hidden_assumptions = max(0, 1.0 - knowledge - 0.1)  # Lower knowledge → more assumptions
        bias_patterns = max(0, drift_pen * 2)  # Drift indicates bias
        confidence_inflation = max(0, abs(confidence_cal - final_score))
        persuasion_tactics = max(0, 0.3 - specificity) * 2  # Vague = more persuasion attempts
        evidence_quality = knowledge
        completeness = (topic + specificity + knowledge) / 3

        # Trust score: weighted composite
        trust_score = (
            0.25 * logical_coherence
            + 0.15 * (1 - hidden_assumptions)
            + 0.15 * (1 - bias_patterns)
            + 0.10 * (1 - confidence_inflation)
            + 0.10 * (1 - persuasion_tactics)
            + 0.15 * evidence_quality
            + 0.10 * completeness
        )
        trust_score = max(0.05, min(trust_score, 0.99))

        # Determine strengths and weaknesses
        strong_points = []
        weak_points = []
        red_flags = []

        if logical_coherence > 0.7:
            strong_points.append("Strong logical structure")
        if evidence_quality > 0.7:
            strong_points.append("Well-grounded in knowledge")
        if completeness > 0.7:
            strong_points.append("Comprehensive coverage")
        if specificity > 0.7:
            strong_points.append("Specific and precise claims")

        if hidden_assumptions > 0.5:
            weak_points.append("Relies on unstated assumptions")
        if bias_patterns > 0.3:
            weak_points.append("Shows potential reasoning bias")
        if confidence_inflation > 0.3:
            weak_points.append("Confidence may be inflated")
        if completeness < 0.4:
            red_flags.append("Incomplete analysis")
        if persuasion_tactics > 0.4:
            red_flags.append("Uses persuasion over evidence")

        # Use other models as auditors in rotation
        auditor_idx = (i + 1) % len(valid_results) if len(valid_results) > 1 else i
        auditor_name = valid_results[auditor_idx].output.model_name

        assessments.append({
            "auditor_name": auditor_name,
            "subject_name": model_name,
            "trust_score": round(trust_score, 4),
            "logical_coherence": round(logical_coherence, 4),
            "hidden_assumptions": round(hidden_assumptions, 4),
            "bias_patterns": round(bias_patterns, 4),
            "confidence_inflation": round(confidence_inflation, 4),
            "persuasion_tactics": round(persuasion_tactics, 4),
            "evidence_quality": round(evidence_quality, 4),
            "completeness": round(completeness, 4),
            "strong_points": strong_points,
            "weak_points": weak_points,
            "red_flags": red_flags,
            "raw_output_preview": (r.output.raw_output or "")[:200],
        })

    # Overall trust = weighted average
    if assessments:
        overall_trust = sum(a["trust_score"] for a in assessments) / len(assessments)
    else:
        overall_trust = 0.5

    # Consensus risk from divergence + volatility
    max_div = (divergence_metrics or {}).get("max_divergence", 0)
    if max_div > 0.5 or volatility_score > 0.4:
        consensus_risk = "HIGH"
    elif max_div > 0.25 or volatility_score > 0.2:
        consensus_risk = "MEDIUM"
    else:
        consensus_risk = "LOW"

    # Tactical map — model-level profiles for graph viz
    model_profiles = {}
    for a in assessments:
        model_profiles[a["subject_name"]] = {
            "trust": a["trust_score"],
            "coherence": a["logical_coherence"],
            "evidence": a["evidence_quality"],
            "bias_risk": a["bias_patterns"],
            "completeness": a["completeness"],
        }

    # Reasoning graph (Graph-RAG nodes + edges)
    graph_nodes = [
        {"id": "query", "type": "query", "label": "Query", "size": 20},
    ]
    graph_edges = []

    for idx, a in enumerate(assessments):
        node_id = f"model_{idx}"
        graph_nodes.append({
            "id": node_id,
            "type": "model",
            "label": a["subject_name"],
            "trust": a["trust_score"],
            "size": 10 + int(a["trust_score"] * 15),
        })
        graph_edges.append({
            "source": "query",
            "target": node_id,
            "type": "responds_to",
            "weight": a["trust_score"],
        })

    # Add consensus node
    if len(assessments) > 1:
        graph_nodes.append({
            "id": "consensus",
            "type": "consensus",
            "label": f"Consensus ({winning_model})",
            "trust": overall_trust,
            "size": 18,
        })
        for idx in range(len(assessments)):
            graph_edges.append({
                "source": f"model_{idx}",
                "target": "consensus",
                "type": "contributes",
                "weight": assessments[idx]["trust_score"],
            })

    # Cross-audit edges (model → model)
    for idx, a in enumerate(assessments):
        auditor_idx_val = next(
            (j for j, b in enumerate(assessments) if b["subject_name"] == a["auditor_name"]),
            None,
        )
        if auditor_idx_val is not None and auditor_idx_val != idx:
            graph_edges.append({
                "source": f"model_{auditor_idx_val}",
                "target": f"model_{idx}",
                "type": "audits",
                "weight": 0.5,
            })

    tactical_map = {
        "model_profiles": model_profiles,
        "winning_model": winning_model,
        "divergence": max_div,
        "volatility": volatility_score,
    }

    return {
        "assessments": assessments,
        "tactical_map": tactical_map,
        "overall_trust": round(overall_trust, 4),
        "consensus_risk": consensus_risk,
        "reasoning_graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
        "models_audited": len(assessments),
        "drift_score": round(drift_score, 4),
    }
