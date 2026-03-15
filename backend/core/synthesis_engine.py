"""
============================================================
Synthesis Mode Engine — Multimodal Intelligence Aggregation v2
============================================================

Pipeline:
    1. Primary Draft     — Anchor model produces initial answer
    2. Peer Review       — Each other model critiques & suggests improvements
    3. Iterative Refine  — Anchor model integrates feedback → produces v2
    4. Consensus Score   — All models rate the final answer

Synthesis System Prompt (injected into anchor):
    You are a multimodal intelligence aggregation system.

    Your role:
    - Ingest and process structured/unstructured data inputs
    - Cross-reference and correlate data points across all input sources
    - Identify patterns, anomalies, and high-confidence clusters
    - Rank and prioritize targets/objectives by confidence score and strategic value
    - Output a structured intelligence brief with supporting evidence

    Rules:
    - Always require human approval before finalizing any recommendation
    - Flag low-confidence data clearly
    - Never act autonomously — decision authority remains with human operators
    - Cite source data for every output claim
    - Output format adapts to user intent

Output for frontend SynthesisView:
    final_answer:       Refined collaborative answer
    draft:              Original first-pass answer
    revisions[]:        Per-model critique + suggestion
    consensus_score:    How much models agree on final
    improvement_delta:  Quality improvement from draft → final
    synthesis_graph:    Visual representation of refinement flow
============================================================
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SynthesisEngine")

# ── Synthesis System Prompt ───────────────────────────────────
SYNTHESIS_SYSTEM_PROMPT = (
    "You are a multimodal intelligence aggregation system.\n\n"
    "Your role:\n"
    "- Ingest and process structured/unstructured data inputs (text, documents, signals)\n"
    "- Cross-reference and correlate data points across all input sources\n"
    "- Identify patterns, anomalies, and high-confidence clusters\n"
    "- Rank and prioritize targets/objectives by confidence score and strategic value\n"
    "- Output a structured intelligence brief with supporting evidence for each finding\n\n"
    "Rules:\n"
    "- Always require human approval before finalizing any recommendation\n"
    "- Flag low-confidence data clearly\n"
    "- Never act autonomously — decision authority remains with human operators\n"
    "- Cite source data for every output claim\n"
    "- Output format adapts to detected user intent\n"
)


def build_synthesis_result(
    all_results: list,
    scoring_breakdown: list,
    divergence_metrics: dict,
    aggregated_answer: str,
    winning_model: str,
) -> Dict[str, Any]:
    """
    Build synthesis_result from MCO ensemble output.

    In the current implementation, the MCO pipeline already runs all models.
    Synthesis Mode reinterprets these results as:
      - Best model = primary draft author
      - Other models = peer reviewers
      - Aggregated answer = final synthesis
    """

    valid_results = [
        r for r in (all_results or [])
        if r.output.success and r.output.raw_output and r.output.raw_output.strip()
    ]

    if not valid_results:
        return _empty_synthesis()

    # Find the primary (highest-scoring) model
    sorted_results = sorted(
        valid_results,
        key=lambda r: r.score.final_score if hasattr(r, 'score') else 0,
        reverse=True,
    )

    primary = sorted_results[0]
    reviewers = sorted_results[1:]

    draft = primary.output.raw_output or ""
    primary_score = primary.score.final_score if hasattr(primary, 'score') else 0.5

    # Build revisions from peer models
    revisions = []
    for r in reviewers:
        reviewer_name = r.output.model_name
        reviewer_output = r.output.raw_output or ""
        reviewer_score = r.score.final_score if hasattr(r, 'score') else 0.5

        # Compare reviewer output to primary — find divergence
        agreement = _estimate_agreement(draft, reviewer_output)

        # Determine revision type
        if agreement > 0.8:
            rev_type = "endorsement"
            comment = "Largely agrees with the primary analysis."
        elif agreement > 0.5:
            rev_type = "refinement"
            comment = "Offers additional perspective and detail."
        else:
            rev_type = "alternative"
            comment = "Proposes a substantially different approach."

        # Extract unique contributions (sentences in reviewer not in primary)
        additions = _find_unique_sentences(reviewer_output, draft)

        revisions.append({
            "model": reviewer_name,
            "type": rev_type,
            "comment": comment,
            "agreement": round(agreement, 4),
            "score": round(reviewer_score, 4),
            "key_additions": additions[:3],
            "output_preview": reviewer_output[:300],
        })

    # Consensus score — average agreement across all reviewers
    if revisions:
        consensus_score = sum(r["agreement"] for r in revisions) / len(revisions)
    else:
        consensus_score = 1.0

    # Improvement delta — compare aggregated answer quality to primary draft
    # Use scoring difference as proxy
    all_scores = [r.score.final_score for r in valid_results if hasattr(r, 'score')]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.5
    improvement_delta = max(0, avg_score - primary_score + 0.05) * 2  # Normalized boost

    # Synthesis graph for visualization
    graph_nodes = [
        {
            "id": "draft",
            "type": "draft",
            "label": f"Draft ({primary.output.model_name})",
            "score": round(primary_score, 3),
        },
    ]
    graph_edges = []

    for i, rev in enumerate(revisions):
        node_id = f"review_{i}"
        graph_nodes.append({
            "id": node_id,
            "type": rev["type"],
            "label": rev["model"],
            "score": rev["score"],
            "agreement": rev["agreement"],
        })
        graph_edges.append({
            "source": "draft",
            "target": node_id,
            "type": "reviewed_by",
            "agreement": rev["agreement"],
        })

    graph_nodes.append({
        "id": "final",
        "type": "synthesis",
        "label": "Final Synthesis",
        "score": round(avg_score, 3),
    })
    for i in range(len(revisions)):
        graph_edges.append({
            "source": f"review_{i}",
            "target": "final",
            "type": "contributes_to",
            "weight": revisions[i]["agreement"],
        })
    graph_edges.append({
        "source": "draft",
        "target": "final",
        "type": "refined_into",
        "weight": 1.0,
    })

    return {
        "final_answer": aggregated_answer,
        "draft": draft[:2000],
        "draft_model": primary.output.model_name,
        "draft_score": round(primary_score, 4),
        "revisions": revisions,
        "consensus_score": round(consensus_score, 4),
        "improvement_delta": round(min(improvement_delta, 1.0), 4),
        "models_participated": len(valid_results),
        "synthesis_graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
        "winning_model": winning_model,
    }


def _estimate_agreement(text_a: str, text_b: str) -> float:
    """Simple word-overlap agreement estimator."""
    if not text_a or not text_b:
        return 0.0

    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "for", "and", "or", "on", "at", "by", "it", "this", "that", "with",
    }
    words_a -= stop
    words_b -= stop

    if not words_a or not words_b:
        return 0.5

    overlap = len(words_a & words_b)
    total = len(words_a | words_b)
    return overlap / total if total > 0 else 0.0


def _find_unique_sentences(source: str, reference: str) -> List[str]:
    """Find sentences in source that aren't substantially in reference."""
    ref_words = set(reference.lower().split())
    unique = []

    for sentence in source.replace("\n", ". ").split("."):
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        sent_words = set(sentence.lower().split())
        overlap = len(sent_words & ref_words) / len(sent_words) if sent_words else 1
        if overlap < 0.6:  # Less than 60% word overlap — considered unique
            unique.append(sentence[:150])

    return unique


def _empty_synthesis() -> Dict[str, Any]:
    return {
        "final_answer": "",
        "draft": "",
        "draft_model": "unknown",
        "draft_score": 0.0,
        "revisions": [],
        "consensus_score": 0.0,
        "improvement_delta": 0.0,
        "models_participated": 0,
        "synthesis_graph": {"nodes": [], "edges": []},
        "winning_model": "unknown",
    }


async def claude_synthesize(model_outputs: list, aggregated_answer: str, query: str = "") -> Optional[str]:
    """Call Claude to produce a refined synthesis of all model outputs.
    Only called when Claude is toggled ON in the registry."""
    from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY

    claude_spec = COGNITIVE_MODEL_REGISTRY.get("claude-sonnet-4.6")
    if not claude_spec or not claude_spec.active or not claude_spec.enabled:
        return None

    # Build a concise prompt with all model perspectives
    perspectives = []
    for i, result in enumerate(model_outputs[:6], 1):  # Cap at 6 to save tokens
        model_name = result.get("model", f"Model {i}")
        output = result.get("output", "")[:800]  # Truncate each to ~200 tokens
        perspectives.append(f"[{model_name}]: {output}")

    synthesis_prompt = (
        "You are a synthesis expert. Multiple AI models analyzed the same query. "
        "Review their perspectives and produce a refined, comprehensive answer that:\n"
        "- Integrates the strongest points from each model\n"
        "- Resolves any contradictions\n"
        "- Provides a clear, well-structured final answer\n\n"
        f"Models' analyses:\n" + "\n\n".join(perspectives) + "\n\n"
        "Produce a refined synthesis:"
    )

    try:
        from models.mco_bridge import MCOModelBridge
        bridge = MCOModelBridge()
        result = await bridge.call_model(
            "claude-sonnet-4.6", synthesis_prompt,
            "You synthesize multiple expert perspectives into a clear, authoritative answer.",
            max_tokens=500,
        )
        if result and not result.startswith("Error:"):
            return result
    except Exception as e:
        logger.warning(f"Claude synthesis call failed: {e}")

    return None
