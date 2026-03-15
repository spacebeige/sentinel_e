"""
============================================================
Evidence Mode Pipeline — Sentinel-E v3
============================================================

Bridges EvidenceEngine search results + MCO model scoring into
the format that frontend EvidenceView.js expects.

EvidenceView expects:
    all_claims[]:         Extracted claims with verdict/agreement
    contradictions[]:     Cross-source contradictions with severity
    bayesian_confidence:  Overall evidence confidence (0-1)
    agreement_score:      Inter-source agreement (0-1)
    source_reliability_avg: Average source quality (0-1)
    sources[]:            Evidence sources with url, title, snippet, reliability
    phase_log[]:          Pipeline execution log
    verbatim_citations[]: Direct quotes from sources
    raw_output:           Raw model output
    refined_output:       Synthesised answer incorporating evidence

This module calls EvidenceEngine for real web search, then
combines results with MCO model output.
============================================================
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("EvidencePipeline")

CLAIM_VISIBILITY_THRESHOLD = 0.20  # Only show claims with ≥20% confidence

_SPECULATIVE_PHRASES = frozenset({
    "might indicate", "could suggest", "possibly means",
    "may be", "it seems", "perhaps",
})


async def build_evidence_result(
    query: str,
    all_results: list,
    scoring_breakdown: list,
    aggregated_answer: str,
    winning_model: str,
) -> Dict[str, Any]:
    """
    Execute evidence pipeline and build forensic_result for EvidenceView.

    1) Run EvidenceEngine search (Tavily)
    2) Extract claims from model outputs
    3) Cross-reference claims with evidence sources
    4) Build frontend-compatible payload
    """
    phase_log = []

    # Step 1: Evidence search
    phase_log.append({
        "phase": "search",
        "name": "Evidence Search",
        "status": "running",
        "detail": f"Searching for evidence: {query[:80]}",
    })

    evidence_result = None
    try:
        from core.evidence_engine import EvidenceEngine
        engine = EvidenceEngine()
        evidence_result = await engine.search_evidence(
            query=query,
            max_results=5,
            search_depth="advanced",
        )
        phase_log[-1]["status"] = "complete"
        phase_log[-1]["detail"] = f"Found {len(evidence_result.sources)} sources"
    except Exception as e:
        logger.warning(f"Evidence search failed: {e}")
        phase_log[-1]["status"] = "failed"
        phase_log[-1]["detail"] = str(e)

    # Step 2: Extract claims from model outputs
    phase_log.append({
        "phase": "claim_extraction",
        "name": "Claim Extraction",
        "status": "running",
        "detail": "Extracting claims from model outputs",
    })

    valid_results = [
        r for r in (all_results or [])
        if r.output.success and r.output.raw_output and r.output.raw_output.strip()
    ]

    all_claims = []
    for r in valid_results:
        model_name = r.output.model_name
        text = r.output.raw_output or ""
        sentences = [
            s.strip() for s in text.replace("\n", ". ").split(".")
            if len(s.strip()) > 20
            and not any(p in s.lower() for p in _SPECULATIVE_PHRASES)
        ]

        for sentence in sentences[:10]:  # Max 10 claims per model
            # Simple claim scoring based on evidence match
            evidence_match = 0.0
            matching_source = None
            if evidence_result and evidence_result.sources:
                for src in evidence_result.sources:
                    # Simple keyword overlap check
                    claim_words = set(sentence.lower().split())
                    source_words = set(src.content_snippet.lower().split())
                    overlap = len(claim_words & source_words - _STOP_WORDS)
                    if overlap > 3:
                        evidence_match = min(overlap / 10, 1.0)
                        matching_source = src.url
                        break

            verdict = "confirmed" if evidence_match > 0.4 else "unverified" if evidence_match > 0.1 else "unsupported"

            all_claims.append({
                "statement": sentence[:200],
                "model_origin": model_name,
                "final_confidence": round(evidence_match, 3),
                "date": None,
                "source_type": "model_output",
                "agreement_count": 1 if evidence_match > 0.3 else 0,
                "contradiction_count": 0,
                "verifications": [{
                    "verdict": verdict,
                    "verifier": "evidence_search",
                    "confidence": round(evidence_match, 3),
                }] if matching_source else [],
                "source_url": matching_source,
                "agreement": round(r.score.final_score, 3) if hasattr(r, 'score') else 0.5,
            })

    visible_claims = [c for c in all_claims if c.get("final_confidence", 0) >= CLAIM_VISIBILITY_THRESHOLD]

    phase_log[-1]["status"] = "complete"
    phase_log[-1]["detail"] = f"Extracted {len(all_claims)} claims from {len(valid_results)} models ({len(visible_claims)} visible)"

    # Step 3: Format sources
    sources = []
    if evidence_result:
        for src in evidence_result.sources:
            sources.append({
                "url": src.url,
                "title": src.title,
                "snippet": src.content_snippet[:300],
                "reliability": round(src.reliability_score, 3),
                "domain": src.domain,
                "timestamp": src.retrieved_at,
            })

    # Step 4: Format contradictions
    contradictions = []
    if evidence_result:
        for c in evidence_result.contradictions:
            contradictions.append({
                "type": "factual",
                "severity": round(c.severity, 3),
                "model_a": c.source_a_url,
                "claim_a": c.claim_a[:200],
                "model_b": c.source_b_url,
                "claim_b": c.claim_b[:200],
            })

    # Step 5: Build verbatim citations from sources
    verbatim_citations = []
    if evidence_result:
        for src in evidence_result.sources[:5]:
            if src.content_snippet and len(src.content_snippet) > 30:
                verbatim_citations.append({
                    "quote": src.content_snippet[:250],
                    "source": src.title,
                    "url": src.url,
                    "reliability": round(src.reliability_score, 3),
                    "model_source": None,
                })

    # Step 6: Compute aggregate metrics
    bayesian_confidence = 0.5
    agreement_score = 0.0
    source_reliability_avg = 0.0

    if evidence_result:
        bayesian_confidence = evidence_result.evidence_confidence
        agreement_score = evidence_result.source_agreement
        if evidence_result.sources:
            source_reliability_avg = (
                sum(s.reliability_score for s in evidence_result.sources)
                / len(evidence_result.sources)
            )

    phase_log.append({
        "phase": "synthesis",
        "name": "Evidence Synthesis",
        "status": "complete",
        "detail": (
            f"Confidence: {bayesian_confidence:.1%}, "
            f"Agreement: {agreement_score:.1%}, "
            f"Sources: {len(sources)}"
        ),
    })

    return {
        "all_claims": all_claims,
        "visible_claims": visible_claims,
        "contradictions": contradictions,
        "bayesian_confidence": round(bayesian_confidence, 4),
        "agreement_score": round(agreement_score, 4),
        "source_reliability_avg": round(source_reliability_avg, 4),
        "sources": sources,
        "evidence_sources": sources,
        "phase_log": phase_log,
        "verbatim_citations": verbatim_citations,
        "raw_output": aggregated_answer,
        "refined_output": aggregated_answer,
        "search_executed": evidence_result.search_executed if evidence_result else False,
        "source_count": len(sources),
        "claim_count": len(all_claims),
        "winning_model": winning_model,
    }


# Stop words for claim matching
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for",
    "and", "or", "on", "at", "by", "it", "this", "that", "with", "from",
    "as", "be", "has", "have", "had", "not", "but", "can", "will", "do",
    "does", "did", "been", "being", "its", "they", "their", "them", "we",
    "our", "you", "your", "he", "she", "his", "her", "i", "my", "me",
})
