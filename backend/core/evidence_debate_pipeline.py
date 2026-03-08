"""
============================================================
Evidence-Integrated Debate Pipeline — Sentinel-E v3
============================================================
Connects Evidence Engine → Debate Engine with Perplexity-style
grounded reasoning.

Pipeline:
    User Prompt
    ↓
    Tavily Search API → Top 5 sources
    ↓
    Source reliability scoring
    ↓
    Evidence summary (injected into debate prompt)
    ↓
    3-Round Debate (with evidence context)
    ↓
    Consensus Engine → Metrics → Visualization

The evidence context is injected into Round 1 prompts so all
models reason over the same factual basis.
============================================================
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("EvidenceDebatePipeline")


@dataclass
class EvidenceContext:
    """Evidence bundle prepared for injection into debate prompts."""
    summary: str = ""
    sources: List[Dict[str, str]] = field(default_factory=list)
    source_count: int = 0
    evidence_confidence: float = 0.0
    search_executed: bool = False

    def to_prompt_injection(self) -> str:
        """Format evidence for injection into debate system prompts."""
        if not self.sources:
            return ""

        parts = [
            "\n[EVIDENCE CONTEXT — Retrieved from web sources]\n",
            f"Evidence confidence: {self.evidence_confidence:.0%}\n",
        ]
        for i, src in enumerate(self.sources[:5], 1):
            parts.append(
                f"\nSource {i}: {src.get('title', 'Unknown')}\n"
                f"URL: {src.get('url', '')}\n"
                f"Excerpt: {src.get('snippet', '')[:200]}\n"
            )
        parts.append(
            "\nUse these sources to ground your reasoning. "
            "Cite specific sources when making factual claims.\n"
        )
        return "".join(parts)

    def to_frontend_dict(self) -> Dict[str, Any]:
        """Format for frontend evidence panel display."""
        return {
            "sources": self.sources,
            "source_count": self.source_count,
            "evidence_confidence": round(self.evidence_confidence, 4),
            "search_executed": self.search_executed,
            "summary": self.summary[:500] if self.summary else "",
        }


async def gather_evidence(query: str, max_results: int = 5) -> EvidenceContext:
    """
    Execute evidence retrieval pipeline.

    Returns an EvidenceContext ready for injection into debate prompts.
    Handles graceful degradation if Tavily is unavailable.
    """
    ctx = EvidenceContext()

    try:
        from core.evidence_engine import EvidenceEngine

        engine = EvidenceEngine()
        result = await engine.search_evidence(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )

        ctx.search_executed = result.search_executed
        ctx.evidence_confidence = result.evidence_confidence

        for source in result.sources:
            ctx.sources.append({
                "title": source.title,
                "url": source.url,
                "snippet": source.content_snippet[:300],
                "reliability": round(source.reliability_score, 3),
                "domain": source.domain,
            })

        ctx.source_count = len(ctx.sources)

        # Build a concise summary from top sources
        if ctx.sources:
            summary_parts = []
            for src in ctx.sources[:3]:
                summary_parts.append(
                    f"[{src['title']}] {src['snippet'][:150]}"
                )
            ctx.summary = " | ".join(summary_parts)

        logger.info(
            f"Evidence gathered: {ctx.source_count} sources, "
            f"confidence={ctx.evidence_confidence:.4f}"
        )
    except Exception as e:
        logger.warning(f"Evidence retrieval failed (non-fatal): {e}")
        ctx.search_executed = False

    return ctx
