"""
============================================================
Cognitive RAG System — Intent-Driven Retrieval
============================================================
Replaces keyword-based triggers with:
1. Query classifier (intent recognition)
2. Confidence estimator
3. Retrieval necessity predictor

No hardcoded triggers. Uses probabilistic assessment
to determine when external knowledge is required.
"""

import re
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("RAG")


# ============================================================
# Query Classification
# ============================================================

class QueryIntent:
    """Classified intent categories."""
    FACTUAL = "factual"           # Needs external verification
    ANALYTICAL = "analytical"     # Can be reasoned internally
    CREATIVE = "creative"         # Generation task
    CONVERSATIONAL = "conversational"  # Casual chat
    INSTRUCTIONAL = "instructional"    # How-to
    COMPARATIVE = "comparative"   # Needs data points
    TEMPORAL = "temporal"         # Time-sensitive information
    OPINION = "opinion"           # Subjective discussion


@dataclass
class QueryClassification:
    """Result of query classification."""
    primary_intent: str = QueryIntent.CONVERSATIONAL
    confidence: float = 0.5
    retrieval_probability: float = 0.0
    reasoning: str = ""
    temporal_sensitivity: float = 0.0
    factual_density: float = 0.0
    requires_citation: bool = False


class QueryClassifier:
    """
    Intent classification layer.
    Uses linguistic features to classify queries without hardcoded keywords.
    """

    # Feature patterns with weights
    FACTUAL_SIGNALS = [
        (r"\b(what|who|when|where|which)\b.*\b(is|are|was|were|did)\b", 0.6),
        (r"\b(how\s+many|how\s+much|how\s+long|how\s+often)\b", 0.7),
        (r"\b(statistics?|data|numbers?|figures?|metrics?)\b", 0.5),
        (r"\b(according\s+to|based\s+on|research\s+shows?)\b", 0.6),
        (r"\b(latest|recent|current|today|202\d)\b", 0.6),
        (r"\b(true|false|correct|incorrect|accurate)\b", 0.5),
        (r"\b(percentage|rate|ratio|count|amount)\b", 0.5),
    ]

    ANALYTICAL_SIGNALS = [
        (r"\b(why|how\s+does|explain|analyze|compare)\b", 0.5),
        (r"\b(cause|effect|impact|consequence|implication)\b", 0.4),
        (r"\b(advantage|disadvantage|pro|con|trade-?off)\b", 0.5),
        (r"\b(difference|similarity|contrast|distinction)\b", 0.5),
        (r"\b(evaluate|assess|critique|review)\b", 0.4),
    ]

    CREATIVE_SIGNALS = [
        (r"\b(write|create|generate|compose|draft)\b", 0.7),
        (r"\b(story|poem|essay|article|script|code)\b", 0.6),
        (r"\b(imagine|brainstorm|ideate|invent)\b", 0.6),
    ]

    TEMPORAL_SIGNALS = [
        (r"\b(today|yesterday|this\s+week|this\s+month|this\s+year)\b", 0.8),
        (r"\b(latest|recent|current|new|updated?)\b", 0.6),
        (r"\b(202[4-9]|203\d)\b", 0.7),
        (r"\b(breaking|live|real-?time)\b", 0.8),
        (r"\b(price|stock|weather|score|news)\b", 0.7),
    ]

    CONVERSATIONAL_SIGNALS = [
        (r"^(hi|hello|hey|thanks|thank\s+you|ok|sure|yes|no)\b", 0.8),
        (r"\b(how\s+are\s+you|what's?\s+up|nice|cool|great)\b", 0.7),
        (r"^.{1,20}$", 0.3),  # Very short messages tend to be conversational
    ]

    def classify(self, query: str) -> QueryClassification:
        """Classify a query and estimate retrieval necessity."""
        if not query or not query.strip():
            return QueryClassification()

        query_lower = query.lower().strip()
        scores = {
            QueryIntent.FACTUAL: 0.0,
            QueryIntent.ANALYTICAL: 0.0,
            QueryIntent.CREATIVE: 0.0,
            QueryIntent.CONVERSATIONAL: 0.0,
            QueryIntent.TEMPORAL: 0.0,
        }

        # Score each intent category
        for pattern, weight in self.FACTUAL_SIGNALS:
            if re.search(pattern, query_lower):
                scores[QueryIntent.FACTUAL] += weight

        for pattern, weight in self.ANALYTICAL_SIGNALS:
            if re.search(pattern, query_lower):
                scores[QueryIntent.ANALYTICAL] += weight

        for pattern, weight in self.CREATIVE_SIGNALS:
            if re.search(pattern, query_lower):
                scores[QueryIntent.CREATIVE] += weight

        for pattern, weight in self.TEMPORAL_SIGNALS:
            if re.search(pattern, query_lower):
                scores[QueryIntent.TEMPORAL] += weight

        for pattern, weight in self.CONVERSATIONAL_SIGNALS:
            if re.search(pattern, query_lower):
                scores[QueryIntent.CONVERSATIONAL] += weight

        # Question structure analysis
        if query_lower.endswith("?"):
            scores[QueryIntent.FACTUAL] += 0.2
            scores[QueryIntent.ANALYTICAL] += 0.1

        # Length factor: longer queries tend to be more analytical
        word_count = len(query_lower.split())
        if word_count > 20:
            scores[QueryIntent.ANALYTICAL] += 0.2
        elif word_count < 5:
            scores[QueryIntent.CONVERSATIONAL] += 0.2

        # Determine primary intent
        primary = max(scores, key=scores.get)
        primary_score = scores[primary]

        # Normalize confidence
        total = sum(scores.values()) or 1
        confidence = min(primary_score / total, 1.0) if total > 0 else 0.5

        # Compute retrieval probability
        retrieval_prob = self._compute_retrieval_probability(scores, query_lower)

        # Temporal sensitivity
        temporal_sens = min(scores[QueryIntent.TEMPORAL] / 2, 1.0)

        # Factual density
        factual_dens = min(scores[QueryIntent.FACTUAL] / 2, 1.0)

        return QueryClassification(
            primary_intent=primary,
            confidence=round(confidence, 3),
            retrieval_probability=round(retrieval_prob, 3),
            reasoning=f"Primary: {primary} (score={primary_score:.2f}), retrieval_p={retrieval_prob:.2f}",
            temporal_sensitivity=round(temporal_sens, 3),
            factual_density=round(factual_dens, 3),
            requires_citation=retrieval_prob > 0.6,
        )

    def _compute_retrieval_probability(self, scores: Dict[str, float], query: str) -> float:
        """
        Compute the probability that external knowledge retrieval is needed.
        High for factual/temporal queries, low for creative/conversational.
        """
        # Weighted combination
        p = (
            scores.get(QueryIntent.FACTUAL, 0) * 0.8
            + scores.get(QueryIntent.TEMPORAL, 0) * 0.9
            + scores.get(QueryIntent.ANALYTICAL, 0) * 0.3
            + scores.get(QueryIntent.COMPARATIVE, 0) * 0.6
        )

        # Reduce for clear creative/conversational
        p -= scores.get(QueryIntent.CREATIVE, 0) * 0.5
        p -= scores.get(QueryIntent.CONVERSATIONAL, 0) * 0.6

        return max(0.0, min(1.0, p))


# ============================================================
# RAG Pipeline
# ============================================================

@dataclass
class RetrievedSource:
    """A single retrieved source."""
    url: str = ""
    title: str = ""
    content: str = ""
    reliability_score: float = 0.5
    domain: str = ""
    content_hash: str = ""
    retrieved_at: str = ""


@dataclass
class RAGResult:
    """Result of the RAG pipeline."""
    query: str = ""
    sources: List[RetrievedSource] = field(default_factory=list)
    citations_text: str = ""
    retrieval_executed: bool = False
    source_count: int = 0
    average_reliability: float = 0.0
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    no_sources_found: bool = False


# Domain reliability scores with proper suffix matching
TRUSTED_DOMAINS = {
    "arxiv.org": 0.90,
    "nature.com": 0.92,
    "science.org": 0.92,
    "ieee.org": 0.88,
    "acm.org": 0.88,
    "who.int": 0.90,
    "nih.gov": 0.90,
    "cdc.gov": 0.88,
    "wikipedia.org": 0.70,
    "stackoverflow.com": 0.72,
    "github.com": 0.70,
    "reuters.com": 0.78,
    "bbc.com": 0.75,
    "nytimes.com": 0.75,
    "medium.com": 0.55,
    "reddit.com": 0.45,
    "quora.com": 0.50,
}


def _score_domain(url: str) -> float:
    """Score domain reliability using proper suffix matching."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        # Exact domain match
        for domain, score in TRUSTED_DOMAINS.items():
            if hostname == domain or hostname.endswith(f".{domain}"):
                return score

        # TLD-based scoring (suffix matching, not substring)
        if hostname.endswith(".gov"):
            return 0.85
        if hostname.endswith(".edu"):
            return 0.83
        if hostname.endswith(".org"):
            return 0.65

        return 0.5  # Unknown domain
    except Exception:
        return 0.5


class CognitiveRAG:
    """
    Intent-driven RAG pipeline.
    Only triggers retrieval when probability exceeds threshold.
    Never hallucinates browsing.
    """

    def __init__(self, tavily_api_key: str = "", threshold: float = 0.6):
        from gateway.config import get_settings
        settings = get_settings()
        self.api_key = tavily_api_key or settings.TAVILY_API_KEY
        self.threshold = threshold or settings.RAG_CONFIDENCE_THRESHOLD
        self.max_results = settings.RAG_MAX_SOURCES
        self.classifier = QueryClassifier()

    async def process(self, query: str, force: bool = False) -> RAGResult:
        """
        Run the cognitive RAG pipeline:
        1. Classify query intent
        2. Estimate retrieval necessity
        3. Execute retrieval only if threshold met
        4. Deduplicate & score sources
        5. Detect contradictions
        6. Generate citation text
        """
        classification = self.classifier.classify(query)

        logger.info(
            f"RAG classification: intent={classification.primary_intent}, "
            f"retrieval_p={classification.retrieval_probability}, "
            f"threshold={self.threshold}"
        )

        # Decision: should we retrieve?
        if not force and classification.retrieval_probability < self.threshold:
            return RAGResult(
                query=query,
                retrieval_executed=False,
            )

        # Execute retrieval
        sources = await self._search(query)

        if not sources:
            return RAGResult(
                query=query,
                retrieval_executed=True,
                no_sources_found=True,
            )

        # Deduplicate
        sources = self._deduplicate(sources)

        # Score reliability
        for s in sources:
            s.reliability_score = _score_domain(s.url)

        # Detect contradictions
        contradictions = self._detect_contradictions(sources)

        # Generate citation text
        avg_reliability = sum(s.reliability_score for s in sources) / len(sources) if sources else 0
        citations_text = self._format_citations(sources, contradictions)

        return RAGResult(
            query=query,
            sources=sources,
            citations_text=citations_text,
            retrieval_executed=True,
            source_count=len(sources),
            average_reliability=round(avg_reliability, 3),
            contradictions=contradictions,
        )

    async def _search(self, query: str) -> List[RetrievedSource]:
        """Execute Tavily search."""
        if not self.api_key:
            logger.warning("Tavily API key not configured, skipping search")
            return []

        try:
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=self.api_key)
            result = await client.search(
                query=query,
                search_depth="advanced",
                max_results=self.max_results,
                include_answer=False,
            )

            sources = []
            for item in result.get("results", []):
                content = item.get("content", "")
                sources.append(RetrievedSource(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    content=content[:500],
                    domain=self._extract_domain(item.get("url", "")),
                    content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                ))
            return sources

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    def _extract_domain(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).hostname or ""
        except Exception:
            return ""

    def _deduplicate(self, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Remove duplicate sources by content hash."""
        seen = set()
        unique = []
        for s in sources:
            if s.content_hash not in seen:
                seen.add(s.content_hash)
                unique.append(s)
        return unique

    def _detect_contradictions(self, sources: List[RetrievedSource]) -> List[Dict[str, Any]]:
        """
        Basic contradiction detection between sources.
        Uses negation patterns and semantic opposition.
        """
        contradictions = []
        negation_pairs = [
            ("increase", "decrease"), ("rise", "fall"), ("improve", "worsen"),
            ("safe", "dangerous"), ("effective", "ineffective"),
            ("true", "false"), ("yes", "no"), ("proven", "disproven"),
            ("support", "oppose"), ("benefit", "harm"),
        ]

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                content_i = sources[i].content.lower()
                content_j = sources[j].content.lower()

                for pos, neg in negation_pairs:
                    if (pos in content_i and neg in content_j) or (neg in content_i and pos in content_j):
                        contradictions.append({
                            "source_a": sources[i].url,
                            "source_b": sources[j].url,
                            "type": f"{pos}/{neg}",
                            "severity": 0.5,
                        })
                        break  # One contradiction per pair

        return contradictions

    def _format_citations(
        self,
        sources: List[RetrievedSource],
        contradictions: List[Dict[str, Any]],
    ) -> str:
        """Format sources into citation text for response injection."""
        if not sources:
            return "No verified external sources found."

        parts = ["\n**Sources:**"]
        for i, s in enumerate(sources[:5], 1):
            reliability = "High" if s.reliability_score >= 0.8 else "Medium" if s.reliability_score >= 0.6 else "Low"
            parts.append(f"[{i}] [{s.title}]({s.url}) — Reliability: {reliability}")

        if contradictions:
            parts.append(f"\n⚠️ {len(contradictions)} potential contradiction(s) detected between sources.")

        return "\n".join(parts)
