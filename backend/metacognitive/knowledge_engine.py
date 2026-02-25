"""
============================================================
API 2 — Knowledge & Retrieval Engine
============================================================
Live + niche knowledge acquisition.

Rules enforced:
  - Store normalized factual blocks (no summarization before storage)
  - Include timestamps
  - Support API + scraping + niche indexing
  - Long-tail expansion mandatory for low-confidence domains
  - Concept expansion for depth > 0

Data flow:
  query_embedding + volatility_score + domain
    → Tavily web search (if volatile)
    → FAISS local index
    → Concept expansion (long-tail)
    → Normalized KnowledgeBlocks with embeddings
============================================================
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import aiohttp

from gateway.config import get_settings
from metacognitive.schemas import (
    KnowledgeBlock,
    KnowledgeRetrievalInput,
    KnowledgeRetrievalOutput,
)
from metacognitive.embedding import (
    embed_text,
    embed_texts,
    cosine_similarity,
)

logger = logging.getLogger("MCO-KnowledgeEngine")


# ============================================================
# Volatility Thresholds
# ============================================================

VOLATILITY_THRESHOLD = 0.35       # Trigger retrieval above this
LOW_CONFIDENCE_THRESHOLD = 0.4    # Trigger long-tail expansion


class KnowledgeRetrievalEngine:
    """
    API 2 — Knowledge & Retrieval Engine.

    Responsibilities:
      ✓ Live web search (Tavily)
      ✓ Local vector index (FAISS)
      ✓ Concept expansion for low-confidence domains
      ✓ Normalized factual block storage
      ✓ Timestamp inclusion
      ✗ No content generation
      ✗ No summarization before storage
      ✗ No session mutation
    """

    def __init__(self):
        self.settings = get_settings()
        self._faiss_index = None
        self._stored_blocks: List[KnowledgeBlock] = []
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _get_http(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._http_session

    async def close(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    # ── Public Interface ─────────────────────────────────────

    async def retrieve(
        self,
        inp: KnowledgeRetrievalInput,
    ) -> KnowledgeRetrievalOutput:
        """
        Main retrieval entry point.
        Follows volatility enforcement rules.
        """
        start = time.monotonic()
        bundles: List[KnowledgeBlock] = []
        sources_queried = 0
        expansion_applied = False

        # 1. Always check local index
        local_results = self._search_local_index(inp.query_embedding, top_k=5)
        bundles.extend(local_results)
        sources_queried += 1

        # 2. If volatile, mandatory web retrieval
        if inp.volatility_score > VOLATILITY_THRESHOLD:
            logger.info(
                f"Volatility {inp.volatility_score:.3f} > threshold {VOLATILITY_THRESHOLD}. "
                "Web retrieval mandatory."
            )
            web_results = await self._web_search(inp.query_text, inp.domain)
            bundles.extend(web_results)
            sources_queried += 1

        # 3. Concept expansion for low-confidence domains
        if inp.concept_expansion_depth > 0:
            expanded = await self._expand_concepts(
                inp.query_text,
                inp.domain,
                depth=inp.concept_expansion_depth,
            )
            if expanded:
                bundles.extend(expanded)
                expansion_applied = True
                sources_queried += 1

        # 4. Compute retrieval confidence
        if bundles:
            relevance_scores = [
                cosine_similarity(inp.query_embedding, b.embedding)
                for b in bundles
                if b.embedding
            ]
            retrieval_confidence = (
                sum(relevance_scores) / len(relevance_scores)
                if relevance_scores else 0.3
            )
        else:
            retrieval_confidence = 0.0

        # 5. Long-tail expansion if confidence still low
        if retrieval_confidence < LOW_CONFIDENCE_THRESHOLD and not expansion_applied:
            logger.info(
                f"Retrieval confidence {retrieval_confidence:.3f} < {LOW_CONFIDENCE_THRESHOLD}. "
                "Long-tail expansion triggered."
            )
            longtail = await self._long_tail_expansion(inp.query_text, inp.domain)
            bundles.extend(longtail)
            expansion_applied = True
            # Recompute confidence
            if bundles:
                relevance_scores = [
                    cosine_similarity(inp.query_embedding, b.embedding)
                    for b in bundles
                    if b.embedding
                ]
                retrieval_confidence = (
                    sum(relevance_scores) / len(relevance_scores)
                    if relevance_scores else 0.0
                )

        # 6. Store all blocks (no summarization)
        for block in bundles:
            if block not in self._stored_blocks:
                self._stored_blocks.append(block)

        # 7. De-duplicate by content hash
        seen = set()
        unique_bundles = []
        for b in bundles:
            key = hash(b.content[:200])
            if key not in seen:
                seen.add(key)
                unique_bundles.append(b)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            f"Retrieval complete: {len(unique_bundles)} blocks, "
            f"confidence={retrieval_confidence:.3f}, "
            f"{elapsed:.0f}ms"
        )

        return KnowledgeRetrievalOutput(
            knowledge_bundle=unique_bundles,
            retrieval_confidence=retrieval_confidence,
            sources_queried=sources_queried,
            expansion_applied=expansion_applied,
        )

    # ── Web Search (Tavily) ──────────────────────────────────

    async def _web_search(
        self,
        query: str,
        domain: str = "",
    ) -> List[KnowledgeBlock]:
        """Search the web using Tavily API."""
        api_key = self.settings.TAVILY_API_KEY
        if not api_key:
            logger.warning("TAVILY_API_KEY not configured. Web search skipped.")
            return []

        search_query = f"{domain}: {query}" if domain else query

        try:
            session = await self._get_http()
            payload = {
                "api_key": api_key,
                "query": search_query,
                "max_results": self.settings.RAG_TAVILY_MAX_RESULTS,
                "include_raw_content": False,
                "include_answer": False,
            }
            async with session.post(
                "https://api.tavily.com/search",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Tavily error {resp.status}: {text[:200]}")
                    return []
                data = await resp.json()

            results = data.get("results", [])
            blocks = []
            for r in results:
                content = r.get("content", "")
                if not content:
                    continue
                emb = embed_text(content)
                blocks.append(KnowledgeBlock(
                    source=r.get("url", "web"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content=content,
                    embedding=emb,
                    confidence=r.get("score", 0.5),
                    domain=domain,
                ))
            logger.info(f"Tavily returned {len(blocks)} results for: {query[:80]}")
            return blocks

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    # ── Local Vector Index ───────────────────────────────────

    def _search_local_index(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[KnowledgeBlock]:
        """Search stored knowledge blocks by embedding similarity."""
        if not self._stored_blocks or not query_embedding:
            return []

        scored = []
        for block in self._stored_blocks:
            if not block.embedding:
                continue
            sim = cosine_similarity(query_embedding, block.embedding)
            scored.append((sim, block))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:top_k]]

    # ── Concept Expansion ────────────────────────────────────

    async def _expand_concepts(
        self,
        query: str,
        domain: str,
        depth: int = 1,
    ) -> List[KnowledgeBlock]:
        """
        Expand query into related concepts and retrieve for each.
        Depth controls how many expansion layers to attempt.
        """
        # Generate expansion queries using simple heuristics
        expansions = self._generate_concept_expansions(query, domain)

        blocks = []
        for i, expanded_query in enumerate(expansions):
            if i >= depth * 3:  # Limit queries per depth level
                break
            web_results = await self._web_search(expanded_query, domain)
            blocks.extend(web_results)

        return blocks

    def _generate_concept_expansions(
        self,
        query: str,
        domain: str,
    ) -> List[str]:
        """Generate related queries for concept expansion."""
        expansions = []
        words = query.split()

        # Add domain-qualified variant
        if domain and domain.lower() not in query.lower():
            expansions.append(f"{domain} {query}")

        # Add "explained" variant for conceptual depth
        expansions.append(f"{query} explained in detail")

        # Add "latest research" variant for temporal freshness
        expansions.append(f"latest {query} 2025 2026")

        # Add comparison variant
        if len(words) >= 2:
            expansions.append(f"{query} vs alternatives comparison")

        return expansions[:5]

    # ── Long-Tail Expansion ──────────────────────────────────

    async def _long_tail_expansion(
        self,
        query: str,
        domain: str,
    ) -> List[KnowledgeBlock]:
        """
        Mandatory for low-confidence domains.
        Performs broader search with relaxed constraints.
        """
        logger.info(f"Long-tail expansion for: {query[:80]} (domain: {domain})")

        # Generate diverse search queries
        queries = [
            f"{query} comprehensive guide",
            f"{query} technical overview",
            f"{domain} {query} state of the art" if domain else f"{query} state of the art",
        ]

        blocks = []
        for q in queries:
            results = await self._web_search(q, domain)
            blocks.extend(results)

        return blocks

    # ── Knowledge Block Storage ──────────────────────────────

    def store_block(self, block: KnowledgeBlock):
        """Store a knowledge block (no summarization)."""
        if not block.timestamp:
            block.timestamp = datetime.now(timezone.utc).isoformat()
        if not block.embedding and block.content:
            block.embedding = embed_text(block.content)
        self._stored_blocks.append(block)

    def get_stored_count(self) -> int:
        return len(self._stored_blocks)

    def clear_stale_blocks(self, max_age_hours: int = 24):
        """Remove blocks older than max_age_hours."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        self._stored_blocks = [
            b for b in self._stored_blocks
            if datetime.fromisoformat(b.timestamp.replace("Z", "+00:00")) > cutoff
        ]
