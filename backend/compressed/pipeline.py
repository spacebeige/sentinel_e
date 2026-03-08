"""
LangGraph pipeline for Sentinel-E compressed reasoning.

Graph flow:
  User Query
    → context_loader   (load session history from SQLite)
    → token_guard      (estimate budget, compress context)
    → search_node      (Serper web search)
    → scrape_node      (fetch & clean HTML)
    → summarize_node   (local summarization of web content)
    → analysis_node    (Step 1: initial reasoning)
    → critique_node    (Step 2: adversarial critique)
    → synthesis_node   (Step 3: final synthesis)
    → memory_writer    (persist to SQLite)
    → Response
"""

import asyncio
import logging
import time
import uuid
from typing import TypedDict, Optional, List, Dict, Any, Annotated

from langgraph.graph import StateGraph, START, END

from compressed.memory_store import CompressedMemory, SessionContext
from compressed.search_engine import SerperSearch, WebScraper, SearchBundle, build_search_context
from compressed.model_clients import ModelRouter
from compressed.token_governor import TokenGovernor, TokenBudget, count_tokens
from compressed.debate_engine import CondensedDebateEngine, DebateResult
from compressed.report_generator import ReportGenerator, SentinelReport

logger = logging.getLogger("compressed.pipeline")


# ── State Schema ──

class PipelineState(TypedDict, total=False):
    # Input
    query: str
    session_id: str
    # Context
    session_ctx: Optional[SessionContext]
    history_context: str
    # Search
    search_bundle: Optional[SearchBundle]
    search_context: str
    # Token
    token_budget: Optional[TokenBudget]
    budget_ok: bool
    # Debate
    debate_result: Optional[DebateResult]
    # Output
    report: Optional[SentinelReport]
    formatted_output: str
    metadata: Dict[str, Any]
    error: Optional[str]
    # Timing
    start_time: float
    node_timings: Dict[str, float]


# ── Node Implementations ──

class PipelineNodes:
    """All node functions for the LangGraph pipeline."""

    def __init__(self):
        self.memory = CompressedMemory()
        self.search = SerperSearch()
        self.scraper = WebScraper()
        self.router = ModelRouter()
        self.report_gen = ReportGenerator()

    async def context_loader(self, state: PipelineState) -> dict:
        """Load session history from SQLite."""
        t0 = time.time()
        session_id = state.get("session_id") or str(uuid.uuid4())
        session_ctx = await self.memory.get_or_create_session(session_id)

        history_context = self.memory.build_context_prompt(session_ctx)

        # Store user message
        query = state["query"]
        await self.memory.add_message(session_id, "user", query)

        return {
            "session_id": session_id,
            "session_ctx": session_ctx,
            "history_context": history_context,
            "node_timings": {**(state.get("node_timings") or {}), "context_loader": (time.time() - t0) * 1000},
        }

    async def token_guard(self, state: PipelineState) -> dict:
        """Initialize token budget and compress context if needed."""
        t0 = time.time()
        budget = TokenBudget()
        governor = TokenGovernor(budget)

        # Compress history if it exceeds budget
        history_ctx = state.get("history_context", "")
        if history_ctx:
            history_ctx = governor.compress_context(history_ctx, budget.history_context)

        return {
            "token_budget": budget,
            "budget_ok": True,
            "history_context": history_ctx,
            "node_timings": {**(state.get("node_timings") or {}), "token_guard": (time.time() - t0) * 1000},
        }

    async def search_node(self, state: PipelineState) -> dict:
        """Execute web search via Serper."""
        t0 = time.time()

        if not self.search.available:
            logger.info("Serper not configured — skipping web search")
            return {
                "search_bundle": SearchBundle(query=state["query"]),
                "search_context": "",
                "node_timings": {**(state.get("node_timings") or {}), "search_node": (time.time() - t0) * 1000},
            }

        try:
            results = await self.search.search(state["query"], num_results=5)
            bundle = SearchBundle(query=state["query"], results=results)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            bundle = SearchBundle(query=state["query"], error=str(e))

        return {
            "search_bundle": bundle,
            "node_timings": {**(state.get("node_timings") or {}), "search_node": (time.time() - t0) * 1000},
        }

    async def scrape_node(self, state: PipelineState) -> dict:
        """Scrape top search result URLs."""
        t0 = time.time()
        bundle = state.get("search_bundle")

        if not bundle or not bundle.results:
            return {
                "search_context": "",
                "node_timings": {**(state.get("node_timings") or {}), "scrape_node": (time.time() - t0) * 1000},
            }

        # Check web cache first
        urls_to_scrape = []
        cached_summaries = []
        for r in bundle.results[:3]:  # Only scrape top 3
            cached = await self.memory.get_cached_summary(r.url)
            if cached:
                cached_summaries.append(cached)
            else:
                urls_to_scrape.append(r.url)

        # Scrape uncached URLs
        if urls_to_scrape:
            try:
                scraped = await self.scraper.scrape(urls_to_scrape, max_chars=3000)
                bundle.scraped = scraped
            except Exception as e:
                logger.warning(f"Scraping failed: {e}")

        return {
            "search_bundle": bundle,
            "node_timings": {**(state.get("node_timings") or {}), "scrape_node": (time.time() - t0) * 1000},
        }

    async def summarize_node(self, state: PipelineState) -> dict:
        """Summarize scraped content locally."""
        t0 = time.time()
        bundle = state.get("search_bundle")

        if not bundle:
            return {
                "search_context": "",
                "node_timings": {**(state.get("node_timings") or {}), "summarize_node": (time.time() - t0) * 1000},
            }

        summaries = []

        # Summarize scraped pages using local model (Groq) if available
        for page in (bundle.scraped or []):
            if not page.text:
                continue

            # Try local summarization first
            if self.router.groq.available and count_tokens(page.text) > 200:
                summary_prompt = (
                    f"Summarize the following webpage content in 2-3 sentences. "
                    f"Focus on factual information relevant to research.\n\n"
                    f"Title: {page.title}\n\nContent: {page.text[:2000]}"
                )
                resp = await self.router.generate(
                    prompt=summary_prompt,
                    max_tokens=200,
                    temperature=0.2,
                    prefer_local=True,
                )
                if resp.ok:
                    summaries.append(f"{page.title}: {resp.content}")
                    # Cache the summary
                    await self.memory.cache_summary(page.url, resp.content)
                    continue

            # Fallback: use snippet truncation
            summaries.append(f"{page.title}: {page.text[:300]}")

        # Also include snippet summaries for non-scraped results
        for r in (bundle.results or []):
            if r.snippet and not any(r.url in s for s in summaries):
                summaries.append(f"{r.title}: {r.snippet}")

        bundle.summaries = summaries[:5]

        # Build compressed search context
        governor = TokenGovernor(state.get("token_budget") or TokenBudget())
        search_ctx = build_search_context(bundle)
        search_ctx = governor.compress_search_context(search_ctx)

        return {
            "search_bundle": bundle,
            "search_context": search_ctx,
            "node_timings": {**(state.get("node_timings") or {}), "summarize_node": (time.time() - t0) * 1000},
        }

    async def debate_node(self, state: PipelineState) -> dict:
        """Run the 3-step condensed debate (Analysis → Critique → Synthesis)."""
        t0 = time.time()
        budget = state.get("token_budget") or TokenBudget()
        governor = TokenGovernor(budget)
        debate = CondensedDebateEngine(self.router, governor)

        result = await debate.run(
            query=state["query"],
            search_context=state.get("search_context", ""),
            history_context=state.get("history_context", ""),
        )

        return {
            "debate_result": result,
            "token_budget": budget,  # Updated with usage
            "node_timings": {**(state.get("node_timings") or {}), "debate": (time.time() - t0) * 1000},
        }

    async def report_node(self, state: PipelineState) -> dict:
        """Generate structured report from debate result."""
        t0 = time.time()
        debate_result = state.get("debate_result")

        if not debate_result or not debate_result.ok:
            error_msg = debate_result.error if debate_result else "No debate result"
            return {
                "formatted_output": f"Sentinel analysis failed: {error_msg}",
                "metadata": {"error": error_msg},
                "error": error_msg,
                "node_timings": {**(state.get("node_timings") or {}), "report": (time.time() - t0) * 1000},
            }

        search_used = bool(state.get("search_context"))
        report = self.report_gen.generate(debate_result, search_used=search_used)

        return {
            "report": report,
            "formatted_output": report.to_formatted_output(),
            "metadata": report.to_metadata(),
            "node_timings": {**(state.get("node_timings") or {}), "report": (time.time() - t0) * 1000},
        }

    async def memory_writer(self, state: PipelineState) -> dict:
        """Persist results to SQLite session memory."""
        t0 = time.time()
        session_id = state.get("session_id", "")
        output = state.get("formatted_output", "")

        if session_id and output:
            await self.memory.add_message(session_id, "assistant", output[:5000])

            # Record token usage
            budget = state.get("token_budget")
            if budget:
                await self.memory.record_token_usage(
                    session_id, "pipeline",
                    budget.tokens_used_in, budget.tokens_used_out,
                )

        total_time = (time.time() - state.get("start_time", time.time())) * 1000

        return {
            "node_timings": {**(state.get("node_timings") or {}), "memory_writer": (time.time() - t0) * 1000},
            "metadata": {
                **(state.get("metadata") or {}),
                "total_latency_ms": total_time,
                "node_timings": state.get("node_timings", {}),
            },
        }


# ── Graph Builder ──

def build_pipeline() -> StateGraph:
    """Build the LangGraph pipeline."""
    nodes = PipelineNodes()

    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("context_loader", nodes.context_loader)
    graph.add_node("token_guard", nodes.token_guard)
    graph.add_node("search_node", nodes.search_node)
    graph.add_node("scrape_node", nodes.scrape_node)
    graph.add_node("summarize_node", nodes.summarize_node)
    graph.add_node("debate_node", nodes.debate_node)
    graph.add_node("report_node", nodes.report_node)
    graph.add_node("memory_writer", nodes.memory_writer)

    # Define edges (linear flow)
    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "token_guard")
    graph.add_edge("token_guard", "search_node")
    graph.add_edge("search_node", "scrape_node")
    graph.add_edge("scrape_node", "summarize_node")
    graph.add_edge("summarize_node", "debate_node")
    graph.add_edge("debate_node", "report_node")
    graph.add_edge("report_node", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph


def compile_pipeline():
    """Compile the pipeline into an executable graph."""
    graph = build_pipeline()
    return graph.compile()


# ── Public API ──

_compiled_pipeline = None


def get_pipeline():
    """Get or create the compiled pipeline (singleton)."""
    global _compiled_pipeline
    if _compiled_pipeline is None:
        _compiled_pipeline = compile_pipeline()
    return _compiled_pipeline


async def run_compressed_pipeline(
    query: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the full compressed reasoning pipeline.

    Returns dict with:
      - formatted_output: str (the full Sentinel report)
      - metadata: dict (omega_metadata-compatible)
      - session_id: str
      - error: Optional[str]
    """
    pipeline = get_pipeline()

    initial_state: PipelineState = {
        "query": query,
        "session_id": session_id or str(uuid.uuid4()),
        "start_time": time.time(),
        "node_timings": {},
    }

    result = await pipeline.ainvoke(initial_state)

    return {
        "formatted_output": result.get("formatted_output", ""),
        "metadata": result.get("metadata", {}),
        "session_id": result.get("session_id", ""),
        "report": result.get("report"),
        "error": result.get("error"),
    }
