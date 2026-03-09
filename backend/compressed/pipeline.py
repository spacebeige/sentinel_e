"""
LangGraph pipeline for Sentinel-E role-based reasoning.

10-node graph:
  context_loader   → load session history from SQLite
  token_guard      → estimate budget, compress context
  search_node      → Dual web search (Tavily + Serper)
  scrape_node      → fetch & clean HTML
  summarize_node   → local summarization (Groq 8B)
  analysis_node    → deep analysis (Llama-3.3-70B / Gemini Flash)
  critique_node    → parallel adversarial critique (Mixtral ∥ Gemma ∥ Qwen)
  synthesis_node   → integrate analysis + critiques (Gemini Flash)
  verification_node→ single fact-check (Llama-3.1-8B)
  report_memory    → generate report + persist to SQLite

~6 API calls per query (down from 21 in original ensemble).
"""

import asyncio
import logging
import time
import uuid
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from compressed.memory_store import CompressedMemory, SessionContext
from compressed.search_engine import DualSearchEngine, SerperSearch, WebScraper, SearchBundle, build_search_context, format_citations
from compressed.model_clients import RoleBasedRouter
from compressed.token_governor import TokenGovernor, TokenBudget, count_tokens
from compressed.role_engine import RoleBasedEngine, RoleResult
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
    citations_block: str
    # Token
    token_budget: Optional[TokenBudget]
    budget_ok: bool
    # Role-based reasoning
    role_result: Optional[RoleResult]
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
        self.search = DualSearchEngine()
        self.scraper = WebScraper()
        self.router = RoleBasedRouter()
        self.report_gen = ReportGenerator()

    async def context_loader(self, state: PipelineState) -> dict:
        """Load session history from SQLite."""
        t0 = time.time()
        session_id = state.get("session_id") or str(uuid.uuid4())
        session_ctx = await self.memory.get_or_create_session(session_id)

        history_context = self.memory.build_context_prompt(session_ctx)

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
        """Execute web search via Tavily + Serper (dual search)."""
        t0 = time.time()

        if not self.search.available:
            logger.info("No search providers configured — skipping web search")
            return {
                "search_bundle": SearchBundle(query=state["query"]),
                "search_context": "",
                "citations_block": "",
                "node_timings": {**(state.get("node_timings") or {}), "search_node": (time.time() - t0) * 1000},
            }

        try:
            results = await self.search.search(state["query"], num_results=5)
            bundle = SearchBundle(query=state["query"], results=results)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            bundle = SearchBundle(query=state["query"], error=str(e))

        citations = format_citations(bundle.results) if bundle.results else ""

        return {
            "search_bundle": bundle,
            "citations_block": citations,
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

        urls_to_scrape = []
        cached_summaries = []
        for r in bundle.results[:3]:
            cached = await self.memory.get_cached_summary(r.url)
            if cached:
                cached_summaries.append(cached)
            else:
                urls_to_scrape.append(r.url)

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
        """Summarize scraped content using fast local model."""
        t0 = time.time()
        bundle = state.get("search_bundle")

        if not bundle:
            return {
                "search_context": "",
                "node_timings": {**(state.get("node_timings") or {}), "summarize_node": (time.time() - t0) * 1000},
            }

        summaries = []

        for page in (bundle.scraped or []):
            if not page.text:
                continue

            if self.router.groq_8b.available and count_tokens(page.text) > 200:
                summary_prompt = (
                    f"Summarize the following webpage content in 2-3 sentences. "
                    f"Focus on factual information relevant to research.\n\n"
                    f"Title: {page.title}\n\nContent: {page.text[:2000]}"
                )
                resp = await self.router.generate(
                    role="summarize",
                    prompt=summary_prompt,
                    max_tokens=200,
                    temperature=0.2,
                )
                if resp.ok:
                    summaries.append(f"{page.title}: {resp.content}")
                    await self.memory.cache_summary(page.url, resp.content)
                    continue

            summaries.append(f"{page.title}: {page.text[:300]}")

        for r in (bundle.results or []):
            if r.snippet and not any(r.url in s for s in summaries):
                summaries.append(f"{r.title}: {r.snippet}")

        bundle.summaries = summaries[:5]

        governor = TokenGovernor(state.get("token_budget") or TokenBudget())
        search_ctx = build_search_context(bundle)
        search_ctx = governor.compress_search_context(search_ctx)

        return {
            "search_bundle": bundle,
            "search_context": search_ctx,
            "node_timings": {**(state.get("node_timings") or {}), "summarize_node": (time.time() - t0) * 1000},
        }

    async def analysis_node(self, state: PipelineState) -> dict:
        """Stage 1: Deep analysis (Gemini Flash / Llama-3.3-70B) — 1 API call."""
        t0 = time.time()
        budget = state.get("token_budget") or TokenBudget()
        governor = TokenGovernor(budget)
        engine = RoleBasedEngine(self.router, governor)

        context_block = ""
        if state.get("search_context"):
            context_block += f"WEB RESEARCH:\n{state['search_context']}\n\n"
        if state.get("history_context"):
            context_block += f"CONVERSATION CONTEXT:\n{state['history_context']}\n\n"

        # Initialize role result
        role_result = RoleResult(query=state["query"])

        analysis = await engine.run_analysis(state["query"], context_block)
        role_result.analysis = analysis
        role_result.api_calls += 1

        if not analysis.content:
            role_result.error = "Analysis stage produced no output"

        return {
            "role_result": role_result,
            "token_budget": budget,
            "node_timings": {**(state.get("node_timings") or {}), "analysis": (time.time() - t0) * 1000},
        }

    async def critique_node(self, state: PipelineState) -> dict:
        """Stage 2: Parallel critique (Gemma-9B ∥ Mistral-7B) — 2 API calls."""
        t0 = time.time()
        role_result = state.get("role_result")
        budget = state.get("token_budget") or TokenBudget()
        governor = TokenGovernor(budget)
        engine = RoleBasedEngine(self.router, governor)

        if not role_result or not role_result.analysis or not role_result.analysis.content:
            return {
                "node_timings": {**(state.get("node_timings") or {}), "critique": (time.time() - t0) * 1000},
            }

        if governor.budget.exhausted:
            logger.warning("Budget exhausted — skipping critique")
            return {
                "token_budget": budget,
                "node_timings": {**(state.get("node_timings") or {}), "critique": (time.time() - t0) * 1000},
            }

        critiques = await engine.run_critiques(state["query"], role_result.analysis.content)
        role_result.critiques = critiques
        role_result.api_calls += len(critiques)

        return {
            "role_result": role_result,
            "token_budget": budget,
            "node_timings": {**(state.get("node_timings") or {}), "critique": (time.time() - t0) * 1000},
        }

    async def synthesis_node(self, state: PipelineState) -> dict:
        """Stage 3: Synthesize analysis + critiques (Gemini Flash) — 1 API call."""
        t0 = time.time()
        role_result = state.get("role_result")
        budget = state.get("token_budget") or TokenBudget()
        governor = TokenGovernor(budget)
        engine = RoleBasedEngine(self.router, governor)

        if not role_result or not role_result.analysis:
            return {
                "node_timings": {**(state.get("node_timings") or {}), "synthesis": (time.time() - t0) * 1000},
            }

        context_block = ""
        if state.get("search_context"):
            context_block += f"WEB RESEARCH:\n{state['search_context']}\n\n"
        if state.get("history_context"):
            context_block += f"CONVERSATION CONTEXT:\n{state['history_context']}\n\n"

        if governor.budget.exhausted:
            logger.warning("Budget exhausted — using analysis as synthesis")
            from compressed.role_engine import StageResult
            role_result.synthesis = StageResult(
                role="synthesis_fallback",
                content=role_result.analysis.content,
                model=role_result.analysis.model,
            )
            return {
                "role_result": role_result,
                "token_budget": budget,
                "node_timings": {**(state.get("node_timings") or {}), "synthesis": (time.time() - t0) * 1000},
            }

        critique_a = role_result.critiques[0].content if role_result.critiques else "No critique available."
        critique_b = role_result.critiques[1].content if len(role_result.critiques) > 1 else "No critique available."
        critique_c = role_result.critiques[2].content if len(role_result.critiques) > 2 else "No critique available."

        synthesis = await engine.run_synthesis(
            state["query"], context_block,
            role_result.analysis.content, critique_a, critique_b,
            critique_c_text=critique_c,
            citations_block=state.get("citations_block", ""),
        )
        role_result.synthesis = synthesis
        role_result.api_calls += 1

        if not synthesis.content:
            from compressed.role_engine import StageResult
            role_result.synthesis = StageResult(
                role="synthesis_fallback",
                content=role_result.analysis.content,
                model=role_result.analysis.model,
            )

        return {
            "role_result": role_result,
            "token_budget": budget,
            "node_timings": {**(state.get("node_timings") or {}), "synthesis": (time.time() - t0) * 1000},
        }

    async def verification_node(self, state: PipelineState) -> dict:
        """Stage 4: Verification (Llama-3.1-8B) — 1 API call."""
        t0 = time.time()
        role_result = state.get("role_result")
        budget = state.get("token_budget") or TokenBudget()
        governor = TokenGovernor(budget)
        engine = RoleBasedEngine(self.router, governor)

        if not role_result or not role_result.synthesis or not role_result.synthesis.content:
            return {
                "node_timings": {**(state.get("node_timings") or {}), "verification": (time.time() - t0) * 1000},
            }

        if governor.budget.exhausted:
            logger.warning("Budget exhausted — skipping verification")
            return {
                "token_budget": budget,
                "node_timings": {**(state.get("node_timings") or {}), "verification": (time.time() - t0) * 1000},
            }

        verification = await engine.run_verification(state["query"], role_result.synthesis.content)
        role_result.verifications = [verification]
        role_result.api_calls += 1

        # Finalize totals
        all_stages = (
            [role_result.analysis] +
            role_result.critiques +
            ([role_result.synthesis] if role_result.synthesis else []) +
            role_result.verifications
        )
        role_result.total_tokens_in = sum(s.tokens_in for s in all_stages if s)
        role_result.total_tokens_out = sum(s.tokens_out for s in all_stages if s)

        return {
            "role_result": role_result,
            "token_budget": budget,
            "node_timings": {**(state.get("node_timings") or {}), "verification": (time.time() - t0) * 1000},
        }

    async def report_memory(self, state: PipelineState) -> dict:
        """Generate structured report and persist to SQLite."""
        t0 = time.time()
        role_result = state.get("role_result")

        # ── Report generation ──
        if not role_result or not role_result.ok:
            error_msg = role_result.error if role_result else "No reasoning result"
            return {
                "formatted_output": f"Sentinel analysis failed: {error_msg}",
                "metadata": {"error": error_msg},
                "error": error_msg,
                "node_timings": {**(state.get("node_timings") or {}), "report_memory": (time.time() - t0) * 1000},
            }

        search_used = bool(state.get("search_context"))
        report = self.report_gen.generate(role_result, search_used=search_used, search_bundle=state.get("search_bundle"))
        formatted = report.to_formatted_output()
        metadata = report.to_metadata()

        # ── Memory persistence ──
        session_id = state.get("session_id", "")
        if session_id and formatted:
            await self.memory.add_message(session_id, "assistant", formatted[:5000])
            budget = state.get("token_budget")
            if budget:
                await self.memory.record_token_usage(
                    session_id, "pipeline",
                    budget.tokens_used_in, budget.tokens_used_out,
                )

        total_time = (time.time() - state.get("start_time", time.time())) * 1000
        metadata["total_latency_ms"] = total_time
        metadata["node_timings"] = state.get("node_timings", {})

        return {
            "report": report,
            "formatted_output": formatted,
            "metadata": metadata,
            "node_timings": {**(state.get("node_timings") or {}), "report_memory": (time.time() - t0) * 1000},
        }


# ── Graph Builder ──

def build_pipeline() -> StateGraph:
    """Build the 10-node LangGraph pipeline."""
    nodes = PipelineNodes()

    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("context_loader", nodes.context_loader)
    graph.add_node("token_guard", nodes.token_guard)
    graph.add_node("search_node", nodes.search_node)
    graph.add_node("scrape_node", nodes.scrape_node)
    graph.add_node("summarize_node", nodes.summarize_node)
    graph.add_node("analysis_node", nodes.analysis_node)
    graph.add_node("critique_node", nodes.critique_node)
    graph.add_node("synthesis_node", nodes.synthesis_node)
    graph.add_node("verification_node", nodes.verification_node)
    graph.add_node("report_memory", nodes.report_memory)

    # Define edges (linear flow with internal parallelism in critique/verification)
    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "token_guard")
    graph.add_edge("token_guard", "search_node")
    graph.add_edge("search_node", "scrape_node")
    graph.add_edge("scrape_node", "summarize_node")
    graph.add_edge("summarize_node", "analysis_node")
    graph.add_edge("analysis_node", "critique_node")
    graph.add_edge("critique_node", "synthesis_node")
    graph.add_edge("synthesis_node", "verification_node")
    graph.add_edge("verification_node", "report_memory")
    graph.add_edge("report_memory", END)

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
    Execute the role-based reasoning pipeline.

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
