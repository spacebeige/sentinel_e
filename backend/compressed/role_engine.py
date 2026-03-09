"""
Role-based reasoning engine for Sentinel-E compressed pipeline.

4-stage pipeline where models are assigned specific reasoning roles:
  Stage 1 — Analysis:      Deep structured analysis (Llama-3.3-70B / Gemini Flash)
  Stage 2 — Critique:      Parallel adversarial review (Mixtral-8x7B + Gemma-7B + Qwen-2.5-VL)
  Stage 3 — Synthesis:     Integrate all perspectives (Gemini Flash)
  Stage 4 — Verification:  Cross-model fact-check (Llama-3.1-8B)

~6 API calls total (down from 21 in the original ensemble).
"""

import asyncio
import logging
import time
from typing import Optional, List
from dataclasses import dataclass, field

from compressed.model_clients import RoleBasedRouter, ModelResponse
from compressed.token_governor import TokenGovernor, count_tokens

logger = logging.getLogger("compressed.role_engine")


# ── Data Classes ──

@dataclass
class StageResult:
    """Result from a single model call within a stage."""
    role: str
    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


@dataclass
class RoleResult:
    """Aggregate result from the 4-stage role-based pipeline."""
    query: str
    analysis: Optional[StageResult] = None
    critiques: List[StageResult] = field(default_factory=list)
    synthesis: Optional[StageResult] = None
    verifications: List[StageResult] = field(default_factory=list)
    api_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.synthesis is not None and bool(self.synthesis.content)

    @property
    def final_output(self) -> str:
        if self.synthesis and self.synthesis.content:
            return self.synthesis.content
        if self.analysis and self.analysis.content:
            return self.analysis.content
        return ""

    @property
    def models_used(self) -> list:
        models = set()
        if self.analysis:
            models.add(self.analysis.model)
        for c in self.critiques:
            models.add(c.model)
        if self.synthesis:
            models.add(self.synthesis.model)
        for v in self.verifications:
            models.add(v.model)
        return sorted(models)

    @property
    def verification_consensus(self) -> Optional[bool]:
        """True if the verifier considers the synthesis sound, None if no verification."""
        if not self.verifications:
            return None
        text = self.verifications[0].content.upper()
        issue_markers = ["ISSUES FOUND", "GAPS FOUND", "INACCURATE", "CONTRADICTS"]
        return not any(m in text for m in issue_markers)


# ── Stage Prompts ──

ANALYSIS_SYSTEM = (
    "You are the ANALYSIS engine of Sentinel-E, an advanced multi-model reasoning system. "
    "Your role: provide deep, structured analysis as the foundation for subsequent critique and synthesis.\n\n"
    "Rules:\n"
    "- Decompose the problem into key factors\n"
    "- Identify risks, opportunities, and dependencies\n"
    "- Consider multiple perspectives\n"
    "- Be specific with evidence and reasoning\n"
    "- Structure your output clearly with numbered points"
)

ANALYSIS_PROMPT = """Analyze the following query thoroughly.

{context}

QUERY: {query}

Provide structured analysis covering:
1. Problem decomposition — key components and their relationships
2. Critical factors — what drives the outcome
3. Risk assessment — potential failure modes and their likelihood
4. Strategic options — ranked approaches with trade-offs

Be concise but comprehensive. This analysis will be critiqued by two independent models."""

CRITIQUE_A_SYSTEM = (
    "You are CRITIQUE MODEL A (Mixtral) in Sentinel-E's multi-model reasoning pipeline. "
    "Your role: provide adversarial critique from a LOGICAL REASONING perspective.\n\n"
    "Focus on: logical gaps, unsupported assumptions, missing evidence, reasoning fallacies.\n"
    "Be constructive but thorough — your critique improves the final answer."
)

CRITIQUE_B_SYSTEM = (
    "You are CRITIQUE MODEL B (Gemma) in Sentinel-E's multi-model reasoning pipeline. "
    "Your role: provide adversarial critique from a PRACTICAL / REAL-WORLD perspective.\n\n"
    "Focus on: implementation feasibility, real-world constraints, missing stakeholders, edge cases.\n"
    "Be constructive but thorough — your critique improves the final answer."
)

CRITIQUE_C_SYSTEM = (
    "You are CRITIQUE MODEL C (Qwen) in Sentinel-E's multi-model reasoning pipeline. "
    "Your role: provide adversarial critique from an ALTERNATIVE PERSPECTIVES angle.\n\n"
    "Focus on: unconsidered viewpoints, cultural or regional factors, ethical dimensions, "
    "second-order effects, and creative solutions the analysis missed.\n"
    "Be constructive but thorough — your critique improves the final answer."
)

CRITIQUE_PROMPT = """Review and critique the following analysis.

ORIGINAL QUERY: {query}

ANALYSIS:
{analysis}

Provide your critique:
1. Specific weaknesses or logical gaps
2. Missing perspectives or blind spots
3. Counter-arguments the analysis should address
4. Suggested improvements
5. Strength rating: STRONG / MODERATE / WEAK (with justification)"""

SYNTHESIS_SYSTEM = (
    "You are the SYNTHESIS engine of Sentinel-E. You integrate an initial analysis "
    "with critiques from three independent models into a definitive, well-reasoned final answer.\n\n"
    "You must produce output in EXACTLY this format:\n\n"
    "## OVERVIEW\n[Brief explanation of the problem]\n\n"
    "## ASSESSMENT\n[Primary reasoning analysis incorporating the initial analysis and all critiques]\n\n"
    "## CRITICAL ISSUES\n[Key risks, logical weaknesses, or concerns from the critique phase]\n\n"
    "## TACTICAL MAP\n[Strategic options or approaches, ranked by viability]\n\n"
    "## FINAL SYNTHESIS\n[Definitive answer integrating all reasoning — be decisive]\n\n"
    "## CONFIDENCE\n[Score 0-100% with brief justification]"
)

SYNTHESIS_PROMPT = """Synthesize the analysis and critiques into a final Sentinel report.

{context}

QUERY: {query}

INITIAL ANALYSIS:
{analysis}

CRITIQUE A (Mixtral — Logical/Reasoning perspective):
{critique_a}

CRITIQUE B (Gemma — Practical/Real-world perspective):
{critique_b}

CRITIQUE C (Qwen — Alternative perspectives):
{critique_c}

{citations}

Produce the final Sentinel Analysis Report. Address the critiques explicitly. Be decisive."""

VERIFY_SYSTEM = (
    "You are the VERIFICATION engine in Sentinel-E's reasoning pipeline. "
    "Your role: fact-check the synthesis for logical consistency, accuracy, and completeness.\n\n"
    "Output format:\n"
    "VERDICT: [VERIFIED | PARTIALLY VERIFIED | ISSUES FOUND]\n"
    "FINDINGS: [bullet points]\n"
    "QUALITY: [HIGH | MEDIUM | LOW]"
)

VERIFY_PROMPT = """Verify the following synthesis report.

ORIGINAL QUERY: {query}

SYNTHESIS:
{synthesis}

Provide your verification:
1. Verdict
2. Specific findings (max 5 bullet points)
3. Overall quality assessment"""


# ── Engine ──

class RoleBasedEngine:
    """4-stage role-based reasoning: Analysis → Critique(×2) → Synthesis → Verify(×2)."""

    def __init__(self, router: RoleBasedRouter, governor: TokenGovernor):
        self.router = router
        self.governor = governor

    async def run_analysis(
        self,
        query: str,
        context_block: str = "",
    ) -> StageResult:
        """Stage 1: Deep analysis (1 API call)."""
        prompt = ANALYSIS_PROMPT.format(
            query=query,
            context=context_block or "No additional context.",
        )
        prompt = self.governor.compress_context(prompt, self.governor.budget.per_call_input)

        t0 = time.time()
        resp = await self.router.generate(
            role="analysis",
            prompt=prompt,
            system_instruction=ANALYSIS_SYSTEM,
            max_tokens=min(1500, self.governor.budget.allowed_output()),
            temperature=0.4,
        )
        latency = (time.time() - t0) * 1000

        self.governor.record_usage(resp.tokens_in, resp.tokens_out)

        return StageResult(
            role="analysis",
            content=resp.content if resp.ok else "",
            model=resp.model,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=latency,
        )

    async def run_critiques(
        self,
        query: str,
        analysis_text: str,
    ) -> List[StageResult]:
        """Stage 2: Parallel critique from three models (3 API calls)."""
        critique_prompt = CRITIQUE_PROMPT.format(
            query=query,
            analysis=analysis_text[:2000],
        )
        critique_prompt = self.governor.compress_context(
            critique_prompt, self.governor.budget.per_call_input
        )
        max_out = min(1024, self.governor.budget.allowed_output())

        async def _call_critique(role: str, system: str) -> StageResult:
            t0 = time.time()
            resp = await self.router.generate(
                role=role,
                prompt=critique_prompt,
                system_instruction=system,
                max_tokens=max_out,
                temperature=0.5,
            )
            latency = (time.time() - t0) * 1000
            self.governor.record_usage(resp.tokens_in, resp.tokens_out)
            return StageResult(
                role=role,
                content=resp.content if resp.ok else f"[{role} failed: {resp.error}]",
                model=resp.model,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                latency_ms=latency,
            )

        results = await asyncio.gather(
            _call_critique("critique_a", CRITIQUE_A_SYSTEM),
            _call_critique("critique_b", CRITIQUE_B_SYSTEM),
            _call_critique("critique_c", CRITIQUE_C_SYSTEM),
        )
        return list(results)

    async def run_synthesis(
        self,
        query: str,
        context_block: str,
        analysis_text: str,
        critique_a_text: str,
        critique_b_text: str,
        critique_c_text: str = "",
        citations_block: str = "",
    ) -> StageResult:
        """Stage 3: Synthesize analysis + critiques (1 API call)."""
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            context=context_block or "No additional context.",
            analysis=analysis_text[:2000],
            critique_a=critique_a_text[:1200],
            critique_b=critique_b_text[:1200],
            critique_c=critique_c_text[:1200] if critique_c_text else "No critique available.",
            citations=citations_block or "",
        )
        prompt = self.governor.compress_context(prompt, self.governor.budget.per_call_input)

        t0 = time.time()
        resp = await self.router.generate(
            role="synthesis",
            prompt=prompt,
            system_instruction=SYNTHESIS_SYSTEM,
            max_tokens=min(2048, self.governor.budget.allowed_output()),
            temperature=0.3,
        )
        latency = (time.time() - t0) * 1000
        self.governor.record_usage(resp.tokens_in, resp.tokens_out)

        return StageResult(
            role="synthesis",
            content=resp.content if resp.ok else "",
            model=resp.model,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=latency,
        )

    async def run_verification(
        self,
        query: str,
        synthesis_text: str,
    ) -> StageResult:
        """Stage 4: Single verification (1 API call)."""
        verify_prompt = VERIFY_PROMPT.format(
            query=query,
            synthesis=synthesis_text[:2500],
        )
        verify_prompt = self.governor.compress_context(
            verify_prompt, self.governor.budget.per_call_input
        )
        max_out = min(512, self.governor.budget.allowed_output())

        t0 = time.time()
        resp = await self.router.generate(
            role="verification",
            prompt=verify_prompt,
            system_instruction=VERIFY_SYSTEM,
            max_tokens=max_out,
            temperature=0.2,
        )
        latency = (time.time() - t0) * 1000
        self.governor.record_usage(resp.tokens_in, resp.tokens_out)
        return StageResult(
            role="verification",
            content=resp.content if resp.ok else f"[verification failed: {resp.error}]",
            model=resp.model,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=latency,
        )

    async def run(
        self,
        query: str,
        search_context: str = "",
        history_context: str = "",
        citations_block: str = "",
    ) -> RoleResult:
        """Execute the full 4-stage role-based pipeline."""
        result = RoleResult(query=query)
        start = time.time()

        context_block = ""
        if search_context:
            context_block += f"WEB RESEARCH:\n{search_context}\n\n"
        if history_context:
            context_block += f"CONVERSATION CONTEXT:\n{history_context}\n\n"

        # ── Stage 1: Analysis ──
        logger.info("Stage 1: Analysis (Llama-3.3-70B)")
        analysis = await self.run_analysis(query, context_block)
        result.analysis = analysis
        result.api_calls += 1

        if not analysis.content:
            result.error = "Analysis stage produced no output"
            result.total_latency_ms = (time.time() - start) * 1000
            return result

        # ── Stage 2: Parallel Critique ──
        if not self.governor.check_budget():
            logger.warning("Budget exhausted after analysis — skipping critique")
            result.synthesis = StageResult(
                role="synthesis", content=analysis.content,
                model=analysis.model, tokens_in=0, tokens_out=0,
            )
            result.total_latency_ms = (time.time() - start) * 1000
            return result

        logger.info("Stage 2: Parallel Critique (Mixtral + Gemma + Qwen)")
        critiques = await self.run_critiques(query, analysis.content)
        result.critiques = critiques
        result.api_calls += len(critiques)

        # ── Stage 3: Synthesis ──
        if not self.governor.check_budget():
            logger.warning("Budget exhausted after critique — using analysis as synthesis")
            result.synthesis = StageResult(
                role="synthesis", content=analysis.content,
                model=analysis.model, tokens_in=0, tokens_out=0,
            )
            result.total_latency_ms = (time.time() - start) * 1000
            return result

        logger.info("Stage 3: Synthesis (Gemini Flash)")
        critique_a_text = critiques[0].content if critiques else "No critique available."
        critique_b_text = critiques[1].content if len(critiques) > 1 else "No critique available."
        critique_c_text = critiques[2].content if len(critiques) > 2 else "No critique available."

        synthesis = await self.run_synthesis(
            query, context_block,
            analysis.content, critique_a_text, critique_b_text,
            critique_c_text=critique_c_text,
            citations_block=citations_block,
        )
        result.synthesis = synthesis
        result.api_calls += 1

        if not synthesis.content:
            result.synthesis = StageResult(
                role="synthesis_fallback", content=analysis.content,
                model=analysis.model, tokens_in=0, tokens_out=0,
            )

        # ── Stage 4: Verification ──
        if not self.governor.check_budget():
            logger.warning("Budget exhausted — skipping verification")
            result.total_latency_ms = (time.time() - start) * 1000
            self._finalize(result, start)
            return result

        logger.info("Stage 4: Verification (Llama-3.1-8B)")
        verification = await self.run_verification(query, result.synthesis.content)
        result.verifications = [verification]
        result.api_calls += 1

        self._finalize(result, start)
        return result

    def _finalize(self, result: RoleResult, start: float):
        """Compute aggregate token counts and latency."""
        all_stages = (
            [result.analysis] +
            result.critiques +
            ([result.synthesis] if result.synthesis else []) +
            result.verifications
        )
        result.total_tokens_in = sum(s.tokens_in for s in all_stages if s)
        result.total_tokens_out = sum(s.tokens_out for s in all_stages if s)
        result.total_latency_ms = (time.time() - start) * 1000

        logger.info(
            f"Role-based pipeline complete: {result.api_calls} API calls, "
            f"{result.total_tokens_in}+{result.total_tokens_out} tokens, "
            f"{result.total_latency_ms:.0f}ms, "
            f"models: {result.models_used}"
        )


# ── Aliases for backward compatibility ──
RoleEngine = RoleBasedEngine
ReasoningState = RoleResult
