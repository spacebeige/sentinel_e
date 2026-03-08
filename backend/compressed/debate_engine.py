"""
Condensed 3-step debate engine.

Replaces the old 7-model × 3-round debate with a compressed chain:
  Step 1 — Analysis  : Initial reasoning on the problem
  Step 2 — Critique  : Evaluate and challenge the analysis 
  Step 3 — Synthesis : Combine all reasoning into final report

Maximum 2-3 external API calls. Local models handle intermediate steps
where possible.
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from compressed.model_clients import ModelRouter, ModelResponse
from compressed.token_governor import TokenGovernor, count_tokens

logger = logging.getLogger("compressed.debate")


@dataclass
class DebateStep:
    name: str
    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


@dataclass 
class DebateResult:
    query: str
    analysis: Optional[DebateStep] = None
    critique: Optional[DebateStep] = None
    synthesis: Optional[DebateStep] = None
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


# ── Prompts ──

ANALYSIS_SYSTEM = """You are a senior analytical reasoning engine. Your role is to provide 
deep, structured analysis of any question or problem presented to you.

Rules:
- Think step by step
- Identify key factors, risks, and opportunities
- Consider multiple perspectives
- Be specific, not generic
- Support claims with reasoning"""

ANALYSIS_PROMPT = """Analyze the following query thoroughly.

{context}

QUERY: {query}

Provide a structured analysis covering:
1. Problem decomposition
2. Key factors and their interactions
3. Potential risks and failure modes
4. Strategic options with trade-offs

Be concise but comprehensive."""

CRITIQUE_SYSTEM = """You are a critical evaluator. Your role is to find weaknesses, 
blind spots, and logical gaps in an analysis. You challenge assumptions and identify 
what the initial analysis missed.

Rules:
- Be adversarial but constructive
- Identify specific logical weaknesses
- Point out missing perspectives or data
- Suggest what would change the conclusion
- Rate the original analysis quality"""

CRITIQUE_PROMPT = """Review and critique the following analysis.

ORIGINAL QUERY: {query}

ANALYSIS:
{analysis}

Provide:
1. Logical weaknesses or gaps
2. Missing perspectives
3. Counter-arguments
4. What evidence would change the conclusion
5. Quality rating (1-10) with justification"""

SYNTHESIS_SYSTEM = """You are the final synthesis engine for Sentinel-E, an advanced 
AI reasoning system. You combine an initial analysis and its critique into a definitive, 
well-reasoned final answer.

You must produce output in EXACTLY this format:

## OVERVIEW
[Brief explanation of the problem]

## ASSESSMENT
[Primary reasoning analysis incorporating both the initial analysis and critique]

## CRITICAL ISSUES
[Key risks, logical weaknesses, or concerns identified during debate]

## TACTICAL MAP
[Strategic options or approaches, ranked by viability]

## FINAL SYNTHESIS
[Definitive answer integrating all reasoning]

## CONFIDENCE
[Score 0-100% with brief justification]"""

SYNTHESIS_PROMPT = """Synthesize the following analysis and critique into a final Sentinel report.

{context}

QUERY: {query}

INITIAL ANALYSIS:
{analysis}

CRITIQUE:
{critique}

Produce the final Sentinel Analysis Report. Be decisive — provide a clear answer, not hedging."""


class CondensedDebateEngine:
    """3-step compressed debate: Analysis → Critique → Synthesis."""

    def __init__(self, router: ModelRouter, governor: TokenGovernor):
        self.router = router
        self.governor = governor

    async def run(
        self,
        query: str,
        search_context: str = "",
        history_context: str = "",
    ) -> DebateResult:
        """Execute the compressed 3-step debate."""
        result = DebateResult(query=query)
        start = time.time()

        context_block = ""
        if search_context:
            context_block += f"WEB RESEARCH:\n{search_context}\n\n"
        if history_context:
            context_block += f"CONVERSATION CONTEXT:\n{history_context}\n\n"

        # ── Step 1: Analysis (prefer local if available) ──
        logger.info("Debate Step 1: Analysis")
        analysis_prompt = ANALYSIS_PROMPT.format(
            query=query,
            context=context_block if context_block else "No additional context.",
        )

        # Compress if needed
        analysis_prompt = self.governor.compress_context(
            analysis_prompt, self.governor.budget.per_call_input
        )

        t0 = time.time()
        analysis_resp = await self.router.generate(
            prompt=analysis_prompt,
            system_instruction=ANALYSIS_SYSTEM,
            max_tokens=min(1500, self.governor.budget.allowed_output()),
            temperature=0.4,
            prefer_local=True,  # Use Groq for analysis if available
        )
        analysis_time = (time.time() - t0) * 1000

        if not analysis_resp.ok:
            result.error = f"Analysis failed: {analysis_resp.error}"
            result.total_latency_ms = (time.time() - start) * 1000
            return result

        result.analysis = DebateStep(
            name="analysis",
            content=analysis_resp.content,
            model=analysis_resp.model,
            tokens_in=analysis_resp.tokens_in,
            tokens_out=analysis_resp.tokens_out,
            latency_ms=analysis_time,
        )
        result.api_calls += 1
        self.governor.record_usage(analysis_resp.tokens_in, analysis_resp.tokens_out)

        # ── Step 2: Critique (prefer local) ──
        if not self.governor.check_budget():
            logger.warning("Token budget exhausted after analysis — skipping critique")
            result.synthesis = DebateStep(
                name="synthesis", content=analysis_resp.content,
                model=analysis_resp.model, tokens_in=0, tokens_out=0,
            )
            result.total_latency_ms = (time.time() - start) * 1000
            return result

        logger.info("Debate Step 2: Critique")
        critique_prompt = CRITIQUE_PROMPT.format(
            query=query,
            analysis=analysis_resp.content[:2000],
        )
        critique_prompt = self.governor.compress_context(
            critique_prompt, self.governor.budget.per_call_input
        )

        t0 = time.time()
        critique_resp = await self.router.generate(
            prompt=critique_prompt,
            system_instruction=CRITIQUE_SYSTEM,
            max_tokens=min(1000, self.governor.budget.allowed_output()),
            temperature=0.5,
            prefer_local=True,  # Use Groq for critique if available
        )
        critique_time = (time.time() - t0) * 1000

        if critique_resp.ok:
            result.critique = DebateStep(
                name="critique",
                content=critique_resp.content,
                model=critique_resp.model,
                tokens_in=critique_resp.tokens_in,
                tokens_out=critique_resp.tokens_out,
                latency_ms=critique_time,
            )
            result.api_calls += 1
            self.governor.record_usage(critique_resp.tokens_in, critique_resp.tokens_out)
        else:
            logger.warning(f"Critique failed: {critique_resp.error} — proceeding to synthesis without it")

        # ── Step 3: Synthesis (always use primary model — Gemini Flash) ──
        if not self.governor.check_budget():
            logger.warning("Token budget exhausted — using analysis as final output")
            result.synthesis = DebateStep(
                name="synthesis", content=analysis_resp.content,
                model=analysis_resp.model, tokens_in=0, tokens_out=0,
            )
            result.total_latency_ms = (time.time() - start) * 1000
            return result

        logger.info("Debate Step 3: Synthesis")
        critique_text = result.critique.content if result.critique else "No critique available — proceed with analysis only."
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            query=query,
            context=context_block if context_block else "No additional context.",
            analysis=analysis_resp.content[:2000],
            critique=critique_text[:1500],
        )
        synthesis_prompt = self.governor.compress_context(
            synthesis_prompt, self.governor.budget.per_call_input
        )

        t0 = time.time()
        synthesis_resp = await self.router.generate(
            prompt=synthesis_prompt,
            system_instruction=SYNTHESIS_SYSTEM,
            max_tokens=min(2048, self.governor.budget.allowed_output()),
            temperature=0.3,
            prefer_local=False,  # Always use best model for synthesis
        )
        synthesis_time = (time.time() - t0) * 1000

        if synthesis_resp.ok:
            result.synthesis = DebateStep(
                name="synthesis",
                content=synthesis_resp.content,
                model=synthesis_resp.model,
                tokens_in=synthesis_resp.tokens_in,
                tokens_out=synthesis_resp.tokens_out,
                latency_ms=synthesis_time,
            )
            result.api_calls += 1
            self.governor.record_usage(synthesis_resp.tokens_in, synthesis_resp.tokens_out)
        else:
            # Fallback: return analysis if synthesis fails
            logger.error(f"Synthesis failed: {synthesis_resp.error}")
            result.synthesis = DebateStep(
                name="synthesis_fallback",
                content=analysis_resp.content,
                model=analysis_resp.model,
                tokens_in=0, tokens_out=0,
            )

        # ── Finalize ──
        result.total_tokens_in = sum(
            s.tokens_in for s in [result.analysis, result.critique, result.synthesis] if s
        )
        result.total_tokens_out = sum(
            s.tokens_out for s in [result.analysis, result.critique, result.synthesis] if s
        )
        result.total_latency_ms = (time.time() - start) * 1000

        logger.info(
            f"Debate complete: {result.api_calls} API calls, "
            f"{result.total_tokens_in}+{result.total_tokens_out} tokens, "
            f"{result.total_latency_ms:.0f}ms"
        )
        return result
