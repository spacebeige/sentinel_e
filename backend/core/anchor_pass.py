"""
============================================================
Anchor Model Pass — Sentinel-E v2
============================================================
Post-debate evaluation by heavyweight reasoning models.

Anchor models do NOT participate in debate rounds.
They run ONCE after the debate to:
  1. Evaluate debate quality
  2. Audit reasoning logic
  3. Produce final synthesis
  4. Calibrate confidence

Pipeline position:
    Debate Engine → Consensus Engine → Metrics Engine
    → ANCHOR PASS → Visualization Engine

Supported anchors (optional — graceful skip if no keys):
    anchor-llama70b   → Groq llama-3.3-70b-versatile
    anchor-claude     → Anthropic Claude (via OpenRouter)
    anchor-gpt4       → OpenAI GPT-4.1 (via OpenRouter)
    anchor-deepseek   → DeepSeek-V3 (via OpenRouter)

Each anchor runs independently and produces an AnchorEvaluation.
Multiple anchors produce a combined AnchorPassResult with
cross-anchor agreement metrics.
============================================================
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("AnchorPass")


# ============================================================
# Anchor Model Specifications
# ============================================================

@dataclass
class AnchorModelSpec:
    """Specification for an anchor evaluation model."""
    name: str
    model_id: str
    provider: str          # groq | openrouter
    api_key_env: str       # env var for this anchor's key
    api_base_url: str
    max_output_tokens: int = 2048
    temperature: float = 0.2


ANCHOR_REGISTRY: Dict[str, AnchorModelSpec] = {
    "anchor-llama70b": AnchorModelSpec(
        name="Llama 3.3 70B (Anchor)",
        model_id="llama-3.3-70b-versatile",
        provider="groq",
        api_key_env="ANCHOR_LLAMA70B_API_KEY",
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        max_output_tokens=2048,
    ),
    "anchor-claude": AnchorModelSpec(
        name="Claude Sonnet (Anchor)",
        model_id="anthropic/claude-sonnet-4-20250514",
        provider="openrouter",
        api_key_env="ANCHOR_CLAUDE_API_KEY",
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        max_output_tokens=2048,
    ),
    "anchor-gpt4": AnchorModelSpec(
        name="GPT-4.1 (Anchor)",
        model_id="openai/gpt-4.1",
        provider="openrouter",
        api_key_env="ANCHOR_GPT4_API_KEY",
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        max_output_tokens=2048,
    ),
    "anchor-deepseek": AnchorModelSpec(
        name="DeepSeek V3 (Anchor)",
        model_id="deepseek/deepseek-chat-v3-0324:free",
        provider="openrouter",
        api_key_env="ANCHOR_DEEPSEEK_API_KEY",
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        max_output_tokens=2048,
    ),
}


# ============================================================
# Anchor Evaluation Prompt
# ============================================================

ANCHOR_EVAL_PROMPT = """You are an expert AI reasoning evaluator performing a POST-DEBATE ANCHOR EVALUATION.

You did NOT participate in the debate. You are an independent judge.

ORIGINAL QUESTION:
{query}

DEBATE SYNTHESIS:
{synthesis}

DEBATE METRICS:
- Models participated: {model_count}
- Rounds completed: {round_count}
- Consensus strength: {consensus_strength}
- Disagreement level: {disagreement}
- Tokens spent: {tokens_spent}

KEY POSITIONS:
{positions_summary}

{evidence_section}

YOUR TASK:
Evaluate this debate independently. You must:

1. QUALITY ASSESSMENT: Rate the debate quality (0.0-1.0). Consider:
   - Were the arguments rigorous and evidence-based?
   - Did models genuinely challenge each other?
   - Was the synthesis fair and well-supported?

2. REASONING AUDIT: Identify logical flaws, unsupported claims, or circular reasoning.

3. FINAL SYNTHESIS: Provide your own independent answer to the original question,
   informed by but not bound to the debate outcome.

4. CONFIDENCE CALIBRATION: Rate your confidence in the debate conclusion (0.0-1.0).
   Consider whether the consensus was genuine or premature.

Respond with EXACTLY this structure:

QUALITY_SCORE: [0.0-1.0]

REASONING_FLAWS:
- [flaw 1]
- [flaw 2]

ANCHOR_SYNTHESIS: [Your independent answer, 2-4 paragraphs]

CONFIDENCE: [0.0-1.0]

VERDICT: [AGREE | PARTIALLY_AGREE | DISAGREE] with the debate consensus

VERDICT_REASON: [One sentence explaining your verdict]"""


# ============================================================
# Data Structures
# ============================================================

@dataclass
class AnchorEvaluation:
    """Result from a single anchor model evaluation."""
    anchor_model: str
    anchor_name: str
    quality_score: float = 0.5
    reasoning_flaws: List[str] = field(default_factory=list)
    synthesis: str = ""
    confidence: float = 0.5
    verdict: str = "PARTIALLY_AGREE"
    verdict_reason: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class AnchorPassResult:
    """Combined result from all anchor evaluations."""
    evaluations: List[AnchorEvaluation] = field(default_factory=list)
    anchor_count: int = 0
    avg_quality_score: float = 0.0
    avg_confidence: float = 0.0
    anchor_agreement: float = 0.0  # Cross-anchor agreement (0-1)
    dominant_verdict: str = "PARTIALLY_AGREE"
    combined_synthesis: str = ""


# ============================================================
# Anchor Pass Engine
# ============================================================

class AnchorPassEngine:
    """
    Runs post-debate anchor evaluations.

    Usage:
        engine = AnchorPassEngine()
        if engine.has_anchors():
            result = await engine.evaluate(
                query="...",
                debate_synthesis="...",
                debate_metrics={...},
                positions_summary="...",
            )
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._available_anchors = self._discover_anchors()

    def _discover_anchors(self) -> Dict[str, AnchorModelSpec]:
        """Find which anchor models have API keys configured."""
        available = {}
        for key, spec in ANCHOR_REGISTRY.items():
            api_key = os.getenv(spec.api_key_env, "").strip()
            if api_key:
                available[key] = spec
                logger.info(f"Anchor '{key}' ({spec.name}): available")
            else:
                logger.debug(f"Anchor '{key}': no key ({spec.api_key_env})")
        if not available:
            logger.info("No anchor models configured — anchor pass will be skipped")
        return available

    def has_anchors(self) -> bool:
        """Check if any anchor models are available."""
        return len(self._available_anchors) > 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def evaluate(
        self,
        query: str,
        debate_synthesis: str,
        debate_metrics: Dict[str, Any],
        positions_summary: str,
        evidence_summary: str = "",
    ) -> AnchorPassResult:
        """
        Run all available anchor models in parallel.
        Each anchor independently evaluates the debate output.
        """
        if not self._available_anchors:
            return AnchorPassResult()

        evidence_section = ""
        if evidence_summary:
            evidence_section = f"EVIDENCE GATHERED:\n{evidence_summary}"

        prompt = ANCHOR_EVAL_PROMPT.format(
            query=query,
            synthesis=debate_synthesis or "(No synthesis produced)",
            model_count=debate_metrics.get("model_count", 0),
            round_count=debate_metrics.get("round_count", 0),
            consensus_strength=debate_metrics.get("consensus_strength", "unknown"),
            disagreement=debate_metrics.get("disagreement", "unknown"),
            tokens_spent=debate_metrics.get("tokens_spent", 0),
            positions_summary=positions_summary or "(No positions available)",
            evidence_section=evidence_section,
        )

        tasks = [
            self._invoke_anchor(key, spec, prompt)
            for key, spec in self._available_anchors.items()
        ]

        evaluations = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for eval_result in evaluations:
            if isinstance(eval_result, Exception):
                logger.warning(f"Anchor evaluation failed: {eval_result}")
                continue
            if eval_result and eval_result.success:
                results.append(eval_result)

        return self._combine_results(results)

    async def _invoke_anchor(
        self,
        anchor_key: str,
        spec: AnchorModelSpec,
        prompt: str,
    ) -> AnchorEvaluation:
        """Invoke a single anchor model and parse its response."""
        api_key = os.getenv(spec.api_key_env, "")
        if not api_key:
            return AnchorEvaluation(
                anchor_model=anchor_key,
                anchor_name=spec.name,
                success=False,
                error="No API key",
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if spec.provider == "openrouter":
            headers["HTTP-Referer"] = "https://sentinel-e.vercel.app"
            headers["X-Title"] = "Sentinel-E Anchor Pass"

        payload = {
            "model": spec.model_id,
            "messages": [
                {"role": "system", "content": "You are an expert AI reasoning evaluator."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": spec.max_output_tokens,
            "temperature": spec.temperature,
        }

        start = time.monotonic()
        try:
            session = await self._get_session()
            async with session.post(
                spec.api_base_url,
                json=payload,
                headers=headers,
            ) as resp:
                latency = (time.monotonic() - start) * 1000

                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        f"Anchor '{anchor_key}' returned {resp.status}: {body[:200]}"
                    )
                    return AnchorEvaluation(
                        anchor_model=anchor_key,
                        anchor_name=spec.name,
                        latency_ms=latency,
                        success=False,
                        error=f"HTTP {resp.status}",
                    )

                data = await resp.json()
                raw_output = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                evaluation = self._parse_anchor_output(
                    anchor_key, spec.name, raw_output, latency
                )
                logger.info(
                    f"Anchor '{anchor_key}': quality={evaluation.quality_score:.2f}, "
                    f"confidence={evaluation.confidence:.2f}, verdict={evaluation.verdict}, "
                    f"latency={latency:.0f}ms"
                )
                return evaluation

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            logger.warning(f"Anchor '{anchor_key}' timed out after {latency:.0f}ms")
            return AnchorEvaluation(
                anchor_model=anchor_key,
                anchor_name=spec.name,
                latency_ms=latency,
                success=False,
                error="Timeout",
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning(f"Anchor '{anchor_key}' error: {exc}")
            return AnchorEvaluation(
                anchor_model=anchor_key,
                anchor_name=spec.name,
                latency_ms=latency,
                success=False,
                error=str(exc),
            )

    def _parse_anchor_output(
        self,
        anchor_key: str,
        anchor_name: str,
        raw: str,
        latency_ms: float,
    ) -> AnchorEvaluation:
        """Parse structured anchor output into AnchorEvaluation."""
        import re

        quality = 0.5
        confidence = 0.5
        verdict = "PARTIALLY_AGREE"
        verdict_reason = ""
        synthesis = ""
        flaws = []

        # Parse QUALITY_SCORE
        m = re.search(r"QUALITY_SCORE:\s*([\d.]+)", raw)
        if m:
            quality = min(1.0, max(0.0, float(m.group(1))))

        # Parse CONFIDENCE
        m = re.search(r"CONFIDENCE:\s*([\d.]+)", raw)
        if m:
            confidence = min(1.0, max(0.0, float(m.group(1))))

        # Parse VERDICT
        m = re.search(r"VERDICT:\s*(AGREE|PARTIALLY_AGREE|DISAGREE)", raw)
        if m:
            verdict = m.group(1)

        # Parse VERDICT_REASON
        m = re.search(r"VERDICT_REASON:\s*(.+?)(?:\n|$)", raw)
        if m:
            verdict_reason = m.group(1).strip()

        # Parse ANCHOR_SYNTHESIS
        m = re.search(
            r"ANCHOR_SYNTHESIS:\s*(.+?)(?=\nCONFIDENCE:|\nVERDICT:|\Z)",
            raw,
            re.DOTALL,
        )
        if m:
            synthesis = m.group(1).strip()

        # Parse REASONING_FLAWS
        m = re.search(
            r"REASONING_FLAWS:\s*(.+?)(?=\nANCHOR_SYNTHESIS:|\Z)",
            raw,
            re.DOTALL,
        )
        if m:
            flaw_text = m.group(1).strip()
            flaws = [
                line.lstrip("- ").strip()
                for line in flaw_text.split("\n")
                if line.strip().startswith("-")
            ]

        return AnchorEvaluation(
            anchor_model=anchor_key,
            anchor_name=anchor_name,
            quality_score=quality,
            reasoning_flaws=flaws,
            synthesis=synthesis,
            confidence=confidence,
            verdict=verdict,
            verdict_reason=verdict_reason,
            latency_ms=latency_ms,
            success=True,
        )

    def _combine_results(
        self, evaluations: List[AnchorEvaluation]
    ) -> AnchorPassResult:
        """Combine multiple anchor evaluations into a single result."""
        if not evaluations:
            return AnchorPassResult()

        n = len(evaluations)
        avg_quality = sum(e.quality_score for e in evaluations) / n
        avg_confidence = sum(e.confidence for e in evaluations) / n

        # Cross-anchor agreement: how many share the same verdict
        verdicts = [e.verdict for e in evaluations]
        most_common = max(set(verdicts), key=verdicts.count)
        agreement = verdicts.count(most_common) / n

        # Combined synthesis: use the highest-quality anchor's synthesis
        best = max(evaluations, key=lambda e: e.quality_score)
        combined_synthesis = best.synthesis

        return AnchorPassResult(
            evaluations=evaluations,
            anchor_count=n,
            avg_quality_score=round(avg_quality, 4),
            avg_confidence=round(avg_confidence, 4),
            anchor_agreement=round(agreement, 4),
            dominant_verdict=most_common,
            combined_synthesis=combined_synthesis,
        )


# ── Module-level singleton ──────────────────────────────────

_anchor_engine: Optional[AnchorPassEngine] = None


def get_anchor_engine() -> AnchorPassEngine:
    """Get or create the singleton AnchorPassEngine."""
    global _anchor_engine
    if _anchor_engine is None:
        _anchor_engine = AnchorPassEngine()
    return _anchor_engine
