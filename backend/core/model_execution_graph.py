"""
============================================================
Model Execution Graph — Sentinel-E Cognitive Engine v7.0
============================================================
Runs all enabled models per round. No filtering. No exclusion.
Failed models appear as status='failed' in output.

Usage:
    graph = ModelExecutionGraph(model_bridge)
    outputs = await graph.run_round(round_number=1, prompt="...", previous_outputs=[])

All models execute. Period. Zero silent filtering.
============================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from core.ensemble_schemas import (
    MIN_MODELS,
    ModelStatus,
    StanceVector,
    StructuredModelOutput,
    EnsembleFailure,
    EnsembleFailureCode,
)

logger = logging.getLogger("sentinel.execution_graph")


# ── Structured output extraction prompt ──────────────────────

STRUCTURED_OUTPUT_PROMPT = """You are part of a multi-model cognitive ensemble. You MUST respond with EXACTLY this structure:

**POSITION**: [Your clear thesis/answer in 2-3 sentences]

**REASONING**: [Step-by-step reasoning chain, 3-5 steps]

**ASSUMPTIONS**: [Key assumptions you are making]

**WEAKNESSES**: [Weaknesses or gaps in your own reasoning]

**REBUTTALS**: [If previous model outputs provided, rebut their weak points. Otherwise write "N/A"]

**CONFIDENCE**: [A number between 0.0 and 1.0]

**STANCE**: certainty=[0-1] specificity=[0-1] risk_tolerance=[0-1] evidence_reliance=[0-1] novelty=[0-1]

Do NOT deviate from this structure. Every section is mandatory."""


def _build_round_prompt(
    query: str,
    round_number: int,
    previous_outputs: List[StructuredModelOutput],
    image_b64: Optional[str] = None,
    model_supports_vision: bool = False,
) -> Tuple[str, str]:
    """Build system prompt and user prompt for a model in this round."""

    system = STRUCTURED_OUTPUT_PROMPT

    parts = [f"## Query\n{query}"]

    if round_number > 1 and previous_outputs:
        parts.append(f"\n## Previous Round Outputs (Round {round_number - 1})")
        for out in previous_outputs:
            status_tag = "✓" if out.succeeded else "✗ FAILED"
            parts.append(
                f"\n### [{status_tag}] {out.model_id}\n"
                f"**Position**: {out.position or '(no position)'}\n"
                f"**Reasoning**: {out.reasoning or '(none)'}\n"
                f"**Assumptions**: {out.assumptions or '(none)'}\n"
                f"**Weaknesses**: {out.weaknesses_found or '(none)'}\n"
                f"**Confidence**: {out.confidence}\n"
            )
        parts.append(
            "\nYou are in Round {round}. Review the above, "
            "then provide YOUR position using the required structure. "
            "If you disagree, explain why. If you shifted position, state that.".format(
                round=round_number
            )
        )

    if image_b64 and model_supports_vision:
        parts.append("\n[An image has been attached. Analyze it as part of your reasoning.]")
    elif image_b64 and not model_supports_vision:
        parts.append(
            "\n[NOTE: An image was attached to this query, but your model does not support "
            "vision input. Base your reasoning on the text only.]"
        )

    return system, "\n".join(parts)


def _parse_structured_output(
    raw: str, model_id: str, round_number: int, latency_ms: float
) -> StructuredModelOutput:
    """Parse a model's raw text into StructuredModelOutput."""
    import re

    def _extract(label: str) -> str:
        pattern = rf"\*\*{label}\*\*\s*:?\s*(.*?)(?=\n\*\*|\Z)"
        m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    position = _extract("POSITION")
    reasoning = _extract("REASONING")
    assumptions = _extract("ASSUMPTIONS")
    weaknesses = _extract("WEAKNESSES")
    rebuttals = _extract("REBUTTALS")

    # Confidence
    conf_text = _extract("CONFIDENCE")
    try:
        conf_val = float(re.search(r"[\d.]+", conf_text).group())
        conf_val = max(0.0, min(1.0, conf_val))
    except Exception:
        conf_val = 0.5

    # Stance vector
    stance_text = _extract("STANCE")
    stance = StanceVector()
    if stance_text:
        for dim in ["certainty", "specificity", "risk_tolerance", "evidence_reliance", "novelty"]:
            m = re.search(rf"{dim}\s*=\s*([\d.]+)", stance_text, re.IGNORECASE)
            if m:
                try:
                    setattr(stance, dim, max(0.0, min(1.0, float(m.group(1)))))
                except Exception:
                    pass

    # If nothing parsed, use raw as position
    if not position and raw.strip():
        position = raw.strip()[:500]

    return StructuredModelOutput(
        model_id=model_id,
        round=round_number,
        position=position,
        reasoning=reasoning,
        assumptions=assumptions,
        weaknesses_found=weaknesses,
        rebuttals=rebuttals,
        confidence=conf_val,
        stance_vector=stance,
        status="success",
        raw_output=raw,
        latency_ms=latency_ms,
    )


class ModelExecutionGraph:
    """
    Executes ALL enabled models per round. No filtering. No exclusion.

    Usage:
        graph = ModelExecutionGraph(model_bridge, model_statuses)
        outputs = await graph.run_round(1, "query", [], image_b64=None)

    Rules:
        - Every model in the registry runs.
        - Failed models get status='failed' with error reason.
        - Zero score ≠ failure.
        - Never drop a model from the output list.
    """

    def __init__(
        self,
        model_bridge,
        model_statuses: List[ModelStatus],
        timeout_seconds: float = 120.0,
    ):
        self.bridge = model_bridge
        self.model_statuses = {s.model_id: s for s in model_statuses}
        self.timeout = timeout_seconds

    async def run_round(
        self,
        round_number: int,
        prompt: str,
        previous_outputs: List[StructuredModelOutput],
        image_b64: Optional[str] = None,
    ) -> List[StructuredModelOutput]:
        """
        Execute all models for a single round. Returns one output per model.
        Failed models are included with status='failed'.
        """
        model_ids = list(self.model_statuses.keys())
        if not model_ids:
            raise EnsembleFailure(
                EnsembleFailureCode.INSUFFICIENT_MODELS,
                "No models registered in execution graph",
                models_available=0,
            )

        tasks = []
        for mid in model_ids:
            ms = self.model_statuses[mid]
            tasks.append(
                self._execute_single_model(
                    model_id=mid,
                    round_number=round_number,
                    prompt=prompt,
                    previous_outputs=previous_outputs,
                    image_b64=image_b64,
                    supports_vision=ms.supports_vision,
                    available=ms.available,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    async def _execute_single_model(
        self,
        model_id: str,
        round_number: int,
        prompt: str,
        previous_outputs: List[StructuredModelOutput],
        image_b64: Optional[str] = None,
        supports_vision: bool = False,
        available: bool = True,
    ) -> StructuredModelOutput:
        """Execute a single model, always returning a StructuredModelOutput."""

        # If provider validation marked unavailable
        if not available:
            ms = self.model_statuses.get(model_id)
            return StructuredModelOutput(
                model_id=model_id,
                round=round_number,
                status="failed",
                error=ms.error if ms else "Model marked unavailable by provider validation",
            )

        system_prompt, user_prompt = _build_round_prompt(
            query=prompt,
            round_number=round_number,
            previous_outputs=previous_outputs,
            image_b64=image_b64,
            model_supports_vision=supports_vision,
        )

        t0 = time.perf_counter()
        try:
            raw_output = await asyncio.wait_for(
                self.bridge.call_model(
                    model_id=model_id,
                    prompt=user_prompt,
                    system_role=system_prompt,
                ),
                timeout=self.timeout,
            )
            latency = (time.perf_counter() - t0) * 1000

            if not raw_output or not raw_output.strip():
                return StructuredModelOutput(
                    model_id=model_id,
                    round=round_number,
                    status="failed",
                    error="Empty response from model",
                    latency_ms=latency,
                )

            return _parse_structured_output(raw_output, model_id, round_number, latency)

        except asyncio.TimeoutError:
            latency = (time.perf_counter() - t0) * 1000
            logger.warning(f"Model {model_id} timed out in round {round_number}")
            return StructuredModelOutput(
                model_id=model_id,
                round=round_number,
                status="failed",
                error="timeout",
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"Model {model_id} failed in round {round_number}: {e}")
            return StructuredModelOutput(
                model_id=model_id,
                round=round_number,
                status="failed",
                error=str(e),
                latency_ms=latency,
            )
