"""
============================================================
Structured Debate Engine — Sentinel-E v6.0
============================================================
Minimum 3-round structured debate across ALL enabled models.

Every model produces structured output per round.
No mode awareness. No provider awareness.
Pure adversarial reasoning with position tracking.

Data Flow:
    Round 1: Independent positions (parallel)
    Round 2: Rebuttals + position shifts (parallel, with transcript)
    Round 3+: Convergence/divergence tracking (parallel, with transcript)
    → Shift table → Consensus detection → DebateResult
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import numpy as np
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from core.ensemble_schemas import (
    MIN_DEBATE_ROUNDS,
    TOTAL_DEBATE_TOKEN_BUDGET,
    ROUND_BUDGET_SPLIT,
    ROUND_HARD_CAPS,
    CONSENSUS_EARLY_STOP,
    STABILITY_EARLY_STOP,
    MIN_REMAINING_BUDGET,
    MAX_SUMMARY_TOKENS,
    DebatePosition,
    DebateResult,
    DebateRound,
    ShiftRecord,
    StanceVector,
    StructuredModelOutput,
    EnsembleFailure,
    EnsembleFailureCode,
)

logger = logging.getLogger("StructuredDebateEngine")


# ============================================================
# System Prompts — Structured Output Enforcement
# ============================================================

STRUCTURED_ROUND_1 = """You are one model in a multi-model adversarial reasoning system.
Your job is to think INDEPENDENTLY, argue your position, challenge others, and refine under pressure.

DEBATE TOPIC: {query}

RULES:
- Think for yourself. Do NOT echo or defer to other models.
- Be adversarial but rational. Attack weak reasoning, not models.
- State your assumptions explicitly so others can challenge them.
- Identify risks in your own position before opponents do.
- Confidence must reflect genuine certainty — do NOT inflate.

Respond with EXACTLY this structure (use these exact headers):

POSITION: [Your clear thesis in 1-2 sentences — what you believe and why]

ARGUMENT: [Step-by-step reasoning chain. Be specific, evidence-based, and logically rigorous.]

ASSUMPTIONS: [List each assumption explicitly, prefixed with "- "]
- assumption 1
- assumption 2

RISKS: [What could go wrong if your position is adopted? Prefixed with "- "]
- risk 1
- risk 2

VULNERABILITIES: [Self-identified weaknesses in your reasoning, prefixed with "- "]
- vulnerability 1
- vulnerability 2

CONFIDENCE: [A number between 0.0 and 1.0 — honest self-assessed certainty]

STANCE: [Rate each dimension 0.0-1.0]
certainty: [0.0-1.0]
specificity: [0.0-1.0]
risk_tolerance: [0.0-1.0]
evidence_reliance: [0.0-1.0]
novelty: [0.0-1.0]"""

STRUCTURED_ROUND_N = """You are in round {round_number} of a multi-model adversarial debate.

DEBATE TOPIC: {query}

PREVIOUS DEBATE TRANSCRIPT:
{transcript}

YOUR PREVIOUS POSITION: {own_previous}

RULES FOR THIS ROUND:
- Read every other model's argument. Find what is weak, missing, or wrong.
- Rebut specific claims with logic and evidence — not dismissal.
- If an opponent made a strong point, ACKNOWLEDGE it and adjust.
- Track whether your position shifted and why (or why not).
- Your confidence MUST change if evidence warrants it.

Respond with EXACTLY this structure:

REBUTTALS: [Address specific opponent arguments, prefixed with "- "]
- [Model X] claimed Y — this fails because Z
- [Model W] assumed V — this is incorrect because...

WEAKNESSES_FOUND: [Weaknesses you identified in OTHER models' reasoning, prefixed with "- "]
- [Model X]: weakness description
- [Model W]: weakness description

POSITION: [Your current thesis — updated if the debate warrants it]

ARGUMENT: [Updated reasoning incorporating debate insights and rebuttal responses]

ASSUMPTIONS: [Current assumptions, prefixed with "- "]
- assumption 1

RISKS: [Current risks if your position is adopted, prefixed with "- "]
- risk 1

VULNERABILITIES: [Current self-identified weaknesses, prefixed with "- "]
- vulnerability 1

POSITION_SHIFTED: [YES or NO — did your position change from last round?]

SHIFT_REASON: [If shifted, explain what argument convinced you. If not, explain why your position held under pressure.]

CONFIDENCE: [Updated confidence 0.0-1.0 — must reflect debate dynamics]

STANCE: [Updated dimensional ratings]
certainty: [0.0-1.0]
specificity: [0.0-1.0]
risk_tolerance: [0.0-1.0]
evidence_reliance: [0.0-1.0]
novelty: [0.0-1.0]"""


STRUCTURED_ROUND_FINAL = """You are in the FINAL round ({round_number}) of a multi-model adversarial debate.

DEBATE TOPIC: {query}

PREVIOUS DEBATE TRANSCRIPT:
{transcript}

YOUR PREVIOUS POSITION: {own_previous}

THIS IS THE FINAL ROUND. Your job is to deliver your REFINED, FINAL stance.

RULES FOR THE FINAL ROUND:
- Synthesize everything you've learned from the debate.
- Acknowledge the strongest points made by opponents.
- State your FINAL position clearly — this is your definitive answer.
- Your confidence must reflect ALL evidence and arguments presented.
- Identify any remaining unresolved disagreements.
- Be honest about what the debate revealed — don't just repeat Round 1.

Respond with EXACTLY this structure:

KEY_TAKEAWAYS: [What the debate revealed, prefixed with "- "]
- takeaway from the debate

CONCESSIONS: [Points from opponents you now accept, prefixed with "- "]
- concession 1

POSITION: [Your FINAL, refined thesis — incorporating debate insights]

ARGUMENT: [Your strongest, most refined reasoning — battle-tested through debate]

REMAINING_DISAGREEMENTS: [Unresolved points where models still diverge, prefixed with "- "]
- disagreement 1

ASSUMPTIONS: [Final assumptions, prefixed with "- "]
- assumption 1

RISKS: [Final risks, prefixed with "- "]
- risk 1

VULNERABILITIES: [Remaining weaknesses, prefixed with "- "]
- vulnerability 1

POSITION_SHIFTED: [YES or NO — did your position change from last round?]

SHIFT_REASON: [If shifted, explain what argument convinced you. If not, explain why your position held under pressure.]

CONFIDENCE: [Final confidence 0.0-1.0 — must reflect the full debate]

STANCE: [Final dimensional ratings]
certainty: [0.0-1.0]
specificity: [0.0-1.0]
risk_tolerance: [0.0-1.0]
evidence_reliance: [0.0-1.0]
novelty: [0.0-1.0]"""


# Type for model caller function (accepts optional max_tokens kwarg)
ModelCaller = Callable[..., Coroutine[Any, Any, str]]


class StructuredDebateEngine:
    """
    Executes structured multi-model debate with budget governance.

    Token budget lifecycle:
      - Hard cap per model per round: R1=350, R2=250, R3=200 tokens.
      - Total debate budget: 5000 tokens.
      - Early stop: consensus >= 60% after R1, stability >= 65% after R2.
      - Remaining budget < 800 → forced stop (Budget-Limited Completion).
      - Models with governed_max_tokens < 200 are skipped, not failed.

    Args:
        call_model: async function(model_id, prompt, system_role, *, max_tokens=None) -> str
        get_enabled_models: function() -> List[dict] with {id, name, role}
    """

    def __init__(
        self,
        call_model: ModelCaller,
        get_enabled_models: Callable[[], List[Dict[str, str]]],
    ):
        self._call_model = call_model
        self._get_enabled_models = get_enabled_models

    async def run_debate(
        self,
        query: str,
        rounds: int = MIN_DEBATE_ROUNDS,
        initial_outputs: Optional[List[StructuredModelOutput]] = None,
    ) -> DebateResult:
        """
        Execute structured debate with token budget lifecycle management.

        Budget rules:
          - Hard cap per model: R1=350, R2=250, R3=200 (not proportional).
          - Total debate budget: 5000 tokens.
          - Early stop after R1 if consensus >= 60%.
          - Early stop after R2 if stability >= 65%.
          - Forced stop if remaining budget < 800.
          - Models with budget < 200 are skipped (not marked as failure).

        Args:
            query: The user's question
            rounds: Number of rounds (minimum 2, enforced)
            initial_outputs: Pre-computed structured outputs from Phase 1

        Returns:
            DebateResult with all rounds, shifts, and consensus.

        Raises:
            EnsembleFailure if <3 models or <2 rounds complete.
        """
        rounds = max(rounds, MIN_DEBATE_ROUNDS)
        models = self._get_enabled_models()

        if len(models) < 3:
            raise EnsembleFailure(
                code=EnsembleFailureCode.INSUFFICIENT_MODELS,
                message=f"Debate requires minimum 3 models, got {len(models)}",
                models_available=len(models),
            )

        # ── Adaptive Token Budget ──────────────────────────────
        from core.ensemble_schemas import get_adaptive_budget
        budget_config = get_adaptive_budget(query)
        remaining_budget = budget_config["total_budget"]
        adaptive_round_caps = budget_config["round_caps"]
        total_tokens_spent = 0
        budget_constrained = False
        num_models = len(models)

        logger.info(
            f"Adaptive budget: multiplier={budget_config['multiplier']}, "
            f"total={budget_config['total_budget']}, "
            f"round_caps={adaptive_round_caps}"
        )

        debate_rounds: List[DebateRound] = []
        all_shifts: List[ShiftRecord] = []
        transcript_parts: List[str] = []
        prev_positions: Dict[str, DebatePosition] = {}
        # Track persistently failed models to exclude from subsequent rounds
        failed_model_ids: set = set()

        for round_num in range(1, rounds + 1):
            # ── Budget gate: stop if remaining < MIN_REMAINING_BUDGET ──
            if remaining_budget < MIN_REMAINING_BUDGET and len(debate_rounds) >= MIN_DEBATE_ROUNDS:
                logger.warning(
                    f"Budget exhausted after round {round_num - 1}: "
                    f"remaining={remaining_budget}, spent={total_tokens_spent}. "
                    f"Returning Budget-Limited Completion."
                )
                budget_constrained = True
                break

            # ── Adaptive hard cap per model ──
            per_model_budget = adaptive_round_caps.get(round_num, adaptive_round_caps.get(3, 200))

            # ── Filter out persistently failed models ──────────
            active_models = [m for m in models if m["id"] not in failed_model_ids]
            if len(active_models) < 2 and len(debate_rounds) >= 1:
                logger.warning(
                    f"Too few active models ({len(active_models)}) after failures. "
                    f"Ending debate after {len(debate_rounds)} rounds."
                )
                break

            logger.info(
                f"Debate round {round_num}/{rounds}: "
                f"per_model_cap={per_model_budget}, remaining={remaining_budget}, "
                f"active_models={len(active_models)}/{len(models)}"
            )

            if round_num == 1 and initial_outputs:
                # Use pre-computed Phase 1 outputs (no extra tokens spent)
                round_result = self._convert_initial_outputs(
                    initial_outputs, round_num
                )
                # Estimate tokens from initial outputs
                round_tokens = sum(
                    len(p.argument or "") // 4
                    for p in round_result.positions
                    if p.status != "failed"
                )
            elif round_num == 1:
                round_result = await self._run_round_1(
                    query, active_models, max_tokens=per_model_budget
                )
                round_tokens = self._estimate_round_tokens(round_result)
            else:
                # Build compressed debate summary (≤ MAX_SUMMARY_TOKENS)
                transcript = self._compress_debate_summary(
                    debate_rounds, prev_positions
                )
                # Use final round prompt if this is the last planned round
                is_final = (round_num == rounds)
                round_result = await self._run_round_n(
                    query, active_models, round_num, transcript, prev_positions,
                    max_tokens=per_model_budget,
                    is_final_round=is_final,
                )
                round_tokens = self._estimate_round_tokens(round_result)

            # Track budget
            total_tokens_spent += round_tokens
            remaining_budget -= round_tokens

            # ── Self-Healing: substitute failed models ─────────
            round_result = await self._self_heal_round(
                round_result, query, round_num, models,
                prev_positions, per_model_budget,
                transcript=self._compress_debate_summary(debate_rounds, prev_positions) if round_num > 1 else "",
            )

            # ── Track persistent failures ──────────────────────
            # Models that fail even after self-healing are excluded from future rounds
            for pos in round_result.positions:
                if pos.status == "failed" or pos.position == "[MODEL FAILED]":
                    failed_model_ids.add(pos.model_id)
                    logger.warning(
                        f"Model '{pos.model_id}' marked unavailable — "
                        f"excluded from future rounds"
                    )

            # Track disagreement
            round_result.round_disagreement = self._compute_disagreement(
                round_result
            )

            # Track convergence delta
            if len(debate_rounds) > 0:
                prev_disagreement = debate_rounds[-1].round_disagreement
                round_result.convergence_delta = (
                    prev_disagreement - round_result.round_disagreement
                )

            # Detect conflicts
            round_result.key_conflicts = self._extract_conflicts(round_result)

            # Track shifts
            if round_num > 1:
                round_shifts = self._detect_shifts(
                    prev_positions, round_result, round_num
                )
                all_shifts.extend(round_shifts)

            # Update transcript (full version for internal use)
            transcript_parts.append(
                self._format_round_transcript(round_result)
            )

            # Update previous positions
            for pos in round_result.positions:
                prev_positions[pos.model_id] = pos

            debate_rounds.append(round_result)

            # ── Early Stop: Consensus after Round 1 ────────────
            # Only allow early-stop if MIN_DEBATE_ROUNDS already satisfied
            if round_num == 1 and len(debate_rounds) >= MIN_DEBATE_ROUNDS:
                consensus_score = self._compute_consensus_score(round_result)
                if consensus_score >= CONSENSUS_EARLY_STOP:
                    logger.info(
                        f"Early stop after round 1: consensus={consensus_score:.3f} "
                        f">= threshold={CONSENSUS_EARLY_STOP}. Skipping remaining rounds."
                    )
                    break

            # ── Early Stop: Stability after Round 2 ────────────
            if round_num == 2 and len(debate_rounds) >= MIN_DEBATE_ROUNDS:
                stability_score = self._compute_stability_score(debate_rounds)
                if stability_score >= STABILITY_EARLY_STOP:
                    logger.info(
                        f"Early stop after round 2: stability={stability_score:.3f} "
                        f">= threshold={STABILITY_EARLY_STOP}. Skipping remaining rounds."
                    )
                    break

        # ── Conditional 3rd round ──────────────────────────────
        # Only extend if: budget allows, all models succeeded,
        # no credit/rate failures, and disagreement remains high
        if (
            len(debate_rounds) == 2
            and rounds <= 2
            and remaining_budget >= MIN_REMAINING_BUDGET
            and not budget_constrained
        ):
            last_round = debate_rounds[-1]
            all_succeeded = all(
                getattr(p, 'status', 'success') != 'failed'
                for p in last_round.positions
            )
            has_credit_failures = any(
                'credit' in (getattr(p, 'argument', '') or '').lower()
                or 'rate limit' in (getattr(p, 'argument', '') or '').lower()
                for rnd in debate_rounds for p in rnd.positions
                if getattr(p, 'status', 'success') == 'failed'
            )
            high_disagreement = last_round.round_disagreement > 0.3

            if all_succeeded and not has_credit_failures and high_disagreement:
                per_model_r3 = ROUND_HARD_CAPS.get(3, 200)

                logger.info(
                    f"Conditional 3rd round triggered: "
                    f"disagreement={last_round.round_disagreement:.3f}, "
                    f"per_model_cap={per_model_r3}"
                )
                transcript = self._compress_debate_summary(
                    debate_rounds, prev_positions
                )
                round_result = await self._run_round_n(
                    query, models, 3, transcript, prev_positions,
                    max_tokens=per_model_r3,
                    is_final_round=True,
                )
                round_tokens = self._estimate_round_tokens(round_result)
                total_tokens_spent += round_tokens
                remaining_budget -= round_tokens

                round_result.round_disagreement = self._compute_disagreement(
                    round_result
                )
                prev_disagreement = debate_rounds[-1].round_disagreement
                round_result.convergence_delta = (
                    prev_disagreement - round_result.round_disagreement
                )
                round_result.key_conflicts = self._extract_conflicts(round_result)
                round_shifts = self._detect_shifts(
                    prev_positions, round_result, 3
                )
                all_shifts.extend(round_shifts)
                transcript_parts.append(
                    self._format_round_transcript(round_result)
                )
                for pos in round_result.positions:
                    prev_positions[pos.model_id] = pos
                debate_rounds.append(round_result)

        # Validate minimum rounds completed
        if len(debate_rounds) < MIN_DEBATE_ROUNDS:
            raise EnsembleFailure(
                code=EnsembleFailureCode.INSUFFICIENT_ROUNDS,
                message=f"Only {len(debate_rounds)} rounds completed, minimum {MIN_DEBATE_ROUNDS} required",
                rounds_completed=len(debate_rounds),
                models_available=len(models),
            )

        # Compute consensus
        final_consensus, consensus_strength = self._compute_consensus(
            debate_rounds
        )

        # Unresolved conflicts
        unresolved = self._find_unresolved_conflicts(debate_rounds)

        # Compute drift/rift analytics
        drift_rift = self._compute_debate_drift_rift(debate_rounds)

        # Build analysis summary
        analysis_fields = self._build_analysis_summary(debate_rounds, all_shifts)

        # Append budget status if constrained
        if budget_constrained:
            logger.info(
                f"Budget-Constrained Completion: {total_tokens_spent} tokens "
                f"spent across {len(debate_rounds)} rounds"
            )

        return DebateResult(
            rounds=debate_rounds,
            total_rounds=len(debate_rounds),
            budget_constrained=budget_constrained,
            tokens_spent=total_tokens_spent,
            shift_table=all_shifts,
            final_consensus=final_consensus,
            consensus_strength=consensus_strength,
            unresolved_conflicts=unresolved,
            drift_index=drift_rift["drift_index"],
            rift_index=drift_rift["rift_index"],
            confidence_spread=drift_rift["confidence_spread"],
            fragility_score=drift_rift["fragility_score"],
            per_model_drift=drift_rift["per_model_drift"],
            per_round_rift=drift_rift["per_round_rift"],
            per_round_disagreement=drift_rift["per_round_disagreement"],
            overall_confidence=drift_rift["overall_confidence"],
            conflict_axes=analysis_fields.get("conflict_axes", []),
            disagreement_strength=analysis_fields.get("disagreement_strength", 0.0),
            logical_stability=analysis_fields.get("logical_stability", 0.5),
            convergence_level=analysis_fields.get("convergence_level", "none"),
            convergence_detail=analysis_fields.get("convergence_detail", ""),
            strongest_argument=analysis_fields.get("strongest_argument", ""),
            weakest_argument=analysis_fields.get("weakest_argument", ""),
            synthesis=analysis_fields.get("synthesis", ""),
        )

    # ── Self-Healing ─────────────────────────────────────────

    async def _self_heal_round(
        self,
        round_result: DebateRound,
        query: str,
        round_num: int,
        models: List[Dict[str, str]],
        prev_positions: Dict[str, DebatePosition],
        per_model_budget: int,
        transcript: str = "",
    ) -> DebateRound:
        """
        Self-healing: retry failed models with fallback substitution.
        Uses MODEL_FALLBACK_MAP from the gateway to find substitutes.
        Never duplicates a model already in the round.
        """
        from metacognitive.cognitive_gateway import MODEL_FALLBACK_MAP, COGNITIVE_MODEL_REGISTRY

        failed_positions = [
            p for p in round_result.positions
            if p.status == "failed" or p.position == "[MODEL FAILED]"
        ]
        if not failed_positions:
            return round_result

        active_ids = {p.model_id for p in round_result.positions if p.status != "failed"}

        for failed_pos in failed_positions:
            fallback_key = MODEL_FALLBACK_MAP.get(failed_pos.model_id)
            if not fallback_key or fallback_key in active_ids:
                continue

            spec = COGNITIVE_MODEL_REGISTRY.get(fallback_key)
            if not spec or not spec.enabled:
                continue

            fallback_model = {"id": fallback_key, "name": spec.name, "role": spec.role.value}
            logger.info(
                f"Self-healing: substituting '{failed_pos.model_id}' "
                f"with fallback '{fallback_key}' in round {round_num}"
            )

            try:
                if round_num == 1:
                    system_prompt = STRUCTURED_ROUND_1.format(query=query)
                    result = await self._call_and_parse_round_1(
                        fallback_model, query, system_prompt, max_tokens=per_model_budget
                    )
                else:
                    own_prev = ""
                    system_prompt = STRUCTURED_ROUND_N.format(
                        query=query,
                        round_number=round_num,
                        transcript=transcript,
                        own_previous=own_prev,
                    )
                    result = await self._call_and_parse_round_n(
                        fallback_model, query, system_prompt, round_num, max_tokens=per_model_budget
                    )

                if result and result.position != "[MODEL FAILED]":
                    # Replace failed position with fallback result
                    idx = round_result.positions.index(failed_pos)
                    round_result.positions[idx] = result
                    active_ids.add(fallback_key)
                    logger.info(f"Self-healing: '{fallback_key}' succeeded as replacement")
            except Exception as e:
                logger.warning(f"Self-healing fallback '{fallback_key}' also failed: {e}")

        return round_result

    # ── Round Execution ──────────────────────────────────────

    async def _run_round_1(
        self, query: str, models: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> DebateRound:
        """Execute round 1: independent positions."""
        system_prompt = STRUCTURED_ROUND_1.format(query=query)

        tasks = [
            self._call_and_parse_round_1(model, query, system_prompt, max_tokens=max_tokens)
            for model in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        positions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model {models[i]['id']} failed in round 1: {result}")
                positions.append(self._failed_position(
                    model=models[i],
                    round_num=1,
                    reason=f"Model did not respond: {result}",
                ))
                continue
            if result is None:
                positions.append(self._failed_position(
                    model=models[i],
                    round_num=1,
                    reason="Model returned empty or invalid structured output",
                ))
                continue
            positions.append(result)

        return DebateRound(round_number=1, positions=positions)

    async def _run_round_n(
        self,
        query: str,
        models: List[Dict[str, str]],
        round_num: int,
        transcript: str,
        prev_positions: Dict[str, DebatePosition],
        max_tokens: Optional[int] = None,
        is_final_round: bool = False,
    ) -> DebateRound:
        """Execute round N: rebuttals and position updates.
        
        Uses STRUCTURED_ROUND_FINAL for the last round to elicit
        refined final stances instead of repeated rebuttals.
        """
        tasks = []
        # Select prompt template based on whether this is the final round
        prompt_template = STRUCTURED_ROUND_FINAL if is_final_round else STRUCTURED_ROUND_N

        for model in models:
            own_prev = ""
            if model["id"] in prev_positions:
                p = prev_positions[model["id"]]
                own_prev = f"Position: {p.position}\nArgument: {p.argument}"
            else:
                own_prev = "No previous position."

            system_prompt = prompt_template.format(
                query=query,
                round_number=round_num,
                transcript=transcript,
                own_previous=own_prev,
            )
            tasks.append(
                self._call_and_parse_round_n(model, query, system_prompt, round_num, max_tokens=max_tokens)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        positions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Model {models[i]['id']} failed in round {round_num}: {result}"
                )
                positions.append(self._failed_position(
                    model=models[i],
                    round_num=round_num,
                    reason=f"Model did not respond: {result}",
                ))
                continue
            if result is None:
                positions.append(self._failed_position(
                    model=models[i],
                    round_num=round_num,
                    reason="Model returned empty or invalid structured output",
                ))
                continue
            positions.append(result)

        return DebateRound(round_number=round_num, positions=positions)

    def _failed_position(self, model: Dict[str, str], round_num: int, reason: str) -> DebatePosition:
        """Create a standardized failed debate position entry."""
        return DebatePosition(
            model_id=model["id"],
            model_name=model.get("name", model["id"]),
            round_number=round_num,
            position="[MODEL FAILED]",
            argument=reason,
            confidence=0.0,
            latency_ms=0.0,
            status="failed",
        )

    # ── Error Detection ────────────────────────────────────

    _ERROR_MARKERS = ("Error:", "error:", "not configured", "not found", "decommissioned", "Invalid API key")

    def _is_model_error(self, raw: str) -> bool:
        """Detect error strings from the model bridge."""
        return any(marker in raw for marker in self._ERROR_MARKERS)

    # ── Model Call + Parse ───────────────────────────────────

    async def _call_and_parse_round_1(
        self,
        model: Dict[str, str],
        query: str,
        system_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[DebatePosition]:
        """Call model and parse round 1 structured output."""
        try:
            start = time.monotonic()
            raw = await self._call_model(model["id"], query, system_prompt, max_tokens=max_tokens)
            latency = (time.monotonic() - start) * 1000

            if not raw or len(raw.strip()) < 10:
                _diag = (
                    f"Model {model['id']} returned insufficient output in round 1. "
                    f"Raw length={len(raw) if raw else 0}, content='{(raw or '')[:200]}'"
                )
                logger.warning(_diag)
                return self._failed_position(model=model, round_num=1, reason=_diag)

            if self._is_model_error(raw):
                _diag = f"Model {model['id']} returned error in round 1: {raw[:300]}"
                logger.warning(_diag)
                return self._failed_position(model=model, round_num=1, reason=_diag)

            parsed = self._parse_structured_output(raw, model, round_number=1)
            parsed.latency_ms = latency
            return parsed

        except Exception as e:
            _diag = f"Round 1 call failed for {model['id']}: {e}"
            logger.error(_diag)
            return self._failed_position(model=model, round_num=1, reason=_diag)

    async def _call_and_parse_round_n(
        self,
        model: Dict[str, str],
        query: str,
        system_prompt: str,
        round_num: int,
        max_tokens: Optional[int] = None,
    ) -> Optional[DebatePosition]:
        """Call model and parse round N structured output."""
        try:
            start = time.monotonic()
            raw = await self._call_model(model["id"], query, system_prompt, max_tokens=max_tokens)
            latency = (time.monotonic() - start) * 1000

            if not raw or len(raw.strip()) < 10:
                _diag = (
                    f"Model {model['id']} returned insufficient output in round {round_num}. "
                    f"Raw length={len(raw) if raw else 0}, content='{(raw or '')[:200]}'"
                )
                logger.warning(_diag)
                return self._failed_position(model=model, round_num=round_num, reason=_diag)

            if self._is_model_error(raw):
                _diag = f"Model {model['id']} returned error in round {round_num}: {raw[:300]}"
                logger.warning(_diag)
                return self._failed_position(model=model, round_num=round_num, reason=_diag)

            parsed = self._parse_structured_output(
                raw, model, round_number=round_num, is_rebuttal=True
            )
            parsed.latency_ms = latency
            return parsed

        except Exception as e:
            _diag = f"Round {round_num} call failed for {model['id']}: {e}"
            logger.error(_diag)
            return self._failed_position(model=model, round_num=round_num, reason=_diag)

    # ── Structured Output Parsing ────────────────────────────

    def _parse_structured_output(
        self,
        raw: str,
        model: Dict[str, str],
        round_number: int,
        is_rebuttal: bool = False,
    ) -> DebatePosition:
        """Parse model output into DebatePosition."""
        position = self._extract_section(raw, "POSITION")
        # Support both ARGUMENT (new) and REASONING (legacy) headers
        reasoning = self._extract_section(raw, "ARGUMENT") or self._extract_section(raw, "REASONING")
        assumptions = self._extract_list(raw, "ASSUMPTIONS")
        vulnerabilities = self._extract_list(raw, "VULNERABILITIES")
        risks = self._extract_list(raw, "RISKS")
        confidence = self._extract_float(raw, "CONFIDENCE", default=0.5)

        # If confidence is exactly 0.5 (default), try to infer from text cues
        if confidence == 0.5:
            lower_raw = raw.lower()
            conf_match = re.search(r'confidence[:\s]+(?:is\s+)?(\d?\.\d+|[01])', lower_raw)
            if conf_match:
                try:
                    confidence = max(0.0, min(1.0, float(conf_match.group(1))))
                except ValueError:
                    pass

        stance = self._extract_stance(raw)

        rebuttals = []
        weaknesses_found = []
        position_shifted = False
        shift_reason = None

        if is_rebuttal:
            rebuttals = self._extract_list(raw, "REBUTTALS")
            weaknesses_found = self._extract_list(raw, "WEAKNESSES_FOUND")
            shifted_text = self._extract_section(raw, "POSITION_SHIFTED")
            position_shifted = shifted_text.upper().startswith("YES") if shifted_text else False
            shift_reason = self._extract_section(raw, "SHIFT_REASON")

        # Fallback: if structured parsing failed, use raw text
        if not position:
            position = raw[:200].strip()
        if not reasoning:
            reasoning = raw

        return DebatePosition(
            model_id=model["id"],
            model_name=model.get("name", model["id"]),
            round_number=round_number,
            position=position,
            argument=reasoning,
            rebuttals=rebuttals,
            assumptions=assumptions,
            vulnerabilities_found=vulnerabilities,
            risks=risks,
            weaknesses_found=weaknesses_found,
            confidence=confidence,
            stance_vector=stance,
            position_shifted=position_shifted,
            shift_reason=shift_reason,
        )

    def _extract_section(self, text: str, header: str) -> str:
        """Extract text after a header until the next header."""
        pattern = rf'{header}:\s*\[?(.*?)(?:\]?\s*\n(?=[A-Z_]+:)|\]?\s*$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('[]')

        # Fallback: simpler pattern
        pattern2 = rf'{header}:\s*(.*?)(?:\n[A-Z_]+:|\Z)'
        match2 = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
        if match2:
            return match2.group(1).strip()

        return ""

    def _extract_list(self, text: str, header: str) -> List[str]:
        """Extract a bulleted list after a header."""
        section = self._extract_section(text, header)
        if not section:
            return []
        items = re.findall(r'[-•*]\s*(.+)', section)
        return [item.strip() for item in items if item.strip()]

    def _extract_float(
        self, text: str, header: str, default: float = 0.5
    ) -> float:
        """Extract a float value after a header."""
        section = self._extract_section(text, header)
        if not section:
            return default
        match = re.search(r'(0?\.\d+|1\.0|0|1)', section)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        return default

    def _extract_stance(self, text: str) -> StanceVector:
        """Extract stance vector dimensions."""
        def get_dim(name: str) -> float:
            pattern = rf'{name}:\s*(0?\.\d+|1\.0|0|1)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return max(0.0, min(1.0, float(match.group(1))))
                except ValueError:
                    pass
            return 0.5

        return StanceVector(
            certainty=get_dim("certainty"),
            specificity=get_dim("specificity"),
            risk_tolerance=get_dim("risk_tolerance"),
            evidence_reliance=get_dim("evidence_reliance"),
            novelty=get_dim("novelty"),
        )

    # ── Initial Output Conversion ────────────────────────────

    def _convert_initial_outputs(
        self,
        outputs: List[StructuredModelOutput],
        round_number: int,
    ) -> DebateRound:
        """Convert Phase 1 StructuredModelOutput to DebatePositions."""
        positions = []
        for o in outputs:
            if not o.succeeded:
                continue
            positions.append(DebatePosition(
                model_id=o.model_id,
                model_name=o.model_name,
                round_number=round_number,
                position=o.position,
                argument=o.reasoning,
                assumptions=o.assumptions,
                vulnerabilities_found=o.vulnerabilities,
                confidence=o.confidence,
                stance_vector=o.stance_vector,
            ))
        return DebateRound(round_number=round_number, positions=positions)

    # ── Disagreement + Conflicts ─────────────────────────────

    def _compute_disagreement(self, round_data: DebateRound) -> float:
        """Compute disagreement score for a round."""
        # Exclude failed models from disagreement computation
        positions = [p for p in round_data.positions if p.status != "failed"]
        if len(positions) < 2:
            return 0.0

        # Confidence spread
        confidences = [p.confidence for p in positions]
        conf_spread = max(confidences) - min(confidences)

        # Position keyword diversity
        all_keywords = []
        for p in positions:
            words = set(re.findall(r'[a-z]+', p.position.lower()))
            all_keywords.append(words)

        # Average pairwise Jaccard distance
        distances = []
        for i in range(len(all_keywords)):
            for j in range(i + 1, len(all_keywords)):
                union = all_keywords[i] | all_keywords[j]
                inter = all_keywords[i] & all_keywords[j]
                jac = len(inter) / len(union) if union else 0
                distances.append(1.0 - jac)

        avg_dist = sum(distances) / len(distances) if distances else 0.0

        return 0.6 * avg_dist + 0.4 * conf_spread

    def _extract_conflicts(self, round_data: DebateRound) -> List[str]:
        """Extract key conflict descriptions from a round."""
        conflicts = []
        # Exclude failed models from conflict analysis
        positions = [p for p in round_data.positions if p.status != "failed"]

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                a, b = positions[i], positions[j]
                # Simple conflict detection: very different positions
                words_a = set(re.findall(r'[a-z]+', a.position.lower()))
                words_b = set(re.findall(r'[a-z]+', b.position.lower()))
                union = words_a | words_b
                inter = words_a & words_b
                sim = len(inter) / len(union) if union else 1.0

                if sim < 0.3:
                    conflicts.append(
                        f"{a.model_name} vs {b.model_name}: "
                        f"divergent positions (similarity {sim:.0%})"
                    )

        return conflicts[:5]  # Cap at 5

    # ── Shift Detection ──────────────────────────────────────

    def _detect_shifts(
        self,
        prev_positions: Dict[str, DebatePosition],
        current_round: DebateRound,
        round_num: int,
    ) -> List[ShiftRecord]:
        """Detect position shifts from previous round."""
        shifts = []
        for pos in current_round.positions:
            if pos.model_id not in prev_positions:
                continue

            prev = prev_positions[pos.model_id]

            # Compute shift magnitude
            words_prev = set(re.findall(r'[a-z]+', prev.position.lower()))
            words_curr = set(re.findall(r'[a-z]+', pos.position.lower()))
            union = words_prev | words_curr
            inter = words_prev & words_curr
            text_shift = 1.0 - (len(inter) / len(union) if union else 1.0)

            # Stance shift
            v_prev = prev.stance_vector.to_vector()
            v_curr = pos.stance_vector.to_vector()
            import math
            stance_shift = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(v_prev, v_curr))
            ) / math.sqrt(5)

            magnitude = 0.6 * text_shift + 0.4 * stance_shift

            if pos.position_shifted or magnitude > 0.2:
                shifts.append(ShiftRecord(
                    model_id=pos.model_id,
                    model_name=pos.model_name,
                    from_round=round_num - 1,
                    to_round=round_num,
                    old_position_summary=prev.position[:100],
                    new_position_summary=pos.position[:100],
                    shift_magnitude=min(1.0, magnitude),
                    reason=pos.shift_reason or "Position evolved during debate",
                ))

        return shifts

    # ── Consensus ────────────────────────────────────────────

    def _compute_consensus(
        self, rounds: List[DebateRound]
    ) -> Tuple[Optional[str], float]:
        """Compute final consensus from last round."""
        if not rounds:
            return None, 0.0

        final_round = rounds[-1]
        # Exclude failed models from consensus computation
        positions = [p for p in final_round.positions if p.status != "failed"]

        if not positions:
            return None, 0.0

        # Find most common position theme via keyword clustering
        all_keywords = []
        for p in positions:
            words = set(re.findall(r'[a-z]+', p.position.lower()))
            all_keywords.append((p, words))

        # Use the position with highest agreement to others
        best_pos = None
        best_agreement = 0.0

        for i, (pos_i, words_i) in enumerate(all_keywords):
            total_sim = 0.0
            for j, (pos_j, words_j) in enumerate(all_keywords):
                if i == j:
                    continue
                union = words_i | words_j
                inter = words_i & words_j
                total_sim += len(inter) / len(union) if union else 0
            avg_sim = total_sim / (len(all_keywords) - 1) if len(all_keywords) > 1 else 0

            if avg_sim > best_agreement:
                best_agreement = avg_sim
                best_pos = pos_i

        consensus = best_pos.position if best_pos else None
        return consensus, best_agreement

    def _find_unresolved_conflicts(
        self, rounds: List[DebateRound]
    ) -> List[str]:
        """Find conflicts that persisted through all rounds."""
        if len(rounds) < 2:
            return []

        # Conflicts in last round = unresolved
        return rounds[-1].key_conflicts

    # ── Budget Estimation ──────────────────────────────────────

    @staticmethod
    def _estimate_round_tokens(round_data: DebateRound) -> int:
        """Estimate total tokens consumed by a debate round.

        Uses char/4 heuristic for both position and argument text.
        Only counts successful model outputs.
        """
        total = 0
        for pos in round_data.positions:
            if pos.status == "failed":
                continue
            text = (pos.position or "") + (pos.argument or "")
            total += len(text) // 4
        return total

    # ── Transcript Compression ───────────────────────────────

    def _compress_transcript(
        self,
        debate_rounds: List[DebateRound],
        prev_positions: Dict[str, DebatePosition],
    ) -> str:
        """Build a compressed transcript for subsequent rounds.

        Instead of the full prior output, sends only:
          - Summarized consensus direction
          - Top 3 disagreements
          - Each model's position headline (max 100 chars)

        This drastically reduces input tokens for rounds 2+.
        """
        parts: List[str] = []

        for rnd in debate_rounds:
            parts.append(f"=== ROUND {rnd.round_number} SUMMARY ===")

            # Model position headlines
            for pos in rnd.positions:
                if pos.status == "failed":
                    parts.append(f"- {pos.model_name}: [FAILED]")
                    continue
                headline = (pos.position or "")[:100]
                parts.append(
                    f"- {pos.model_name} (conf={pos.confidence:.2f}): {headline}"
                )

            # Top conflicts for this round
            conflicts = rnd.key_conflicts or []
            if conflicts:
                parts.append("Key disagreements:")
                for c in conflicts[:3]:
                    parts.append(f"  * {c[:120]}")

            # Convergence signal
            if rnd.convergence_delta != 0:
                direction = "converging" if rnd.convergence_delta > 0 else "diverging"
                parts.append(
                    f"Trend: {direction} (delta={rnd.convergence_delta:.3f})"
                )

        return "\n".join(parts)

    # ── Debate Summary for Rounds 2+ (≤ MAX_SUMMARY_TOKENS) ──

    def _compress_debate_summary(
        self,
        debate_rounds: List[DebateRound],
        prev_positions: Dict[str, DebatePosition],
    ) -> str:
        """Build a compressed debate summary for injection into rounds 2+.

        Instead of passing full prior model outputs, generates:
          - Majority stance  (1-2 sentences)
          - Minority stance  (1-2 sentences)
          - Key disagreement (1 sentence)

        Hard-capped at MAX_SUMMARY_TOKENS (~300 tokens ≈ 1200 chars).
        """
        # Collect successful positions from the latest round
        latest = debate_rounds[-1] if debate_rounds else None
        if not latest:
            return "No prior debate data."

        successful = [
            p for p in latest.positions if p.status != "failed"
        ]
        if not successful:
            return "All models failed in previous round."

        # ── Determine majority vs minority by position clustering ──
        # Use simple keyword overlap to partition into majority/minority
        word_sets = []
        for p in successful:
            words = set(
                w.lower() for w in re.split(r"\W+", (p.position or ""))
                if len(w) > 3
            )
            word_sets.append((p, words))

        # Compute pairwise similarity and find majority cluster
        n = len(word_sets)
        avg_sims = []
        for i in range(n):
            total = 0.0
            for j in range(n):
                if i == j:
                    continue
                union = word_sets[i][1] | word_sets[j][1]
                inter = word_sets[i][1] & word_sets[j][1]
                total += len(inter) / len(union) if union else 0
            avg_sims.append(total / max(1, n - 1))

        # Models with above-median similarity → majority
        median_sim = sorted(avg_sims)[len(avg_sims) // 2] if avg_sims else 0
        majority_positions = []
        minority_positions = []
        for i, (pos, _) in enumerate(word_sets):
            if avg_sims[i] >= median_sim:
                majority_positions.append(pos)
            else:
                minority_positions.append(pos)

        # If no minority (all agree), put a note
        if not minority_positions:
            minority_positions = []

        # ── Build summary ──
        parts: List[str] = ["DEBATE SUMMARY (compressed):"]

        # Majority stance (pick highest-confidence majority member)
        if majority_positions:
            top_maj = max(majority_positions, key=lambda p: p.confidence)
            maj_text = (top_maj.position or "")[:200]
            parts.append(
                f"MAJORITY ({len(majority_positions)} models, "
                f"lead conf={top_maj.confidence:.2f}): {maj_text}"
            )

        # Minority stance
        if minority_positions:
            top_min = max(minority_positions, key=lambda p: p.confidence)
            min_text = (top_min.position or "")[:200]
            parts.append(
                f"MINORITY ({len(minority_positions)} models, "
                f"lead conf={top_min.confidence:.2f}): {min_text}"
            )
        else:
            parts.append("MINORITY: None — all models broadly agree.")

        # Key disagreement
        conflicts = latest.key_conflicts or []
        if conflicts:
            parts.append(f"KEY DISAGREEMENT: {conflicts[0][:150]}")
        else:
            parts.append("KEY DISAGREEMENT: No major conflicts identified.")

        summary = "\n".join(parts)

        # Hard-cap character length (MAX_SUMMARY_TOKENS * 4 chars/token)
        max_chars = MAX_SUMMARY_TOKENS * 4
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "…"

        return summary

    # ── Early Stop: Consensus Score ──────────────────────────

    def _compute_consensus_score(self, round_data: DebateRound) -> float:
        """Compute consensus score for a single round (0.0–1.0).

        Uses pairwise Jaccard similarity across model positions.
        Returns 0.0 if fewer than 2 successful models.
        """
        successful = [
            p for p in round_data.positions
            if p.status != "failed" and p.position
        ]
        if len(successful) < 2:
            return 0.0

        word_sets = []
        for p in successful:
            words = set(
                w.lower() for w in re.split(r"\W+", (p.position or ""))
                if len(w) > 3
            )
            word_sets.append(words)

        n = len(word_sets)
        total_sim = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                union = word_sets[i] | word_sets[j]
                inter = word_sets[i] & word_sets[j]
                total_sim += len(inter) / len(union) if union else 0
                pairs += 1

        return total_sim / pairs if pairs > 0 else 0.0

    # ── Early Stop: Stability Score ──────────────────────────

    def _compute_stability_score(self, debate_rounds: List[DebateRound]) -> float:
        """Compute stability score across rounds (0.0–1.0).

        Measures how little positions changed between consecutive rounds.
        High stability = positions are not shifting (converged).
        Returns 0.0 if fewer than 2 rounds.
        """
        if len(debate_rounds) < 2:
            return 0.0

        prev_round = debate_rounds[-2]
        curr_round = debate_rounds[-1]

        prev_map: Dict[str, str] = {}
        for p in prev_round.positions:
            if p.status != "failed":
                prev_map[p.model_id] = (p.position or "").lower()

        stable_count = 0
        total_count = 0

        for p in curr_round.positions:
            if p.status == "failed":
                continue
            total_count += 1
            prev_pos = prev_map.get(p.model_id, "")
            curr_pos = (p.position or "").lower()

            if not prev_pos:
                continue

            # Jaccard similarity of word sets
            prev_words = set(w for w in re.split(r"\W+", prev_pos) if len(w) > 3)
            curr_words = set(w for w in re.split(r"\W+", curr_pos) if len(w) > 3)
            union = prev_words | curr_words
            inter = prev_words & curr_words
            sim = len(inter) / len(union) if union else 1.0

            if sim >= 0.5:  # Position substantially unchanged
                stable_count += 1

        return stable_count / total_count if total_count > 0 else 0.0

    # ── Transcript Formatting ────────────────────────────────

    def _format_round_transcript(self, round_data: DebateRound) -> str:
        """Format a round into a text transcript for subsequent rounds."""
        parts = [f"=== ROUND {round_data.round_number} ==="]
        for pos in round_data.positions:
            if pos.status == "failed":
                parts.append(f"\n--- {pos.model_name} [FAILED] ---")
                continue
            parts.append(f"\n--- {pos.model_name} ---")
            parts.append(f"Position: {pos.position}")
            parts.append(f"Argument: {pos.argument[:500]}")
            if pos.assumptions:
                parts.append(f"Assumptions: {', '.join(pos.assumptions[:3])}")
            if pos.vulnerabilities_found:
                parts.append(
                    f"Vulnerabilities: {', '.join(pos.vulnerabilities_found[:3])}"
                )
            parts.append(f"Confidence: {pos.confidence}")
            if pos.rebuttals:
                parts.append(f"Rebuttals: {'; '.join(pos.rebuttals[:3])}")
        return "\n".join(parts)

    # ── Drift / Rift / Fragility Computation ─────────────────

    def _compute_debate_drift_rift(self, rounds: List[DebateRound]) -> Dict[str, Any]:
        """
        Compute drift, rift, confidence spread, and fragility from debate rounds.

        Drift: cosine distance of each model's position between consecutive rounds (TF-IDF).
        Rift: mean pairwise cosine distance between all models within each round.
        Fragility: weighted composite of drift + rift + confidence spread.
        """
        if not rounds:
            return {
                "drift_index": 0.0, "rift_index": 0.0, "confidence_spread": 0.0,
                "fragility_score": 0.0, "per_model_drift": {}, "per_round_rift": [],
                "per_round_disagreement": [], "overall_confidence": 0.5,
            }

        # Collect per-model positions across rounds — exclude failed models
        model_ids = list({
            pos.model_id for rnd in rounds for pos in rnd.positions
            if pos.status != "failed"
        })
        model_positions: Dict[str, List[str]] = {mid: [] for mid in model_ids}
        for rnd in rounds:
            round_model_ids = set()
            for pos in rnd.positions:
                if pos.status == "failed":
                    continue
                model_positions[pos.model_id].append(pos.position or "")
                round_model_ids.add(pos.model_id)
            # Fill gaps for models that didn't respond in this round
            for mid in model_ids:
                if mid not in round_model_ids:
                    model_positions[mid].append("")

        # --- Drift: cosine distance between rounds per model ---
        per_model_drift: Dict[str, List[float]] = {}
        all_drift_values = []
        for mid, positions in model_positions.items():
            drifts = []
            for i in range(len(positions) - 1):
                if positions[i] and positions[i + 1]:
                    try:
                        vec = TfidfVectorizer()
                        tfidf = vec.fit_transform([positions[i], positions[i + 1]])
                        sim = sk_cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                        drift_val = 1.0 - float(sim)
                    except Exception:
                        drift_val = 0.5
                else:
                    drift_val = 0.5
                drifts.append(round(drift_val, 4))
                all_drift_values.append(drift_val)
            per_model_drift[mid] = drifts

        drift_index = float(np.mean(all_drift_values)) if all_drift_values else 0.0

        # --- Rift: mean pairwise distance within each round ---
        per_round_rift = []
        for rnd in rounds:
            positions_text = [
                pos.position for pos in rnd.positions
                if pos.position and pos.status != "failed"
            ]
            if len(positions_text) < 2:
                per_round_rift.append(0.0)
                continue
            try:
                vec = TfidfVectorizer()
                tfidf = vec.fit_transform(positions_text)
                sim_matrix = sk_cosine_similarity(tfidf)
                n = sim_matrix.shape[0]
                distances = []
                for i in range(n):
                    for j in range(i + 1, n):
                        distances.append(1.0 - sim_matrix[i][j])
                rift_val = float(np.mean(distances)) if distances else 0.0
            except Exception:
                rift_val = 0.5
            per_round_rift.append(round(rift_val, 4))

        rift_index = float(np.mean(per_round_rift)) if per_round_rift else 0.0

        # --- Confidence spread (exclude failed models) ---
        all_confs = [
            pos.confidence for rnd in rounds for pos in rnd.positions
            if pos.status != "failed"
        ]
        confidence_spread = float(np.std(all_confs)) if len(all_confs) > 1 else 0.0
        overall_confidence = float(np.mean(all_confs)) if all_confs else 0.5

        # --- Per-round disagreement ---
        per_round_disagreement = [
            round(rnd.round_disagreement, 4) for rnd in rounds
        ]

        # --- Fragility ---
        fragility_score = drift_index * 0.35 + rift_index * 0.35 + confidence_spread * 0.30

        return {
            "drift_index": round(drift_index, 4),
            "rift_index": round(rift_index, 4),
            "confidence_spread": round(confidence_spread, 4),
            "fragility_score": round(fragility_score, 4),
            "per_model_drift": per_model_drift,
            "per_round_rift": per_round_rift,
            "per_round_disagreement": per_round_disagreement,
            "overall_confidence": round(overall_confidence, 4),
        }

    def _build_analysis_summary(
        self, rounds: List[DebateRound], shifts: List[ShiftRecord]
    ) -> Dict[str, Any]:
        """
        Build a structured analysis summary from debate rounds.
        Populates conflict_axes, strongest/weakest, convergence, synthesis.
        """
        if not rounds:
            return {}

        # Conflict axes: collect all key_conflicts across rounds
        all_conflicts = []
        for rnd in rounds:
            all_conflicts.extend(rnd.key_conflicts or [])
        conflict_axes = list(set(all_conflicts))[:10]

        # Disagreement strength: mean disagreement across rounds
        disagreements = [rnd.round_disagreement for rnd in rounds]
        disagreement_strength = float(np.mean(disagreements)) if disagreements else 0.0

        # Logical stability: inverse of convergence_delta variance
        deltas = [abs(rnd.convergence_delta) for rnd in rounds if rnd.convergence_delta != 0]
        logical_stability = 1.0 - min(1.0, float(np.std(deltas))) if deltas else 0.5

        # Convergence level
        final_disagreement = rounds[-1].round_disagreement if rounds else 0.0
        if final_disagreement < 0.2:
            convergence_level = "strong"
        elif final_disagreement < 0.4:
            convergence_level = "moderate"
        elif final_disagreement < 0.6:
            convergence_level = "partial"
        else:
            convergence_level = "none"

        convergence_detail = (
            f"Final round disagreement: {final_disagreement:.2f}. "
            f"{len(shifts)} position shifts detected across {len(rounds)} rounds."
        )

        # Strongest/weakest by confidence
        all_positions = [pos for rnd in rounds for pos in rnd.positions]
        strongest = ""
        weakest = ""
        if all_positions:
            sorted_by_conf = sorted(all_positions, key=lambda p: p.confidence, reverse=True)
            top = sorted_by_conf[0]
            bottom = sorted_by_conf[-1]
            strongest = f"{top.model_name} (confidence: {top.confidence:.2f})"
            weakest = f"{bottom.model_name} (confidence: {bottom.confidence:.2f})"

        # Synthesis from final round
        synthesis_parts = []
        if rounds:
            final = rounds[-1]
            for pos in final.positions:
                synthesis_parts.append(f"{pos.model_name}: {pos.position[:150]}")
        synthesis = " | ".join(synthesis_parts)

        return {
            "conflict_axes": conflict_axes,
            "disagreement_strength": round(disagreement_strength, 4),
            "logical_stability": round(logical_stability, 4),
            "convergence_level": convergence_level,
            "convergence_detail": convergence_detail,
            "strongest_argument": strongest,
            "weakest_argument": weakest,
            "synthesis": synthesis,
        }
