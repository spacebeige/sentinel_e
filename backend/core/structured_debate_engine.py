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
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from core.ensemble_schemas import (
    MIN_DEBATE_ROUNDS,
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

STRUCTURED_ROUND_1 = """You are a rigorous analytical model participating in a structured multi-model debate.

QUERY: {query}

You must respond with EXACTLY this structure (use these exact headers):

POSITION: [Your clear thesis statement in 1-2 sentences]

REASONING: [Step-by-step reasoning chain supporting your position. Be specific and evidence-based.]

ASSUMPTIONS: [List each assumption on its own line, prefixed with "- "]
- assumption 1
- assumption 2

VULNERABILITIES: [Self-identify weaknesses in your position, prefixed with "- "]
- vulnerability 1
- vulnerability 2

CONFIDENCE: [A number between 0.0 and 1.0 representing your confidence]

STANCE: [Rate each dimension 0.0-1.0]
certainty: [0.0-1.0]
specificity: [0.0-1.0]
risk_tolerance: [0.0-1.0]
evidence_reliance: [0.0-1.0]
novelty: [0.0-1.0]

Be honest about uncertainties. Do not inflate confidence."""

STRUCTURED_ROUND_N = """You are participating in round {round_number} of a structured multi-model debate.

QUERY: {query}

PREVIOUS DEBATE TRANSCRIPT:
{transcript}

YOUR PREVIOUS POSITION: {own_previous}

Review other models' arguments carefully. You MUST:
1. Address specific arguments from opponents
2. Identify weaknesses in their reasoning
3. Update your position if warranted (with explanation)
4. Maintain or revise your confidence based on the debate

Respond with EXACTLY this structure:

REBUTTALS: [Address specific opponent arguments, prefixed with "- "]
- [Model X] claimed Y, but this fails because Z
- [Model W] assumed V, which is incorrect because...

POSITION: [Your current thesis — may be updated based on debate]

REASONING: [Updated reasoning incorporating debate insights]

ASSUMPTIONS: [Current assumptions, prefixed with "- "]
- assumption 1

VULNERABILITIES: [Current self-identified weaknesses, prefixed with "- "]
- vulnerability 1

POSITION_SHIFTED: [YES or NO — did your position change?]

SHIFT_REASON: [If shifted, explain why. If not, explain why your position held.]

CONFIDENCE: [Updated confidence 0.0-1.0]

STANCE: [Updated dimensional ratings]
certainty: [0.0-1.0]
specificity: [0.0-1.0]
risk_tolerance: [0.0-1.0]
evidence_reliance: [0.0-1.0]
novelty: [0.0-1.0]"""


# Type for model caller function
ModelCaller = Callable[[str, str, str], Coroutine[Any, Any, str]]


class StructuredDebateEngine:
    """
    Executes structured multi-model debate with minimum 3 rounds.

    Args:
        call_model: async function(model_id, prompt, system_role) -> str
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
        Execute structured debate with mandatory minimum 3 rounds.

        Args:
            query: The user's question
            rounds: Number of rounds (minimum 3, enforced)
            initial_outputs: Pre-computed structured outputs from Phase 1

        Returns:
            DebateResult with all rounds, shifts, and consensus.

        Raises:
            EnsembleFailure if <3 models or <3 rounds complete.
        """
        rounds = max(rounds, MIN_DEBATE_ROUNDS)
        models = self._get_enabled_models()

        if len(models) < 3:
            raise EnsembleFailure(
                code=EnsembleFailureCode.INSUFFICIENT_MODELS,
                message=f"Debate requires minimum 3 models, got {len(models)}",
                models_available=len(models),
            )

        debate_rounds: List[DebateRound] = []
        all_shifts: List[ShiftRecord] = []
        transcript_parts: List[str] = []
        prev_positions: Dict[str, DebatePosition] = {}

        for round_num in range(1, rounds + 1):
            logger.info(f"Debate round {round_num}/{rounds} with {len(models)} models")

            if round_num == 1 and initial_outputs:
                # Use pre-computed Phase 1 outputs
                round_result = self._convert_initial_outputs(
                    initial_outputs, round_num
                )
            elif round_num == 1:
                # Fresh round 1
                round_result = await self._run_round_1(query, models)
            else:
                # Subsequent rounds with transcript
                transcript = "\n\n".join(transcript_parts)
                round_result = await self._run_round_n(
                    query, models, round_num, transcript, prev_positions
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

            # Update transcript
            transcript_parts.append(
                self._format_round_transcript(round_result)
            )

            # Update previous positions
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

        return DebateResult(
            rounds=debate_rounds,
            total_rounds=len(debate_rounds),
            shift_table=all_shifts,
            final_consensus=final_consensus,
            consensus_strength=consensus_strength,
            unresolved_conflicts=unresolved,
        )

    # ── Round Execution ──────────────────────────────────────

    async def _run_round_1(
        self, query: str, models: List[Dict[str, str]]
    ) -> DebateRound:
        """Execute round 1: independent positions."""
        system_prompt = STRUCTURED_ROUND_1.format(query=query)

        tasks = [
            self._call_and_parse_round_1(model, query, system_prompt)
            for model in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        positions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model {models[i]['id']} failed in round 1: {result}")
                continue
            if result is not None:
                positions.append(result)

        return DebateRound(round_number=1, positions=positions)

    async def _run_round_n(
        self,
        query: str,
        models: List[Dict[str, str]],
        round_num: int,
        transcript: str,
        prev_positions: Dict[str, DebatePosition],
    ) -> DebateRound:
        """Execute round N: rebuttals and position updates."""
        tasks = []
        for model in models:
            own_prev = ""
            if model["id"] in prev_positions:
                p = prev_positions[model["id"]]
                own_prev = f"Position: {p.position}\nArgument: {p.argument}"
            else:
                own_prev = "No previous position."

            system_prompt = STRUCTURED_ROUND_N.format(
                query=query,
                round_number=round_num,
                transcript=transcript,
                own_previous=own_prev,
            )
            tasks.append(
                self._call_and_parse_round_n(model, query, system_prompt, round_num)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        positions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Model {models[i]['id']} failed in round {round_num}: {result}"
                )
                continue
            if result is not None:
                positions.append(result)

        return DebateRound(round_number=round_num, positions=positions)

    # ── Model Call + Parse ───────────────────────────────────

    async def _call_and_parse_round_1(
        self,
        model: Dict[str, str],
        query: str,
        system_prompt: str,
    ) -> Optional[DebatePosition]:
        """Call model and parse round 1 structured output."""
        try:
            start = time.monotonic()
            raw = await self._call_model(model["id"], query, system_prompt)
            latency = (time.monotonic() - start) * 1000

            if not raw or len(raw.strip()) < 10:
                logger.warning(f"Model {model['id']} returned empty output in round 1")
                return None

            parsed = self._parse_structured_output(raw, model, round_number=1)
            return parsed

        except Exception as e:
            logger.error(f"Round 1 call failed for {model['id']}: {e}")
            return None

    async def _call_and_parse_round_n(
        self,
        model: Dict[str, str],
        query: str,
        system_prompt: str,
        round_num: int,
    ) -> Optional[DebatePosition]:
        """Call model and parse round N structured output."""
        try:
            start = time.monotonic()
            raw = await self._call_model(model["id"], query, system_prompt)
            latency = (time.monotonic() - start) * 1000

            if not raw or len(raw.strip()) < 10:
                return None

            parsed = self._parse_structured_output(
                raw, model, round_number=round_num, is_rebuttal=True
            )
            return parsed

        except Exception as e:
            logger.error(f"Round {round_num} call failed for {model['id']}: {e}")
            return None

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
        reasoning = self._extract_section(raw, "REASONING")
        assumptions = self._extract_list(raw, "ASSUMPTIONS")
        vulnerabilities = self._extract_list(raw, "VULNERABILITIES")
        confidence = self._extract_float(raw, "CONFIDENCE", default=0.5)
        stance = self._extract_stance(raw)

        rebuttals = []
        position_shifted = False
        shift_reason = None

        if is_rebuttal:
            rebuttals = self._extract_list(raw, "REBUTTALS")
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
        positions = round_data.positions
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
        positions = round_data.positions

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
        positions = final_round.positions

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

    # ── Transcript Formatting ────────────────────────────────

    def _format_round_transcript(self, round_data: DebateRound) -> str:
        """Format a round into a text transcript for subsequent rounds."""
        parts = [f"=== ROUND {round_data.round_number} ==="]
        for pos in round_data.positions:
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
