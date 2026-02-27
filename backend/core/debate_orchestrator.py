"""
Debate Orchestrator — Sentinel-E Omega v4.5

True multi-round adversarial debate engine.
Each model runs independently per round with full transcript injection.

Architecture:
- Round 1: All models receive user query simultaneously (parallel via asyncio.gather)
- Round 2+: Each model receives its own previous answer + all opponents' answers + full transcript
- No merging. No synthesis until final analysis.
- Each model's output is preserved independently per round.

Models:
- Model A (Groq/LLaMA 3.1): Fast analytical
- Model B (Llama 3.3 70B): Primary reasoning
- Model C (Qwen 2.5 7B): Careful methodical

Output: Per-round, per-model structured data + final debate analysis.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY

logger = logging.getLogger("Debate-Orchestrator")


# ============================================================
# MODEL DEFINITIONS — Derived from COGNITIVE_MODEL_REGISTRY
# ============================================================

# Color palette for debate model cards
_DEBATE_COLORS = ["blue", "indigo", "purple", "emerald", "amber", "cyan", "rose"]

# Legacy ID mapping: maps legacy caller IDs used by DebateOrchestrator
# to canonical registry keys
_LEGACY_ID_MAP = {
    "groq-small": "groq",
    "llama-3.3": "llama70b",
    "qwen-vl-2.5": "qwen",
    "qwen3-coder": "qwen3-coder",
    "qwen3-vl": "qwen3-vl",
    "nemotron-nano": "nemotron",
    "kimi-2.5": "kimi",
}


def _build_debate_models():
    """Build debate model list from the authoritative COGNITIVE_MODEL_REGISTRY."""
    models = []
    for i, (key, spec) in enumerate(COGNITIVE_MODEL_REGISTRY.items()):
        if not spec.enabled:
            continue
        legacy_id = _LEGACY_ID_MAP.get(key, key)
        models.append({
            "id": legacy_id,
            "registry_key": key,
            "label": spec.name,
            "provider": legacy_id,  # Used as key into _model_callers
            "name": spec.name,
            "color": _DEBATE_COLORS[i % len(_DEBATE_COLORS)],
        })
    return models


# Built at import time; refreshed when models change
DEBATE_MODELS = _build_debate_models()

# Valid debate roles
DEBATE_ROLES = {"for", "against", "judge", "neutral"}


# ============================================================
# SYSTEM PROMPTS
# ============================================================

ROUND_1_SYSTEM = """You are {model_name}, a rigorous analytical debater in a multi-model adversarial reasoning system.
{role_instruction}
ROUND 1 — OPENING STATEMENT

You are presenting your INDEPENDENT analysis. No other model has spoken yet.
The other debaters are: {other_models}.

RULES:
1. State your POSITION clearly and directly.
2. Present your ARGUMENT with evidence and reasoning.
3. List your KEY ASSUMPTIONS explicitly.
4. Identify RISKS and potential failure modes in your own reasoning.
5. Be specific, not vague. Commit to a stance.
6. Do NOT hedge excessively. Take a position.

OUTPUT FORMAT (follow exactly):
POSITION: [Your clear position in 1-2 sentences]
ARGUMENT: [Your detailed argument, 3-8 sentences]
ASSUMPTIONS: [Bullet list of key assumptions]
RISKS: [Bullet list of risks to your position]
CONFIDENCE: [0.0-1.0 how confident you are]"""

ROUND_N_SYSTEM = """You are {model_name}, a rigorous analytical debater in Round {round_num} of a multi-model adversarial debate.
{role_instruction}
PREVIOUS ROUND TRANSCRIPT:
{transcript}

YOUR PREVIOUS POSITION:
{own_previous}

RULES FOR THIS ROUND:
1. You MUST directly address and rebut specific points from other models.
2. Reference opponents BY NAME (e.g., "Groq claims X, but this fails because...", "Llama70B argues Y, which overlooks...")
3. Identify WEAKNESSES in opponents' arguments.
4. STRENGTHEN your own position or SHIFT if opponents made compelling points.
5. If you shift position, explain WHY explicitly.
6. Do NOT repeat your previous argument verbatim.
7. Be adversarial. Challenge assumptions. Find logical gaps.

OUTPUT FORMAT (follow exactly):
REBUTTALS: [Direct responses to specific opponent claims, referencing them by name]
POSITION: [Your updated position — may evolve from previous round]
ARGUMENT: [Your strengthened/modified argument]
POSITION_SHIFT: [none/minor/major — explain if shifted]
WEAKNESSES_FOUND: [Specific weaknesses in opponents' reasoning]
CONFIDENCE: [0.0-1.0 updated confidence]"""

ANALYSIS_SYSTEM = """You are an impartial debate analyst. Analyze this multi-round adversarial debate transcript.
The debaters are: {debater_names}.
Always refer to them by name.

FULL DEBATE TRANSCRIPT:
{transcript}

Produce a structured analysis. Be specific and reference actual arguments made by name.

OUTPUT FORMAT (follow exactly):
CONFLICT_AXES: [List the main points of disagreement between models, naming which models disagree]
DISAGREEMENT_STRENGTH: [0.0-1.0 how strongly models disagree]
LOGICAL_STABILITY: [0.0-1.0 how logically sound the overall debate was]
POSITION_SHIFTS: [List any position changes that occurred, by which named model, and why]
CONVERGENCE_LEVEL: [none/low/moderate/high — did models converge?]
CONVERGENCE_DETAIL: [Explain convergence or lack thereof]
STRONGEST_ARGUMENT: [Which named model had the strongest overall argument and why]
WEAKEST_ARGUMENT: [Which named model had the weakest overall argument and why]
CONFIDENCE_RECALIBRATION: [0.0-1.0 overall debate confidence after analysis]
SYNTHESIS: [Brief balanced conclusion incorporating the strongest elements from all positions]"""


# Role-specific instructions for debate
ROLE_INSTRUCTIONS = {
    "for": "You are assigned the role of ADVOCATE (FOR). You MUST argue IN FAVOR of the proposition. Even if you personally disagree, you must construct the strongest possible case FOR the claim.",
    "against": "You are assigned the role of OPPOSITION (AGAINST). You MUST argue AGAINST the proposition. Even if the claim seems reasonable, you must construct the strongest possible case AGAINST it.",
    "judge": "You are assigned the role of JUDGE. You must NOT take a position. Instead, evaluate the other debaters' arguments for logical strength, evidence quality, and coherence. Score each argument.",
    "neutral": "",  # No special instruction
}


JUDGE_SCORING_SYSTEM = """You are the JUDGE in this multi-model debate. You must NOT take a position on the topic itself.

DEBATE TRANSCRIPT:
{transcript}

DEBATERS: {debater_names}

Your task is to evaluate each debater's performance. Score each on:
1. Logical Strength (0.0-1.0): How logically sound are their arguments?
2. Evidence Quality (0.0-1.0): How well-supported are their claims?
3. Coherence (0.0-1.0): How internally consistent is their reasoning?
4. Rebuttal Effectiveness (0.0-1.0): How well do they counter opponents?

OUTPUT FORMAT (follow exactly):
STRONGEST_ARGUMENT: [Which model and why]
WEAKEST_ARGUMENT: [Which model and why]
SCORING_MATRIX:
{scoring_entries}
STABILITY_ASSESSMENT: [How stable is the overall debate? Are positions well-founded or fragile?]
WINNER: [Which model presents the most compelling overall case]"""


@dataclass
class ModelRoundOutput:
    """Output from a single model in a single round."""
    model_id: str
    model_label: str
    model_name: str
    model_color: str
    round_num: int
    position: str = ""
    argument: str = ""
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    rebuttals: str = ""
    position_shift: str = "none"
    weaknesses_found: str = ""
    confidence: float = 0.5
    raw_output: str = ""


@dataclass
class DebateAnalysis:
    """Final debate analysis after all rounds."""
    conflict_axes: List[str] = field(default_factory=list)
    disagreement_strength: float = 0.0
    logical_stability: float = 0.5
    position_shifts: List[str] = field(default_factory=list)
    convergence_level: str = "none"
    convergence_detail: str = ""
    strongest_argument: str = ""
    weakest_argument: str = ""
    confidence_recalibration: float = 0.5
    synthesis: str = ""
    # 3.X additions
    disagreement_trend: List[float] = field(default_factory=list)  # Per-round disagreement scores
    judge_scoring: Optional[Dict[str, Any]] = None  # Judge model scoring matrix
    semantic_divergence: float = 0.0  # Mean pairwise embedding distance


@dataclass 
class DebateResult:
    """Complete debate result across all rounds."""
    query: str
    rounds: List[List[ModelRoundOutput]] = field(default_factory=list)
    analysis: Optional[DebateAnalysis] = None
    total_rounds: int = 0
    models_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        rounds_data = []
        for round_outputs in self.rounds:
            round_data = []
            for mo in round_outputs:
                round_data.append({
                    "model_id": mo.model_id,
                    "model_label": mo.model_label,
                    "model_name": mo.model_name,
                    "model_color": mo.model_color,
                    "round_num": mo.round_num,
                    "position": mo.position,
                    "argument": mo.argument,
                    "assumptions": mo.assumptions,
                    "risks": mo.risks,
                    "rebuttals": mo.rebuttals,
                    "position_shift": mo.position_shift,
                    "weaknesses_found": mo.weaknesses_found,
                    "confidence": mo.confidence,
                })
            rounds_data.append(round_data)
        
        analysis_data = None
        if self.analysis:
            analysis_data = {
                "conflict_axes": self.analysis.conflict_axes,
                "disagreement_strength": self.analysis.disagreement_strength,
                "logical_stability": self.analysis.logical_stability,
                "position_shifts": self.analysis.position_shifts,
                "convergence_level": self.analysis.convergence_level,
                "convergence_detail": self.analysis.convergence_detail,
                "strongest_argument": self.analysis.strongest_argument,
                "weakest_argument": self.analysis.weakest_argument,
                "confidence_recalibration": self.analysis.confidence_recalibration,
                "synthesis": self.analysis.synthesis,
                "disagreement_trend": self.analysis.disagreement_trend,
                "judge_scoring": self.analysis.judge_scoring,
                "semantic_divergence": self.analysis.semantic_divergence,
            }

        return {
            "query": self.query,
            "rounds": rounds_data,
            "analysis": analysis_data,
            "total_rounds": self.total_rounds,
            "models_used": self.models_used,
        }


class DebateOrchestrator:
    """
    True multi-round adversarial debate engine (v4.5).
    
    Routes ALL model calls through MCOModelBridge.call_model(), eliminating
    hardcoded legacy callers. Every enabled model in COGNITIVE_MODEL_REGISTRY
    participates in debate automatically.
    
    Supports:
    - Role assignment (for, against, judge, neutral)
    - Iterative N-round debate with cross-model awareness
    - Semantic disagreement quantification
    - Judge scoring matrix
    - Disagreement trend tracking across rounds
    - Dynamic analysis model selection with retry
    """

    # Preferred models for analysis (in priority order)
    _ANALYSIS_MODEL_PRIORITY = ["llama70b", "qwen3-coder", "groq", "nemotron", "kimi"]

    def __init__(self, cloud_client):
        self.client = cloud_client
        # ALL models route through call_model() — no hardcoded callers
        self._model_callers = {}
        for model in DEBATE_MODELS:
            mid = model["id"]
            self._model_callers[mid] = (
                lambda prompt, system_role="You are a rigorous analytical debater.", _mid=mid:
                self.client.call_model(_mid, prompt, system_role)
            )
        logger.info(
            f"DebateOrchestrator initialized with {len(self._model_callers)} models: "
            f"{list(self._model_callers.keys())}"
        )

    async def run_debate(
        self, query: str, rounds: int = 3,
        role_map: Dict[str, str] = None,
    ) -> DebateResult:
        """
        Execute a full multi-round adversarial debate.
        
        Args:
            query: User's input question/claim
            rounds: Number of debate rounds (default 3)
            role_map: Optional role assignments {model_id: role}
                      e.g. {"groq": "for", "llama70b": "against", "qwen": "judge"}
        """
        rounds = max(1, min(rounds, 10))
        role_map = role_map or {}
        
        # Validate roles
        for model_id, role in role_map.items():
            if role not in DEBATE_ROLES:
                logger.warning(f"Invalid role '{role}' for {model_id}, defaulting to neutral")
                role_map[model_id] = "neutral"
        
        result = DebateResult(
            query=query,
            total_rounds=rounds,
            models_used=[m["name"] for m in DEBATE_MODELS],
        )
        
        transcript_parts = []
        disagreement_trend = []

        for round_num in range(1, rounds + 1):
            logger.info(f"Debate Round {round_num}/{rounds} — {len(DEBATE_MODELS)} models participating")
            
            if round_num == 1:
                round_outputs = await self._execute_round_1(query, role_map)
            else:
                transcript = "\n\n".join(transcript_parts)
                round_outputs = await self._execute_round_n(
                    query, round_num, transcript, result.rounds, role_map
                )
            
            result.rounds.append(round_outputs)
            
            # Track per-round disagreement
            round_disagreement = self._compute_round_disagreement(round_outputs)
            disagreement_trend.append(round_disagreement)

            # Structured round telemetry
            succeeded = sum(1 for o in round_outputs if o.confidence > 0)
            failed = len(round_outputs) - succeeded
            logger.info(
                f"Round {round_num} complete: succeeded={succeeded}, failed={failed}, "
                f"disagreement={round_disagreement:.4f}, "
                f"confidences={[round(o.confidence, 3) for o in round_outputs]}"
            )
            
            # Add this round to transcript
            round_transcript = self._format_round_transcript(round_num, round_outputs)
            transcript_parts.append(round_transcript)
        
        # Final analysis
        full_transcript = "\n\n".join(transcript_parts)
        result.analysis = await self._run_analysis(full_transcript)
        result.analysis.disagreement_trend = disagreement_trend
        
        # Run judge scoring if a judge role is assigned
        judge_model = None
        for model_id, role in role_map.items():
            if role == "judge":
                judge_model = model_id
                break
        
        if judge_model:
            judge_scoring = await self._run_judge_scoring(full_transcript, judge_model)
            result.analysis.judge_scoring = judge_scoring
        
        return result

    async def _execute_round_1(self, query: str, role_map: Dict[str, str] = None) -> List[ModelRoundOutput]:
        """Round 1: All models respond independently to user query with role awareness."""
        role_map = role_map or {}
        tasks = []
        
        for model in DEBATE_MODELS:
            role = role_map.get(model["id"], "neutral")
            role_instruction = ROLE_INSTRUCTIONS.get(role, "")
            other_models = ", ".join(
                m["label"] for m in DEBATE_MODELS if m["id"] != model["id"]
            )
            
            system_prompt = ROUND_1_SYSTEM.format(
                model_name=model["label"],
                role_instruction=role_instruction,
                other_models=other_models,
            )
            user_prompt = f"DEBATE TOPIC:\n{query}\n\nPresent your opening statement."
            
            caller = self._model_callers[model["provider"]]
            tasks.append(self._call_and_parse_round1(caller, user_prompt, system_prompt, model))
        
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, output in enumerate(outputs):
            if isinstance(output, Exception):
                logger.error(f"{DEBATE_MODELS[i]['label']} Round 1 failed: {output}")
                results.append(ModelRoundOutput(
                    model_id=DEBATE_MODELS[i]["id"],
                    model_label=DEBATE_MODELS[i]["label"],
                    model_name=DEBATE_MODELS[i]["name"],
                    model_color=DEBATE_MODELS[i]["color"],
                    round_num=1,
                    position="[Model unavailable this round]",
                    argument="Model failed to respond. Continuing debate without this participant.",
                    confidence=0.0,
                    raw_output=str(output),
                ))
            else:
                results.append(output)
        
        return results

    async def _execute_round_n(
        self, query: str, round_num: int, 
        transcript: str, previous_rounds: List[List[ModelRoundOutput]],
        role_map: Dict[str, str] = None,
    ) -> List[ModelRoundOutput]:
        """Round 2+: Each model gets full transcript + must rebut opponents with role enforcement."""
        role_map = role_map or {}
        tasks = []
        
        for model in DEBATE_MODELS:
            own_previous = self._get_model_previous(model["id"], previous_rounds)
            role = role_map.get(model["id"], "neutral")
            role_instruction = ROLE_INSTRUCTIONS.get(role, "")
            
            system_prompt = ROUND_N_SYSTEM.format(
                model_name=model["label"],
                round_num=round_num,
                transcript=transcript,
                own_previous=own_previous,
                role_instruction=role_instruction,
            )
            user_prompt = (
                f"ORIGINAL TOPIC:\n{query}\n\n"
                f"This is Round {round_num}. You must rebut your opponents and strengthen your position."
            )
            
            caller = self._model_callers[model["provider"]]
            tasks.append(self._call_and_parse_round_n(caller, user_prompt, system_prompt, model, round_num))
        
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, output in enumerate(outputs):
            if isinstance(output, Exception):
                logger.error(f"{DEBATE_MODELS[i]['label']} Round {round_num} failed: {output}")
                results.append(ModelRoundOutput(
                    model_id=DEBATE_MODELS[i]["id"],
                    model_label=DEBATE_MODELS[i]["label"],
                    model_name=DEBATE_MODELS[i]["name"],
                    model_color=DEBATE_MODELS[i]["color"],
                    round_num=round_num,
                    position="[Model unavailable this round]",
                    argument="Model failed to respond.",
                    confidence=0.0,
                    raw_output=str(output),
                ))
            else:
                results.append(output)
        
        return results

    async def _run_analysis(self, transcript: str) -> DebateAnalysis:
        """
        Run final debate analysis using the best available reasoning model.
        
        Tries models in priority order with retry logic. Falls back gracefully
        if all models fail.
        """
        debater_names = ", ".join(m["label"] for m in DEBATE_MODELS)
        system_prompt = ANALYSIS_SYSTEM.format(
            transcript=transcript, debater_names=debater_names
        )
        user_prompt = "Analyze this debate and produce your structured analysis."

        # Try analysis models in priority order
        available_ids = list(self._model_callers.keys())
        candidates = [m for m in self._ANALYSIS_MODEL_PRIORITY if m in available_ids]
        # Append any remaining models not in the priority list
        candidates.extend(m for m in available_ids if m not in candidates)

        last_error = None
        for model_id in candidates:
            for attempt in range(2):  # 2 attempts per model
                try:
                    raw = await self._model_callers[model_id](user_prompt, system_prompt)
                    if raw and not raw.strip().startswith(("Error", "Model '")):
                        logger.info(f"Debate analysis completed by {model_id} (attempt {attempt + 1})")
                        return self._parse_analysis(raw)
                    else:
                        logger.warning(f"Analysis model {model_id} returned error: {raw[:100] if raw else 'empty'}")
                        last_error = raw
                except Exception as e:
                    logger.warning(f"Analysis model {model_id} attempt {attempt + 1} failed: {e}")
                    last_error = str(e)

        logger.error(f"All analysis models failed. Last error: {last_error}")
        return DebateAnalysis(
            synthesis="Analysis generation failed across all models. Review individual round outputs.",
            confidence_recalibration=0.5,
        )

    # ============================================================
    # LLM CALL + PARSE HELPERS
    # ============================================================

    async def _call_and_parse_round1(
        self, caller, user_prompt: str, system_prompt: str, model: dict
    ) -> ModelRoundOutput:
        """Call a model for Round 1 and parse the structured output."""
        raw = await caller(user_prompt, system_prompt)
        parsed = self._parse_round1_output(raw)
        
        return ModelRoundOutput(
            model_id=model["id"],
            model_label=model["label"],
            model_name=model["name"],
            model_color=model["color"],
            round_num=1,
            position=parsed.get("position", ""),
            argument=parsed.get("argument", ""),
            assumptions=parsed.get("assumptions", []),
            risks=parsed.get("risks", []),
            confidence=parsed.get("confidence", 0.5),
            raw_output=raw,
        )

    async def _call_and_parse_round_n(
        self, caller, user_prompt: str, system_prompt: str, model: dict, round_num: int
    ) -> ModelRoundOutput:
        """Call a model for Round N and parse the structured output."""
        raw = await caller(user_prompt, system_prompt)
        parsed = self._parse_round_n_output(raw)
        
        return ModelRoundOutput(
            model_id=model["id"],
            model_label=model["label"],
            model_name=model["name"],
            model_color=model["color"],
            round_num=round_num,
            position=parsed.get("position", ""),
            argument=parsed.get("argument", ""),
            rebuttals=parsed.get("rebuttals", ""),
            position_shift=parsed.get("position_shift", "none"),
            weaknesses_found=parsed.get("weaknesses_found", ""),
            confidence=parsed.get("confidence", 0.5),
            raw_output=raw,
        )

    # ============================================================
    # OUTPUT PARSERS
    # ============================================================

    def _parse_round1_output(self, raw: str) -> Dict[str, Any]:
        """Parse Round 1 structured output from model."""
        result = {
            "position": "",
            "argument": "",
            "assumptions": [],
            "risks": [],
            "confidence": 0.5,
        }
        
        # Extract labeled sections
        result["position"] = self._extract_section(raw, "POSITION")
        result["argument"] = self._extract_section(raw, "ARGUMENT")
        
        # Parse bullet lists
        assumptions_raw = self._extract_section(raw, "ASSUMPTIONS")
        result["assumptions"] = self._parse_bullets(assumptions_raw)
        
        risks_raw = self._extract_section(raw, "RISKS")
        result["risks"] = self._parse_bullets(risks_raw)
        
        # Parse confidence
        conf_raw = self._extract_section(raw, "CONFIDENCE")
        result["confidence"] = self._parse_float(conf_raw, 0.5)
        
        # Fallback: if parsing failed, use raw text as position
        if not result["position"] and not result["argument"]:
            result["position"] = raw[:200] if raw else "[No response]"
            result["argument"] = raw if raw else "[No response]"
        
        return result

    def _parse_round_n_output(self, raw: str) -> Dict[str, Any]:
        """Parse Round N structured output from model."""
        result = {
            "rebuttals": "",
            "position": "",
            "argument": "",
            "position_shift": "none",
            "weaknesses_found": "",
            "confidence": 0.5,
        }
        
        result["rebuttals"] = self._extract_section(raw, "REBUTTALS")
        result["position"] = self._extract_section(raw, "POSITION")
        result["argument"] = self._extract_section(raw, "ARGUMENT")
        result["position_shift"] = self._extract_section(raw, "POSITION_SHIFT") or "none"
        result["weaknesses_found"] = self._extract_section(raw, "WEAKNESSES_FOUND")
        conf_raw = self._extract_section(raw, "CONFIDENCE")
        result["confidence"] = self._parse_float(conf_raw, 0.5)
        
        # Fallback
        if not result["position"] and not result["rebuttals"]:
            result["position"] = raw[:200] if raw else "[No response]"
            result["argument"] = raw if raw else "[No response]"
        
        return result

    def _parse_analysis(self, raw: str) -> DebateAnalysis:
        """Parse the final debate analysis output."""
        analysis = DebateAnalysis()
        
        # Parse each field
        conflict_raw = self._extract_section(raw, "CONFLICT_AXES")
        analysis.conflict_axes = self._parse_bullets(conflict_raw) if conflict_raw else []
        
        analysis.disagreement_strength = self._parse_float(
            self._extract_section(raw, "DISAGREEMENT_STRENGTH"), 0.5
        )
        analysis.logical_stability = self._parse_float(
            self._extract_section(raw, "LOGICAL_STABILITY"), 0.5
        )
        
        shifts_raw = self._extract_section(raw, "POSITION_SHIFTS")
        analysis.position_shifts = self._parse_bullets(shifts_raw) if shifts_raw else []
        
        analysis.convergence_level = (
            self._extract_section(raw, "CONVERGENCE_LEVEL") or "none"
        ).strip().lower()
        analysis.convergence_detail = self._extract_section(raw, "CONVERGENCE_DETAIL") or ""
        analysis.strongest_argument = self._extract_section(raw, "STRONGEST_ARGUMENT") or ""
        analysis.weakest_argument = self._extract_section(raw, "WEAKEST_ARGUMENT") or ""
        analysis.confidence_recalibration = self._parse_float(
            self._extract_section(raw, "CONFIDENCE_RECALIBRATION"), 0.5
        )
        analysis.synthesis = self._extract_section(raw, "SYNTHESIS") or raw[-500:] if raw else ""
        
        return analysis

    # ============================================================
    # JUDGE SCORING & DISAGREEMENT QUANTIFICATION
    # ============================================================

    async def _run_judge_scoring(self, transcript: str, judge_model: str) -> Dict[str, Any]:
        """Run judge scoring using the assigned judge model."""
        debater_names = ", ".join(m["label"] for m in DEBATE_MODELS)
        scoring_entries = "\n".join(
            f"- {m['label']}: logic={{score}} evidence={{score}} coherence={{score}} rebuttal={{score}} total={{score}}"
            for m in DEBATE_MODELS
        )
        system_prompt = JUDGE_SCORING_SYSTEM.format(
            transcript=transcript,
            debater_names=debater_names,
            scoring_entries=scoring_entries,
        )
        user_prompt = "Score each debater and determine the winner."
        
        caller = self._model_callers.get(judge_model)
        if not caller:
            return {"error": f"Judge model '{judge_model}' not available"}
        
        try:
            raw = await caller(user_prompt, system_prompt)
            scoring = {
                "strongest_argument": self._extract_section(raw, "STRONGEST_ARGUMENT"),
                "weakest_argument": self._extract_section(raw, "WEAKEST_ARGUMENT"),
                "stability_assessment": self._extract_section(raw, "STABILITY_ASSESSMENT"),
                "winner": self._extract_section(raw, "WINNER"),
                "raw_scoring": self._extract_section(raw, "SCORING_MATRIX"),
            }
            
            # Parse scoring matrix into structured data
            scoring_matrix = {}
            for model in DEBATE_MODELS:
                model_line = ""
                scoring_raw = scoring.get("raw_scoring", "")
                for line in scoring_raw.split("\n"):
                    if model["label"].lower() in line.lower():
                        model_line = line
                        break
                
                if model_line:
                    scores = {}
                    for metric in ["logic", "evidence", "coherence", "rebuttal", "total"]:
                        match = re.search(rf'{metric}\s*=\s*(\d+\.?\d*)', model_line, re.IGNORECASE)
                        scores[metric] = float(match.group(1)) if match else 0.5
                    scoring_matrix[model["label"]] = scores
                else:
                    scoring_matrix[model["label"]] = {
                        "logic": 0.5, "evidence": 0.5, "coherence": 0.5,
                        "rebuttal": 0.5, "total": 0.5
                    }
            
            scoring["scoring_matrix"] = scoring_matrix
            return scoring
            
        except Exception as e:
            logger.error(f"Judge scoring failed: {e}")
            return {"error": str(e)}

    def _compute_round_disagreement(self, round_outputs: List[ModelRoundOutput]) -> float:
        """
        Compute disagreement score for a single round.
        
        Uses mean pairwise position difference as a proxy for semantic divergence.
        Considers:
        - Position text similarity (simple word overlap)
        - Confidence spread
        - Explicit position shifts
        """
        if len(round_outputs) < 2:
            return 0.0
        
        # Method 1: Confidence spread (normalized standard deviation)
        confidences = [mo.confidence for mo in round_outputs if mo.confidence > 0]
        if len(confidences) >= 2:
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            conf_spread = variance ** 0.5  # std dev
        else:
            conf_spread = 0.0
        
        # Method 2: Position word overlap (pairwise Jaccard distance)
        positions = [set(mo.position.lower().split()) for mo in round_outputs if mo.position]
        pairwise_distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i] or positions[j]:
                    intersection = len(positions[i] & positions[j])
                    union = len(positions[i] | positions[j])
                    jaccard = intersection / max(union, 1)
                    pairwise_distances.append(1.0 - jaccard)  # Distance
        
        word_disagreement = sum(pairwise_distances) / max(len(pairwise_distances), 1)
        
        # Method 3: Position shift signals
        shift_count = sum(
            1 for mo in round_outputs 
            if mo.position_shift and mo.position_shift.lower() not in ("none", "")
        )
        shift_factor = min(shift_count / len(round_outputs), 1.0) * 0.3
        
        # Weighted combination
        disagreement = (
            conf_spread * 0.25
            + word_disagreement * 0.50
            + shift_factor * 0.25
        )
        
        return round(min(1.0, disagreement), 4)

    # ============================================================
    # TEXT EXTRACTION UTILITIES
    # ============================================================

    def _extract_section(self, text: str, label: str) -> str:
        """Extract content after a labeled section header."""
        if not text:
            return ""
        
        # Pattern: LABEL: content (until next LABEL: or end)
        # Support both "LABEL:" and "**LABEL:**" and "LABEL :"
        pattern = rf'(?:^|\n)\s*\**{re.escape(label)}\**\s*:\s*(.*?)(?=\n\s*\**[A-Z_]+\**\s*:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_bullets(self, text: str) -> List[str]:
        """Parse bullet list from text."""
        if not text:
            return []
        bullets = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("-", "•", "*", "–")):
                item = line.lstrip("-•*– ").strip()
                if item:
                    bullets.append(item)
            elif line and not any(line.startswith(p) for p in ["POSITION", "ARGUMENT", "CONFIDENCE"]):
                # Single-line non-header content
                if len(bullets) == 0:
                    bullets.append(line)
        return bullets

    def _parse_float(self, text: str, default: float = 0.5) -> float:
        """Parse a float from text."""
        if not text:
            return default
        # Find first number in text
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return default

    def _get_model_previous(self, model_id: str, previous_rounds: List[List[ModelRoundOutput]]) -> str:
        """Get a model's most recent output for transcript injection."""
        if not previous_rounds:
            return "[No previous output]"
        
        last_round = previous_rounds[-1]
        for mo in last_round:
            if mo.model_id == model_id:
                parts = []
                if mo.position:
                    parts.append(f"Position: {mo.position}")
                if mo.argument:
                    parts.append(f"Argument: {mo.argument}")
                if mo.confidence:
                    parts.append(f"Confidence: {mo.confidence}")
                return "\n".join(parts) if parts else "[No substantive output]"
        
        return "[No previous output found]"

    def _format_round_transcript(self, round_num: int, outputs: List[ModelRoundOutput]) -> str:
        """Format a round's outputs into transcript text for injection."""
        parts = [f"=== ROUND {round_num} ==="]
        
        for mo in outputs:
            parts.append(f"\n--- {mo.model_label} ({mo.model_name}) ---")
            if mo.rebuttals:
                parts.append(f"Rebuttals: {mo.rebuttals}")
            if mo.position:
                parts.append(f"Position: {mo.position}")
            if mo.argument:
                parts.append(f"Argument: {mo.argument}")
            if mo.assumptions:
                parts.append(f"Assumptions: {', '.join(mo.assumptions)}")
            if mo.risks:
                parts.append(f"Risks: {', '.join(mo.risks)}")
            if mo.weaknesses_found:
                parts.append(f"Weaknesses Found: {mo.weaknesses_found}")
            if mo.position_shift and mo.position_shift != "none":
                parts.append(f"Position Shift: {mo.position_shift}")
            parts.append(f"Confidence: {mo.confidence:.2f}")
        
        return "\n".join(parts)
