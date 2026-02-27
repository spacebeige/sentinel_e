"""
============================================================
Cognitive Orchestrator — Sentinel-E v6.0
============================================================
THE SINGLE ENTRY POINT for all requests.

No mode-based routing. No bypass paths. No single-model fallback.

Every request:
  1. Validates minimum 3 models available
  2. Executes ALL enabled models in parallel (structured output)
  3. Builds agreement matrix (pairwise similarity)
  4. Runs minimum 3 rounds structured debate
  5. Computes ensemble metrics (entropy, contradiction, stability, velocity, fragility)
  6. Calibrates confidence from metrics (not self-reported)
  7. Builds tactical map
  8. Updates session intelligence
  9. Synthesizes final output
  10. Returns EnsembleResponse (full metadata)

Hard failures:
  - <3 models: INSUFFICIENT_MODELS
  - <3 rounds: INSUFFICIENT_ROUNDS
  - <2 analytics outputs: INSUFFICIENT_ANALYTICS
  - Empty tactical map: EMPTY_TACTICAL_MAP
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from core.ensemble_schemas import (
    MIN_MODELS,
    MIN_DEBATE_ROUNDS,
    MIN_ANALYTICS_OUTPUTS,
    EnsembleResponse,
    StructuredModelOutput,
    StanceVector,
    AgreementMatrix,
    DebateResult,
    EnsembleMetrics,
    CalibratedConfidence,
    TacticalMap,
    TacticalMapEntry,
    SessionIntelligenceSnapshot,
    EnsembleFailure,
    EnsembleFailureCode,
)
from core.agreement_matrix import AgreementMatrixEngine
from core.structured_debate_engine import StructuredDebateEngine
from core.confidence_calibrator import ConfidenceCalibrator
from core.ensemble_session import EnsembleSessionEngine

logger = logging.getLogger("CognitiveOrchestrator")


# ============================================================
# Structured Extraction Prompt
# ============================================================

STRUCTURED_EXTRACTION_SYSTEM = """You are an analytical model. Answer the following query with structured reasoning.

QUERY: {query}

You MUST respond with EXACTLY this structure:

POSITION: [Your clear thesis/answer in 1-2 sentences]

REASONING: [Step-by-step reasoning chain. Be specific and evidence-based.]

ASSUMPTIONS: [List assumptions, one per line with "- " prefix]
- assumption 1

VULNERABILITIES: [Self-identified weaknesses, one per line with "- " prefix]
- weakness 1

CONFIDENCE: [A number 0.0 to 1.0]

STANCE:
certainty: [0.0-1.0]
specificity: [0.0-1.0]
risk_tolerance: [0.0-1.0]
evidence_reliance: [0.0-1.0]
novelty: [0.0-1.0]

Be thorough but honest about uncertainties."""


class CognitiveOrchestrator:
    """
    The single execution engine for Sentinel-E v6.0.

    All requests flow through process(). No mode routing.
    No bypass. No single-model fallback.
    """

    def __init__(self, model_bridge):
        """
        Args:
            model_bridge: MCOModelBridge instance with call_model(),
                          get_enabled_model_ids(), get_enabled_models_info()
        """
        self._bridge = model_bridge
        self._agreement_engine = AgreementMatrixEngine()
        self._confidence_calibrator = ConfidenceCalibrator()
        self._debate_engine = StructuredDebateEngine(
            call_model=self._call_model,
            get_enabled_models=self._get_models,
        )
        # Session engines are per-session, stored externally
        self._sessions: Dict[str, EnsembleSessionEngine] = {}

    async def process(
        self,
        query: str,
        chat_id: str = "",
        chat_name: str = "",
        rounds: int = MIN_DEBATE_ROUNDS,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> EnsembleResponse:
        """
        THE SINGLE ENTRY POINT.

        Every request goes through this pipeline:
          Phase 1: Parallel structured model execution
          Phase 2: Agreement matrix
          Phase 3: Structured debate (3+ rounds)
          Phase 4: Ensemble metrics
          Phase 5: Confidence calibration
          Phase 6: Tactical map
          Phase 7: Session intelligence
          Phase 8: Synthesis
          Phase 9: Hard failure validation
          Phase 10: EnsembleResponse assembly
        """
        start_time = time.monotonic()
        rounds = max(rounds, MIN_DEBATE_ROUNDS)

        if not chat_id:
            chat_id = str(uuid.uuid4())

        # Initialize session
        session = self._get_or_create_session(chat_id)

        try:
            # ── PHASE 1: Parallel Structured Model Execution ───
            logger.info(f"Phase 1: Parallel execution for chat {chat_id}")
            models = self._get_models()
            self._validate_minimum_models(models)

            structured_outputs = await self._phase1_parallel_execution(
                query, models
            )

            # Validate sufficient successful outputs
            successful = [o for o in structured_outputs if o.succeeded]
            if len(successful) < MIN_MODELS:
                raise EnsembleFailure(
                    code=EnsembleFailureCode.INSUFFICIENT_MODELS,
                    message=(
                        f"Only {len(successful)} models succeeded, "
                        f"minimum {MIN_MODELS} required. "
                        f"Failed: {[o.model_id for o in structured_outputs if not o.succeeded]}"
                    ),
                    models_available=len(successful),
                )

            # ── PHASE 2: Agreement Matrix ──────────────────────
            logger.info("Phase 2: Computing agreement matrix")
            agreement_matrix = self._agreement_engine.compute(successful)

            # ── PHASE 3: Structured Debate ─────────────────────
            logger.info(f"Phase 3: Running {rounds}-round debate")
            debate_result = await self._debate_engine.run_debate(
                query=query,
                rounds=rounds,
                initial_outputs=successful,
            )

            # ── PHASE 4: Ensemble Metrics ──────────────────────
            logger.info("Phase 4: Computing ensemble metrics")
            entropy = self._agreement_engine.compute_disagreement_entropy(
                successful
            )
            contradiction_density = (
                self._agreement_engine.compute_contradiction_density(successful)
            )
            ensemble_metrics = self._confidence_calibrator.compute_ensemble_metrics(
                outputs=structured_outputs,
                matrix=agreement_matrix,
                debate=debate_result,
                disagreement_entropy=entropy,
                contradiction_density=contradiction_density,
            )

            # Validate analytics inputs
            if ensemble_metrics.successful_models < MIN_ANALYTICS_OUTPUTS:
                raise EnsembleFailure(
                    code=EnsembleFailureCode.INSUFFICIENT_ANALYTICS,
                    message=(
                        f"Analytics requires minimum {MIN_ANALYTICS_OUTPUTS} "
                        f"outputs, got {ensemble_metrics.successful_models}"
                    ),
                    models_available=ensemble_metrics.successful_models,
                )

            # ── PHASE 5: Confidence Calibration ────────────────
            logger.info("Phase 5: Calibrating confidence")
            calibrated_confidence = self._confidence_calibrator.calibrate(
                metrics=ensemble_metrics,
                matrix=agreement_matrix,
                debate=debate_result,
                outputs=successful,
            )

            # ── PHASE 6: Tactical Map ──────────────────────────
            logger.info("Phase 6: Building tactical map")
            tactical_map = self._build_tactical_map(
                successful, agreement_matrix, debate_result
            )

            if not tactical_map.entries:
                raise EnsembleFailure(
                    code=EnsembleFailureCode.EMPTY_TACTICAL_MAP,
                    message="Tactical map is empty — no model positions available",
                    models_available=len(successful),
                )

            # ── PHASE 7: Session Intelligence ──────────────────
            logger.info("Phase 7: Updating session intelligence")
            session_snapshot = session.update(
                query=query,
                outputs=structured_outputs,
                metrics=ensemble_metrics,
                matrix=agreement_matrix,
                confidence=calibrated_confidence,
                boundary_hit=ensemble_metrics.fragility_score > 0.7,
            )

            # ── PHASE 8: Synthesis ─────────────────────────────
            logger.info("Phase 8: Synthesizing output")
            formatted_output = self._synthesize_output(
                query=query,
                outputs=successful,
                debate=debate_result,
                matrix=agreement_matrix,
                metrics=ensemble_metrics,
                confidence=calibrated_confidence,
                tactical_map=tactical_map,
            )

            # ── PHASE 9: Confidence Evolution ──────────────────
            conf_evolution = {
                "initial": calibrated_confidence.evolution[0]["value"]
                    if calibrated_confidence.evolution else 0.0,
                "post_debate": calibrated_confidence.evolution[1]["value"]
                    if len(calibrated_confidence.evolution) > 1 else 0.0,
                "post_calibration": calibrated_confidence.final_confidence,
                "final": calibrated_confidence.final_confidence,
            }

            # ── PHASE 10: Assembly ─────────────────────────────
            total_latency = (time.monotonic() - start_time) * 1000

            omega_metadata = {
                "omega_version": "6.0.0",
                "pipeline": "ensemble_cognitive",
                "total_latency_ms": round(total_latency, 1),
                "models_executed": len(structured_outputs),
                "models_succeeded": len(successful),
                "debate_rounds": debate_result.total_rounds,
                "mean_agreement": round(agreement_matrix.mean_agreement, 4),
                "disagreement_entropy": round(ensemble_metrics.disagreement_entropy, 4),
                "contradiction_density": round(ensemble_metrics.contradiction_density, 4),
                "stability_index": round(ensemble_metrics.stability_index, 4),
                "consensus_velocity": round(ensemble_metrics.consensus_velocity, 4),
                "fragility_score": round(ensemble_metrics.fragility_score, 4),
                "calibrated_confidence": round(calibrated_confidence.final_confidence, 4),
                "confidence_method": calibrated_confidence.calibration_method,
            }

            response = EnsembleResponse(
                chat_id=chat_id,
                chat_name=chat_name or query[:40],
                formatted_output=formatted_output,
                synthesis_method="ensemble_weighted",
                model_outputs=structured_outputs,
                models_executed=len(structured_outputs),
                models_succeeded=len(successful),
                models_failed=len(structured_outputs) - len(successful),
                debate_result=debate_result,
                agreement_matrix=agreement_matrix,
                ensemble_metrics=ensemble_metrics,
                confidence=calibrated_confidence,
                tactical_map=tactical_map,
                session_intelligence=session_snapshot,
                confidence_evolution=conf_evolution,
                omega_metadata=omega_metadata,
            )

            logger.info(
                f"Ensemble complete: {len(successful)} models, "
                f"{debate_result.total_rounds} rounds, "
                f"confidence {calibrated_confidence.final_confidence:.2f}, "
                f"latency {total_latency:.0f}ms"
            )

            return response

        except EnsembleFailure as e:
            logger.error(f"Hard failure: {e}")
            return e.to_response()

        except Exception as e:
            logger.exception(f"Unexpected error in CognitiveOrchestrator: {e}")
            return EnsembleResponse(
                chat_id=chat_id,
                error=f"Internal error: {str(e)}",
                error_code="INTERNAL_ERROR",
                formatted_output=f"**System Error**: {str(e)}",
            )

    # ── Phase 1: Parallel Execution ──────────────────────────

    async def _phase1_parallel_execution(
        self, query: str, models: List[Dict[str, str]]
    ) -> List[StructuredModelOutput]:
        """Execute all models in parallel with structured output extraction."""
        system_prompt = STRUCTURED_EXTRACTION_SYSTEM.format(query=query)

        tasks = [
            self._call_and_extract(model, query, system_prompt)
            for model in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model {models[i]['id']} failed: {result}")
                outputs.append(StructuredModelOutput(
                    model_id=models[i]["id"],
                    model_name=models[i].get("name", models[i]["id"]),
                    error=str(result),
                ))
            elif result is not None:
                outputs.append(result)
            else:
                # Never silently drop a model — record explicit failure
                logger.error(f"Model {models[i]['id']} returned None")
                outputs.append(StructuredModelOutput(
                    model_id=models[i]["id"],
                    model_name=models[i].get("name", models[i]["id"]),
                    error="No response returned",
                ))

        return outputs

    async def _call_and_extract(
        self,
        model: Dict[str, str],
        query: str,
        system_prompt: str,
    ) -> StructuredModelOutput:
        """Call a single model and extract structured output."""
        start = time.monotonic()
        try:
            raw = await self._call_model(model["id"], query, system_prompt)
            latency = (time.monotonic() - start) * 1000

            if not raw or len(raw.strip()) < 10:
                return StructuredModelOutput(
                    model_id=model["id"],
                    model_name=model.get("name", model["id"]),
                    error="Empty or minimal output",
                    latency_ms=latency,
                )

            # Parse structured output
            return self._parse_structured_output(
                raw, model["id"], model.get("name", model["id"]), latency
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return StructuredModelOutput(
                model_id=model["id"],
                model_name=model.get("name", model["id"]),
                error=str(e),
                latency_ms=latency,
            )

    def _parse_structured_output(
        self, raw: str, model_id: str, model_name: str, latency_ms: float
    ) -> StructuredModelOutput:
        """Parse raw model output into StructuredModelOutput."""
        position = self._extract_section(raw, "POSITION")
        reasoning = self._extract_section(raw, "REASONING")
        assumptions = self._extract_list(raw, "ASSUMPTIONS")
        vulnerabilities = self._extract_list(raw, "VULNERABILITIES")
        confidence = self._extract_float(raw, "CONFIDENCE", 0.5)
        stance = self._extract_stance(raw)

        # Fallback if structured parsing fails
        if not position:
            # Use first sentence as position
            sentences = re.split(r'[.!?]+', raw)
            position = sentences[0].strip() if sentences else raw[:200]
        if not reasoning:
            reasoning = raw

        return StructuredModelOutput(
            model_id=model_id,
            model_name=model_name,
            position=position,
            reasoning=reasoning,
            assumptions=assumptions,
            vulnerabilities=vulnerabilities,
            confidence=confidence,
            stance_vector=stance,
            raw_output=raw,
            latency_ms=latency_ms,
        )

    # ── Tactical Map ─────────────────────────────────────────

    def _build_tactical_map(
        self,
        outputs: List[StructuredModelOutput],
        matrix: AgreementMatrix,
        debate: DebateResult,
    ) -> TacticalMap:
        """Build tactical map from model outputs."""
        # Use final debate positions if available, otherwise Phase 1 outputs
        final_positions: Dict[str, Any] = {}
        if debate.rounds:
            last_round = debate.rounds[-1]
            for pos in last_round.positions:
                final_positions[pos.model_id] = pos

        entries = []
        consensus_pos = debate.final_consensus or ""

        for output in outputs:
            # Get final debate position if available
            debate_pos = final_positions.get(output.model_id)
            position_summary = (
                debate_pos.position if debate_pos else output.position
            )
            confidence = (
                debate_pos.confidence if debate_pos else output.confidence
            )
            stance = (
                debate_pos.stance_vector if debate_pos else output.stance_vector
            )
            vulnerabilities = (
                debate_pos.vulnerabilities_found if debate_pos
                else output.vulnerabilities
            )

            # Compute agreement with consensus
            consensus_agreement = 0.0
            if consensus_pos:
                cons_words = set(re.findall(r'[a-z]+', consensus_pos.lower()))
                pos_words = set(re.findall(r'[a-z]+', position_summary.lower()))
                union = cons_words | pos_words
                inter = cons_words & pos_words
                consensus_agreement = len(inter) / len(union) if union else 0.0

            # Key differentiator
            differentiator = ""
            if vulnerabilities:
                differentiator = f"Key concern: {vulnerabilities[0]}"
            elif output.assumptions:
                differentiator = f"Key assumption: {output.assumptions[0]}"

            entries.append(TacticalMapEntry(
                model_id=output.model_id,
                model_name=output.model_name,
                position_summary=position_summary[:200],
                confidence=confidence,
                stance_vector=stance,
                agreement_with_consensus=consensus_agreement,
                key_differentiator=differentiator,
                vulnerabilities=vulnerabilities[:3],
            ))

        # Primary axis of disagreement
        primary_axis = ""
        if matrix.dissenting_models:
            primary_axis = (
                f"Models {', '.join(matrix.dissenting_models)} "
                f"diverge from majority consensus"
            )
        elif debate.unresolved_conflicts:
            primary_axis = debate.unresolved_conflicts[0]

        # Model clusters
        model_clusters = [
            {"models": cluster, "type": "agreement_cluster"}
            for cluster in matrix.agreement_clusters
        ]

        return TacticalMap(
            entries=entries,
            consensus_position=consensus_pos or "",
            primary_axis_of_disagreement=primary_axis,
            model_clusters=model_clusters,
        )

    # ── Synthesis ────────────────────────────────────────────

    def _synthesize_output(
        self,
        query: str,
        outputs: List[StructuredModelOutput],
        debate: DebateResult,
        matrix: AgreementMatrix,
        metrics: EnsembleMetrics,
        confidence: CalibratedConfidence,
        tactical_map: TacticalMap,
    ) -> str:
        """Synthesize final markdown output from ensemble results."""
        parts = []

        # ── Consensus Answer ────────────────────────────────
        if debate.final_consensus:
            parts.append(debate.final_consensus)
        else:
            # Use highest-agreement model's position
            best_output = max(outputs, key=lambda o: o.confidence)
            parts.append(best_output.position)

        parts.append("")  # Blank line

        # ── Ensemble Synthesis ──────────────────────────────
        # Merge unique reasoning points from all models
        all_reasoning_points = []
        seen_points = set()
        for output in sorted(outputs, key=lambda o: -o.confidence):
            # Extract key sentences from reasoning
            sentences = re.split(r'[.!?]+', output.reasoning)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20 and sent.lower() not in seen_points:
                    seen_points.add(sent.lower())
                    all_reasoning_points.append(sent)

        if all_reasoning_points:
            parts.append("")
            for point in all_reasoning_points[:8]:
                parts.append(f"{point}.")

        # ── Divergence Notice ───────────────────────────────
        if metrics.contradiction_density > 0.3:
            parts.append("")
            parts.append(
                f"⚠️ **Note**: Significant divergence detected across models "
                f"(contradiction density: {metrics.contradiction_density:.0%}). "
                f"Consider multiple perspectives."
            )

        if matrix.dissenting_models:
            dissenters = ", ".join(matrix.dissenting_models)
            parts.append(
                f"Models with differing views: {dissenters}"
            )

        # ── Unresolved ──────────────────────────────────────
        if debate.unresolved_conflicts:
            parts.append("")
            parts.append("**Unresolved points:**")
            for conflict in debate.unresolved_conflicts[:3]:
                parts.append(f"- {conflict}")

        return "\n".join(parts)

    # ── Model Bridge Wrappers ────────────────────────────────

    async def _call_model(
        self, model_id: str, prompt: str, system_role: str
    ) -> str:
        """Call a model through the bridge."""
        return await self._bridge.call_model(model_id, prompt, system_role)

    def _get_models(self) -> List[Dict[str, str]]:
        """Get enabled models from bridge."""
        return self._bridge.get_enabled_models_info()

    def _validate_minimum_models(self, models: List[Dict[str, str]]):
        """Validate minimum model count."""
        if len(models) < MIN_MODELS:
            raise EnsembleFailure(
                code=EnsembleFailureCode.INSUFFICIENT_MODELS,
                message=(
                    f"Ensemble requires minimum {MIN_MODELS} models, "
                    f"only {len(models)} enabled. Check API keys."
                ),
                models_available=len(models),
            )

    # ── Session Management ───────────────────────────────────

    def _get_or_create_session(
        self, chat_id: str
    ) -> EnsembleSessionEngine:
        """Get or create session for chat."""
        if chat_id not in self._sessions:
            self._sessions[chat_id] = EnsembleSessionEngine(
                session_id=chat_id
            )
        return self._sessions[chat_id]

    def get_session_snapshot(
        self, chat_id: str
    ) -> Optional[SessionIntelligenceSnapshot]:
        """Get session snapshot for API."""
        session = self._sessions.get(chat_id)
        if session:
            return session.snapshot()
        return None

    # ── Parsing Helpers ──────────────────────────────────────

    def _extract_section(self, text: str, header: str) -> str:
        pattern = rf'{header}:\s*\[?(.*?)(?:\]?\s*\n(?=[A-Z_]+:)|\]?\s*$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('[]')
        pattern2 = rf'{header}:\s*(.*?)(?:\n[A-Z_]+:|\Z)'
        match2 = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return ""

    def _extract_list(self, text: str, header: str) -> List[str]:
        section = self._extract_section(text, header)
        if not section:
            return []
        items = re.findall(r'[-•*]\s*(.+)', section)
        return [item.strip() for item in items if item.strip()]

    def _extract_float(
        self, text: str, header: str, default: float = 0.5
    ) -> float:
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


# ── v7.0 alias — main.py imports this name ──────────────────
CognitiveCoreEngine = CognitiveOrchestrator
