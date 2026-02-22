"""
Omega Cognitive Kernel v3.X — Sentinel-E Cognitive Engine

Central orchestrator for the Omega Cognitive Kernel system.
COMPLETE REWRITE from v2.x architecture.

Mode System:
  MODE = { STANDARD, RESEARCH }
  RESEARCH.sub_modes = { DEBATE, GLASS, EVIDENCE, STRESS }
  KILL = diagnostic state inside GLASS

Architecture:
  API Request → ModeConfig → OmegaKernel.process()
    → SessionIntelligence.update()
    → MultiPassReasoning.pre_passes(1-4)
    → Mode-specific pipeline:
        STANDARD:  Sigma LLM → ConfidenceEngine → single answer
        DEBATE:    DebateOrchestrator (named models, roles) → consensus confidence
        GLASS:     LLM + BehavioralAnalytics + ConfidenceTrace (or KILL diagnostic)
        EVIDENCE:  EvidenceEngine.run_full_evidence_analysis() → claim mapping
        STRESS:    StressEngine → adversarial attack → stability analysis
    → MultiPassReasoning.post_passes(5-9)
    → OmegaFormatter.format(mode)
    → SentinelOmegaResponse
"""

import logging
import uuid
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from backend.core.mode_config import ModeConfig, Mode, SubMode
from backend.core.session_intelligence import SessionIntelligence
from backend.core.multipass_reasoning import MultiPassReasoningEngine
from backend.core.omega_boundary import OmegaBoundaryEvaluator
from backend.core.omega_formatter import OmegaOutputFormatter
from backend.core.behavioral_analytics import BehavioralAnalyzer
from backend.core.evidence_engine import EvidenceEngine
from backend.core.debate_orchestrator import DebateOrchestrator
from backend.core.confidence_engine import ConfidenceEngine, ConfidenceTrace
from backend.core.stress_engine import StressEngine

logger = logging.getLogger("Omega-Kernel")


# ============================================================
# BACKWARD COMPATIBILITY — OmegaConfig alias
# ============================================================

VALID_MODES = {"standard", "research", "experimental", "kill", "conversational", "forensic"}
OMEGA_MODES = {"standard", "research", "experimental", "kill"}
VALID_SUB_MODES = {"debate", "glass", "evidence", "stress"}


@dataclass
class OmegaConfig:
    """
    BACKWARD COMPATIBILITY wrapper.
    
    New code should use ModeConfig directly.
    OmegaConfig.to_mode_config() converts to the new system.
    """
    text: str
    mode: str = "standard"
    sub_mode: str = "debate"
    kill: bool = False
    enable_shadow: bool = False
    rounds: int = 3
    chat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            logger.warning(f"Invalid mode '{self.mode}', defaulting to 'standard'")
            self.mode = "standard"
        if self.sub_mode not in VALID_SUB_MODES:
            self.sub_mode = "debate"
        if self.kill and self.sub_mode != "glass":
            logger.warning("Kill switch only valid in glass sub-mode. Forcing sub_mode=glass.")
            self.sub_mode = "glass"

    def to_mode_config(self) -> ModeConfig:
        """Convert legacy OmegaConfig → ModeConfig."""
        return ModeConfig.from_legacy(
            text=self.text,
            mode=self.mode,
            sub_mode=self.sub_mode,
            kill_switch=self.kill,
            enable_shadow=self.enable_shadow,
            rounds=self.rounds,
            chat_id=self.chat_id,
            history=self.history,
        )


class OmegaCognitiveKernel:
    """
    Omega Cognitive Kernel v3.X — Production-grade multi-agent AI research engine.

    Implements:
    - STANDARD mode: Calibrated assistant with mathematically defensible confidence
    - RESEARCH mode: Sub-modes for debate, glass, evidence, and stress testing
    - Adaptive learning: model weights influenced by historical performance
    - Session Intelligence 2.0: cognitive drift, topic stability, confidence volatility
    """

    def __init__(self, sigma_orchestrator=None, knowledge_learner=None):
        """
        Initialize the Omega Kernel.

        Args:
            sigma_orchestrator: The existing SentinelSigmaOrchestratorV4 instance.
            knowledge_learner: KnowledgeLearner instance for adaptive model weighting.
        """
        self.sigma = sigma_orchestrator
        self.knowledge_learner = knowledge_learner
        self.session = SessionIntelligence()
        self.reasoning = MultiPassReasoningEngine()
        self.boundary = OmegaBoundaryEvaluator()
        self.formatter = OmegaOutputFormatter()
        self.behavioral = BehavioralAnalyzer()
        self.evidence = EvidenceEngine()
        self.confidence_engine = ConfidenceEngine()
        self.debate = None
        self.stress = None
        self._initialized = False

        # Initialize model-dependent engines
        if self.sigma and hasattr(self.sigma, 'client'):
            self.debate = DebateOrchestrator(self.sigma.client)
            self.stress = StressEngine(self.sigma.client)

        logger.info("Omega Cognitive Kernel v3.X initialized.")

    # ============================================================
    # SESSION PERSISTENCE
    # ============================================================

    def serialize_session(self) -> Dict[str, Any]:
        """Serialize session state for Redis/Postgres persistence."""
        return {
            "session": self.session.serialize(),
            "initialized": self._initialized,
            "boundary_evaluations": self.boundary.get_trend(),
        }

    @classmethod
    def restore_from_session(cls, session_data: Dict[str, Any],
                             sigma_orchestrator=None,
                             knowledge_learner=None) -> "OmegaCognitiveKernel":
        """Restore kernel from persisted session data."""
        kernel = cls(sigma_orchestrator=sigma_orchestrator,
                     knowledge_learner=knowledge_learner)
        session_payload = session_data.get("session", {})
        if session_payload:
            kernel.session = SessionIntelligence.from_dict(session_payload)
            kernel._initialized = session_data.get("initialized", False)
        return kernel

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    async def process(self, config) -> Dict[str, Any]:
        """
        Main entry point for all Omega processing.

        Accepts either ModeConfig (v3.X) or OmegaConfig (legacy).

        Routes based on ModeConfig:
          STANDARD → _process_standard
          RESEARCH → sub_mode routing:
            DEBATE   → _process_debate
            GLASS    → _process_glass (with optional kill_override)
            EVIDENCE → _process_evidence
            STRESS   → _process_stress
        """
        # Convert legacy OmegaConfig if necessary
        if isinstance(config, OmegaConfig):
            config = config.to_mode_config()
        elif not isinstance(config, ModeConfig):
            # Dict fallback
            config = ModeConfig.from_legacy(**config) if isinstance(config, dict) else config

        logger.info(
            f"Omega processing: mode={config.mode.value}, "
            f"sub_mode={config.sub_mode.value if config.is_research else 'N/A'}, "
            f"text_len={len(config.text)}"
        )

        # 1. Session Intelligence Update
        if not self._initialized:
            self.session.initialize(config.text, config.mode.value)
            self._initialized = True
            is_first = True
        else:
            is_first = False

        # 2. Mode Routing
        if config.is_standard:
            return await self._process_standard(config, is_first)
        elif config.is_debate:
            return await self._process_debate(config, is_first)
        elif config.is_glass:
            return await self._process_glass(config, is_first)
        elif config.is_evidence:
            return await self._process_evidence(config, is_first)
        elif config.is_stress:
            return await self._process_stress(config, is_first)
        else:
            # Fallback to standard
            return await self._process_standard(config, is_first)

    # ============================================================
    # STANDARD MODE — Calibrated production AI assistant
    # ============================================================

    async def _process_standard(self, config: ModeConfig, is_first: bool) -> Dict[str, Any]:
        """
        STANDARD mode: ONE coherent answer, ONE confidence score, ONE risk explanation.

        No duplicate summaries. No multiple narrative sections. No raw telemetry.

        Pipeline:
        1. Multi-pass pre-analysis
        2. LLM execution via Sigma
        3. Multi-pass post-analysis
        4. Boundary evaluation
        5. Confidence computation (mathematically defensible)
        6. Learning-adjusted weighting
        7. Single formatted output
        """
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-generation analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_confidence = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_confidence, "Pre-analysis baseline")

        # LLM Execution
        llm_result = None
        llm_text = ""
        if self.sigma:
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            sigma_config = SigmaV4Config(
                text=config.text,
                mode="conversational",
                enable_shadow=config.enable_shadow,
                rounds=config.rounds,
                chat_id=config.chat_id,
                history=config.history,
            )
            llm_result = await self.sigma.run_sentinel(sigma_config)
            llm_text = llm_result.data.get("priority_answer", "")
        else:
            llm_text = "[ANALYSIS-ONLY MODE: No LLM backend connected]"

        # PASS 5-9: Post-generation analysis
        post_analysis = self.reasoning.execute_post_passes(llm_text, pre_analysis)

        # Boundary Evaluation
        boundary_result = self.boundary.evaluate(
            config.text,
            context_observations=[llm_text[:500]] if llm_text else [],
            session_data=self.session.get_state_dict(),
        )
        trace.record_stage(
            "post_boundary",
            post_analysis.get("final_confidence", 0.5),
            f"Boundary severity: {boundary_result.get('severity_score', 0)}"
        )

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result=boundary_result,
        )
        session_state = self.session.get_state_dict()

        # Adaptive Learning — model weight
        model_weight = 1.0
        if config.adaptive_learning and self.knowledge_learner:
            model_weight = self.knowledge_learner.get_model_weight("groq")

        # Confidence Computation (mathematically defensible)
        confidence_components = self.confidence_engine.compute_standard(
            base_confidence=post_analysis.get("final_confidence", 0.5),
            boundary_result=boundary_result,
            session_state=session_state,
            model_weight=model_weight,
        )
        final_confidence = confidence_components.final_confidence
        trace.record_stage("final", final_confidence, "After all adjustments")

        # Risk explanation (1-2 sentences)
        risk_explanation = self._generate_risk_explanation(
            boundary_result, confidence_components, session_state
        )

        # Decomposition
        decomposition = self._build_decomposition(pre_analysis)

        # Structured Output — single coherent answer
        formatter_data = {
            "is_first_message": is_first,
            "chat_name": session_state.get("chat_name"),
            "executive_summary": llm_text[:1000] if llm_text else "No analysis generated.",
            "problem_decomposition": decomposition,
            "assumptions": pre_analysis.get("assumptions", {}),
            "logical_gaps": pre_analysis.get("gaps", {}),
            "solution": llm_text,
            "boundary": {
                "risk_level": boundary_result.get("risk_level", "LOW"),
                "severity_score": boundary_result.get("severity_score", 0),
                "explanation": risk_explanation,
            },
            "session": {
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
                "error_patterns": session_state.get("error_patterns", []),
                "reasoning_depth": session_state.get("reasoning_depth", "standard"),
            },
            "confidence": {
                "value": final_confidence,
                "explanation": risk_explanation,
            },
        }

        # Check if coding task
        intent = pre_analysis.get("intent", {})
        if intent.get("requires_code"):
            formatted = self.formatter.format_coding_task(formatter_data)
        else:
            formatted = self.formatter.format_standard(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "standard",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Standard Analysis"),
            "session_state": session_state,
            "boundary_result": boundary_result,
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_confidence,
            "llm_result": llm_result,
        }

    # ============================================================
    # DEBATE SUB-MODE — True iterative multi-agent discourse
    # ============================================================

    async def _process_debate(self, config: ModeConfig, is_first: bool) -> Dict[str, Any]:
        """
        DEBATE sub-mode: Named models (Qwen, Groq, Mistral), role assignment,
        iterative rounds with cross-model awareness, disagreement quantification.

        Pipeline:
        1. Pre-analysis
        2. DebateOrchestrator.run_debate(query, rounds, role_map)
        3. Boundary evaluation across debate positions
        4. Confidence consensus (ConfidenceEngine)
        5. Session update with debate metrics
        6. Formatted per-round per-model output
        """
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_conf, "Pre-analysis baseline")

        # True Multi-Round Debate
        debate_result = None
        if self.debate:
            role_map_str = {k: v.value if hasattr(v, 'value') else str(v)
                           for k, v in config.role_map.items()}
            debate_result = await self.debate.run_debate(
                query=config.text,
                rounds=config.rounds,
                role_map=role_map_str,
            )
        else:
            logger.warning("DebateOrchestrator not available — falling back to sigma")
            if self.sigma:
                from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
                sigma_config = SigmaV4Config(
                    text=config.text, mode="experimental",
                    enable_shadow=config.enable_shadow,
                    rounds=config.rounds, chat_id=config.chat_id,
                    history=config.history,
                )
                await self.sigma.run_sentinel(sigma_config)

        # Build analysis
        if debate_result:
            analysis_text = debate_result.analysis.synthesis if debate_result.analysis else ""
            debate_dict = debate_result.to_dict()
            debate_conf = debate_result.analysis.confidence_recalibration if debate_result.analysis else 0.5
            trace.record_stage("post_debate", debate_conf, "Debate consensus confidence")

            model_confidences = {}
            if debate_result.rounds:
                last_round = debate_result.rounds[-1]
                for mo in last_round:
                    model_confidences[mo.model_id] = mo.confidence

            disagreement = debate_result.analysis.disagreement_strength if debate_result.analysis else 0.0
            convergence = debate_result.analysis.convergence_level if debate_result.analysis else "none"
        else:
            analysis_text = ""
            debate_dict = {}
            debate_conf = 0.5
            model_confidences = {}
            disagreement = 0.0
            convergence = "none"

        # POST-PASSES on debate synthesis
        post_analysis = self.reasoning.execute_post_passes(analysis_text, pre_analysis)

        # Debate Boundary Analysis
        debate_positions = []
        if debate_result and debate_result.rounds:
            last_round = debate_result.rounds[-1]
            for mo in last_round:
                debate_positions.append({
                    "model": mo.model_label,
                    "position": mo.position,
                    "key_points": [mo.argument[:200]] if mo.argument else [],
                })

        debate_boundary = self.boundary.evaluate_debate_boundaries(
            model_positions=debate_positions,
            agreements=[],
            disagreements=debate_result.analysis.conflict_axes if debate_result and debate_result.analysis else [],
        )
        boundary_severity = debate_boundary.get("compound_severity", 0)
        trace.record_stage(
            "post_boundary", debate_conf,
            f"Debate boundary severity: {boundary_severity}"
        )

        # Confidence consensus (ConfidenceEngine)
        model_weights = {}
        if self.knowledge_learner and config.adaptive_learning:
            model_weights = self.knowledge_learner.get_all_model_weights()

        consensus_components = self.confidence_engine.compute_debate_consensus(
            model_confidences=model_confidences,
            disagreement_score=disagreement,
            convergence_level=convergence,
            boundary_severity=boundary_severity,
            model_weights=model_weights,
        )
        final_conf = consensus_components.final_confidence
        trace.record_stage("final", final_conf, "Consensus confidence after all adjustments")

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result={
                "severity_score": boundary_severity,
                "risk_level": debate_boundary.get("risk_level", "LOW"),
                "claim_type": "debate_synthesis",
                "explanation": debate_boundary.get("explanation", ""),
            },
            disagreement_data={
                "disagreements": debate_result.analysis.conflict_axes if debate_result and debate_result.analysis else [],
                "agreements": [],
            },
        )
        session_state = self.session.get_state_dict()

        # Record debate in session history graph
        if debate_result and debate_result.analysis:
            self.session.record_debate_round({
                "disagreement_strength": disagreement,
                "convergence_level": convergence,
                "confidence_recalibration": debate_conf,
                "models_used": debate_result.models_used,
            })

        # Format output
        formatter_data = {
            "session": {
                "primary_goal": session_state.get("primary_goal", "Analysis"),
                "inferred_domain": session_state.get("inferred_domain", "General"),
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
            },
            "debate": debate_dict,
            "boundary": {
                "detected_count": len(debate_boundary.get("per_model_boundaries", [])),
                "severity_score": boundary_severity,
                "explanation": debate_boundary.get("explanation", ""),
            },
            "confidence_evolution": trace.to_dict(),
            "fragility": {
                "score": session_state.get("fragility_index", 0.0),
                "explanation": self._fragility_explanation(session_state.get("fragility_index", 0.0)),
            },
            "synthesis": debate_result.analysis.synthesis if debate_result and debate_result.analysis else analysis_text,
        }
        formatted = self.formatter.format_debate(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "research",
            "sub_mode": "debate",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Adversarial Debate"),
            "session_state": session_state,
            "boundary_result": debate_boundary,
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_components": consensus_components.to_dict(),
            "confidence_evolution": trace.to_dict(),
            "fragility_index": session_state.get("fragility_index", 0.0),
            "debate_result": debate_dict,
            "llm_result": None,
        }

    # ============================================================
    # GLASS SUB-MODE — Interpretive, not raw JSON
    # ============================================================

    async def _process_glass(self, config: ModeConfig, is_first: bool = False) -> Dict[str, Any]:
        """
        GLASS sub-mode: Interpretive analysis console.

        Exposes: confidence pipeline, boundary reasoning, disagreement analysis,
        fragility score, behavioral risk profile, session intelligence metrics.
        Every metric includes interpretation.

        When config.kill_override=True: KILL diagnostic (inside GLASS only).
          - Disables learning adjustments
          - Disables boundary penalties
          - Disables fragility penalties
          - Shows raw model outputs and base confidence
          - Does NOT modify state
        """
        # --- KILL OVERRIDE (diagnostic inside Glass) ---
        if config.is_kill:
            return await self._process_kill_diagnostic(config)

        # --- GLASS ANALYSIS ---
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_conf, "Pre-analysis baseline")

        # LLM Execution
        llm_result = None
        llm_text = ""
        if self.sigma:
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            sigma_config = SigmaV4Config(
                text=config.text, mode="experimental",
                enable_shadow=config.enable_shadow,
                rounds=config.rounds, chat_id=config.chat_id,
                history=config.history,
            )
            llm_result = await self.sigma.run_sentinel(sigma_config)
            llm_text = llm_result.data.get("priority_answer", "")
        else:
            llm_text = "[ANALYSIS-ONLY MODE]"

        # PASS 5-9: Post-analysis
        post_analysis = self.reasoning.execute_post_passes(llm_text, pre_analysis)

        # Boundary Evaluation
        boundary_result = self.boundary.evaluate(
            config.text,
            context_observations=[llm_text[:500]] if llm_text else [],
            session_data=self.session.get_state_dict(),
        )
        trace.record_stage(
            "post_boundary",
            post_analysis.get("final_confidence", 0.5),
            f"Boundary severity: {boundary_result.get('severity_score', 0)}"
        )

        # Behavioral Analytics (full profile)
        behavioral_profile = self.behavioral.analyze(
            text=config.text,
            llm_output=llm_text,
            shadow_result=None,
            context=self.session.get_state_dict(),
        )

        # Update session with behavioral risk
        self.session.update_behavioral_risk(behavioral_profile.to_dict())

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result=boundary_result,
        )
        session_state = self.session.get_state_dict()

        # Model weight (adaptive learning)
        model_weight = 1.0
        if config.adaptive_learning and self.knowledge_learner:
            model_weight = self.knowledge_learner.get_model_weight("groq")

        # Confidence Computation (full component visibility)
        confidence_components = self.confidence_engine.compute_standard(
            base_confidence=post_analysis.get("final_confidence", 0.5),
            boundary_result=boundary_result,
            session_state=session_state,
            model_weight=model_weight,
        )
        final_conf = confidence_components.final_confidence
        trace.record_stage("final", final_conf, "After all adjustments")

        # Format glass output (every metric includes interpretation)
        formatter_data = {
            "session": {
                "primary_goal": session_state.get("primary_goal", "Analysis"),
                "inferred_domain": session_state.get("inferred_domain", "General"),
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
            },
            "boundary": {
                "risk_level": boundary_result.get("risk_level", "LOW"),
                "severity_score": boundary_result.get("severity_score", 0),
                "explanation": boundary_result.get("explanation", ""),
            },
            "behavioral_risk": behavioral_profile.to_dict(),
            "confidence_evolution": trace.to_dict(),
            "confidence_components": confidence_components.to_dict(),
            "fragility": {
                "score": session_state.get("fragility_index", 0.0),
                "explanation": self._fragility_explanation(session_state.get("fragility_index", 0.0)),
                "formula": "0.4 * disagreement + 0.3 * boundary_severity + 0.2 * evidence_contradictions + 0.1 * confidence_volatility",
            },
            "session_intelligence": {
                "cognitive_drift": session_state.get("cognitive_drift_score", 0.0),
                "topic_stability": session_state.get("topic_stability_score", 1.0),
                "confidence_volatility": session_state.get("confidence_volatility_index", 0.0),
                "behavioral_risk_accumulator": session_state.get("behavioral_risk_accumulator", 0.0),
                "model_reliability_scores": session_state.get("model_reliability_scores", {}),
            },
            "severity_trend": self.boundary.get_trend(),
            "synthesis": llm_text,
        }
        formatted = self.formatter.format_glass(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "research",
            "sub_mode": "glass",
            "kill_active": False,
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Glass Analysis"),
            "session_state": session_state,
            "boundary_result": boundary_result,
            "behavioral_risk": behavioral_profile.to_dict(),
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_components": confidence_components.to_dict(),
            "confidence_evolution": trace.to_dict(),
            "fragility_index": session_state.get("fragility_index", 0.0),
            "llm_result": llm_result,
        }

    async def _process_kill_diagnostic(self, config: ModeConfig) -> Dict[str, Any]:
        """
        KILL diagnostic state (inside GLASS only).

        - Disables learning adjustments
        - Disables boundary penalties
        - Disables fragility penalties
        - Shows raw model outputs and raw base confidence
        - Does NOT modify state
        """
        diagnostic = self.session.get_kill_diagnostic()
        boundary_trend = self.boundary.get_trend()

        raw_session_conf = diagnostic.get("session_confidence", 0.5)

        kill_data = {
            "session_state": {
                "chat_name": diagnostic.get("chat_name", "Unknown"),
                "primary_goal": diagnostic.get("primary_goal", "Unknown"),
                "inferred_domain": diagnostic.get("inferred_domain", "Unknown"),
                "user_expertise_score": diagnostic.get("user_expertise_score", 0.0),
            },
            "boundary_snapshot": {
                "latest_severity": diagnostic.get("latest_boundary_severity", 0),
                "trend": boundary_trend.get("trend", "stable"),
            },
            "raw_base_confidence": raw_session_conf,
            "note": "KILL diagnostic: learning adjustments, boundary penalties, and fragility penalties DISABLED. Showing raw values.",
            "disagreement_score": diagnostic.get("disagreement_score", 0.0),
            "fragility_index": diagnostic.get("fragility_index", 0.0),
            "error_patterns": diagnostic.get("recurring_error_patterns", []),
            "session_confidence": raw_session_conf,
            "confidence_explanation": diagnostic.get("session_confidence_explanation", ""),
            "behavioral_risk_profile": self.session.state.behavioral_risk_profile,
            "session_intelligence": {
                "cognitive_drift": self.session.state.cognitive_drift_score,
                "topic_stability": self.session.state.topic_stability_score,
                "confidence_volatility": self.session.state.confidence_volatility_index,
                "model_reliability_scores": self.session.state.model_reliability_scores,
            },
        }

        formatted = self.formatter.format_kill(kill_data)

        return {
            "formatted_output": formatted,
            "mode": "research",
            "sub_mode": "glass",
            "kill_active": True,
            "chat_id": config.chat_id,
            "chat_name": diagnostic.get("chat_name", "Glass Kill Diagnostic"),
            "session_state": self.session.get_state_dict(),
            "diagnostic": diagnostic,
            "confidence": raw_session_conf,
            "llm_result": None,
        }

    # ============================================================
    # EVIDENCE SUB-MODE — Research-grade evidence analysis
    # ============================================================

    async def _process_evidence(self, config: ModeConfig, is_first: bool) -> Dict[str, Any]:
        """
        EVIDENCE sub-mode: Research-grade evidence analysis.

        Pipeline:
        1. Pre-analysis
        2. LLM execution (for answer generation)
        3. Full evidence analysis (claim extraction, source mapping, traceability)
        4. Confidence adjusted by evidence strength
        5. Formatted output
        """
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_conf, "Pre-analysis baseline")

        # LLM Execution
        llm_result = None
        llm_text = ""
        if self.sigma:
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            sigma_config = SigmaV4Config(
                text=config.text, mode="experimental",
                enable_shadow=config.enable_shadow,
                rounds=config.rounds, chat_id=config.chat_id,
                history=config.history,
            )
            llm_result = await self.sigma.run_sentinel(sigma_config)
            llm_text = llm_result.data.get("priority_answer", "")
        else:
            llm_text = "[ANALYSIS-ONLY MODE]"

        # Full Evidence Analysis (3.X pipeline)
        evidence_result = await self.evidence.run_full_evidence_analysis(
            query=config.text,
            answer=llm_text,
            max_results=5,
        )
        trace.record_stage(
            "post_evidence",
            evidence_result.evidence_confidence,
            f"Evidence: {len(evidence_result.sources)} sources, "
            f"{len(evidence_result.contradictions)} contradictions"
        )

        # POST-PASSES
        post_analysis = self.reasoning.execute_post_passes(llm_text, pre_analysis)

        # Boundary Evaluation
        boundary_result = self.boundary.evaluate(
            config.text,
            context_observations=[llm_text[:500]] if llm_text else [],
            session_data=self.session.get_state_dict(),
        )
        trace.record_stage(
            "post_boundary",
            post_analysis.get("final_confidence", 0.5),
            f"Boundary severity: {boundary_result.get('severity_score', 0)}"
        )

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result=boundary_result,
        )
        session_state = self.session.get_state_dict()

        # Model weight
        model_weight = 1.0
        if config.adaptive_learning and self.knowledge_learner:
            model_weight = self.knowledge_learner.get_model_weight("groq")

        # Base confidence
        base_components = self.confidence_engine.compute_standard(
            base_confidence=post_analysis.get("final_confidence", 0.5),
            boundary_result=boundary_result,
            session_state=session_state,
            model_weight=model_weight,
        )

        # Evidence-adjusted confidence
        evidence_components = self.confidence_engine.compute_evidence_adjusted(
            base_components=base_components,
            evidence_result=evidence_result.to_dict(),
        )
        final_conf = evidence_components.final_confidence
        trace.record_stage("final", final_conf, "After evidence adjustment")

        # Format output
        formatter_data = {
            "session": {
                "primary_goal": session_state.get("primary_goal", "Analysis"),
                "inferred_domain": session_state.get("inferred_domain", "General"),
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
            },
            "evidence": evidence_result.to_dict(),
            "boundary": {
                "risk_level": boundary_result.get("risk_level", "LOW"),
                "severity_score": boundary_result.get("severity_score", 0),
                "explanation": boundary_result.get("explanation", ""),
            },
            "confidence_evolution": trace.to_dict(),
            "synthesis": llm_text,
        }
        formatted = self.formatter.format_evidence(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "research",
            "sub_mode": "evidence",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Evidence Analysis"),
            "session_state": session_state,
            "boundary_result": boundary_result,
            "evidence_result": evidence_result.to_dict(),
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_components": evidence_components.to_dict(),
            "confidence_evolution": trace.to_dict(),
            "fragility_index": session_state.get("fragility_index", 0.0),
            "llm_result": llm_result,
        }

    # ============================================================
    # STRESS SUB-MODE — Adversarial answer stress testing
    # ============================================================

    async def _process_stress(self, config: ModeConfig, is_first: bool) -> Dict[str, Any]:
        """
        STRESS sub-mode: Attempts to break the answer.

        Simulates:
        - Extreme counterexamples
        - Adversarial prompts
        - Logical inversions
        - Boundary amplification

        Returns stability metrics and revised confidence.
        Stress affects fragility.
        """
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_conf, "Pre-analysis baseline")

        # LLM Execution (generate answer to stress-test)
        llm_result = None
        llm_text = ""
        if self.sigma:
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            sigma_config = SigmaV4Config(
                text=config.text, mode="experimental",
                enable_shadow=config.enable_shadow,
                rounds=config.rounds, chat_id=config.chat_id,
                history=config.history,
            )
            llm_result = await self.sigma.run_sentinel(sigma_config)
            llm_text = llm_result.data.get("priority_answer", "")
        else:
            llm_text = "[ANALYSIS-ONLY MODE]"

        # POST-PASSES
        post_analysis = self.reasoning.execute_post_passes(llm_text, pre_analysis)

        # Boundary Evaluation
        boundary_result = self.boundary.evaluate(
            config.text,
            context_observations=[llm_text[:500]] if llm_text else [],
            session_data=self.session.get_state_dict(),
        )
        boundary_severity = boundary_result.get("severity_score", 0)
        base_conf = post_analysis.get("final_confidence", 0.5)
        trace.record_stage("post_boundary", base_conf, f"Boundary severity: {boundary_severity}")

        # STRESS TESTING
        stress_result = None
        if self.stress:
            stress_result = await self.stress.run_stress_test(
                query=config.text,
                answer=llm_text,
                boundary_severity=boundary_severity,
                base_confidence=base_conf,
            )
            trace.record_stage(
                "post_stress",
                stress_result.revised_confidence,
                f"Stability: {stress_result.stability_after_stress:.2f}, "
                f"Contradictions: {stress_result.contradictions_found}"
            )
        else:
            logger.warning("StressEngine not available")

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result=boundary_result,
        )
        session_state = self.session.get_state_dict()

        # Confidence after stress
        model_weight = 1.0
        if config.adaptive_learning and self.knowledge_learner:
            model_weight = self.knowledge_learner.get_model_weight("groq")

        base_components = self.confidence_engine.compute_standard(
            base_confidence=base_conf,
            boundary_result=boundary_result,
            session_state=session_state,
            model_weight=model_weight,
        )

        if stress_result:
            stress_components = self.confidence_engine.compute_stress_adjusted(
                base_components=base_components,
                stress_result=stress_result.to_dict(),
            )
            final_conf = stress_components.final_confidence
        else:
            stress_components = base_components
            final_conf = base_components.final_confidence

        trace.record_stage("final", final_conf, "After stress adjustment")

        # Format stress output
        stress_dict = stress_result.to_dict() if stress_result else {}
        formatter_data = {
            "session": {
                "primary_goal": session_state.get("primary_goal", "Analysis"),
                "inferred_domain": session_state.get("inferred_domain", "General"),
            },
            "stress": stress_dict,
            "boundary": {
                "risk_level": boundary_result.get("risk_level", "LOW"),
                "severity_score": boundary_severity,
                "explanation": boundary_result.get("explanation", ""),
            },
            "confidence_evolution": trace.to_dict(),
            "confidence_components": stress_components.to_dict(),
            "fragility": {
                "score": session_state.get("fragility_index", 0.0),
                "explanation": self._fragility_explanation(session_state.get("fragility_index", 0.0)),
            },
            "synthesis": llm_text,
        }
        formatted = self.formatter.format_stress(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "research",
            "sub_mode": "stress",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Stress Test"),
            "session_state": session_state,
            "boundary_result": boundary_result,
            "stress_result": stress_dict,
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_components": stress_components.to_dict(),
            "confidence_evolution": trace.to_dict(),
            "fragility_index": session_state.get("fragility_index", 0.0),
            "llm_result": llm_result,
        }

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _generate_risk_explanation(
        self,
        boundary_result: Dict,
        components: Any,
        session_state: Dict,
    ) -> str:
        """Generate a 1-2 sentence risk explanation for STANDARD mode."""
        severity = boundary_result.get("severity_score", 0)
        risk_level = boundary_result.get("risk_level", "LOW")
        conf = components.final_confidence if hasattr(components, 'final_confidence') else 0.5

        if risk_level == "LOW" and conf >= 0.75:
            return "Low risk. Analysis is well-grounded with high confidence."
        elif risk_level == "LOW":
            return f"Low boundary risk, but confidence is moderate ({conf:.0%}) due to domain uncertainty or limited evidence."
        elif risk_level == "MEDIUM":
            return f"Moderate risk detected (severity {severity}/100). Some claims may lack full grounding. Treat as advisory."
        elif risk_level == "HIGH":
            return f"Elevated risk (severity {severity}/100). Multiple boundary signals active. Verify independently before acting."
        else:
            return f"Critical risk (severity {severity}/100). Strong boundary violations detected. Do not rely on this output without verification."

    def _build_decomposition(self, pre_analysis: Dict) -> List[Dict[str, str]]:
        """Build problem decomposition from pre-analysis."""
        components = []
        intent = pre_analysis.get("intent", {})

        components.append({
            "component": "Intent Classification",
            "description": f"Type: {intent.get('intent_type', 'unknown')}, Complexity: {intent.get('complexity', 'medium')}",
        })

        gaps = pre_analysis.get("gaps", {}).get("gaps", [])
        if gaps:
            components.append({
                "component": "Logical Gap Resolution",
                "description": f"{len(gaps)} gap(s) requiring attention: {gaps[0].get('description', '')}" if gaps else "None",
            })

        assumptions = pre_analysis.get("assumptions", {})
        total = assumptions.get("total_count", 0)
        if total > 0:
            components.append({
                "component": "Assumption Validation",
                "description": f"{total} assumption(s) to validate or make explicit",
            })

        boundary = pre_analysis.get("boundary", {})
        if boundary.get("high_risk_domains_detected"):
            components.append({
                "component": "High-Risk Domain Handling",
                "description": "Requires explicit caveats and domain-appropriate disclaimers",
            })

        return components

    def _confidence_explanation(self, post_analysis: Dict) -> str:
        """Generate confidence explanation from post-analysis."""
        conf = post_analysis.get("final_confidence", 0.5)
        trace = post_analysis.get("trace_summary", {})

        parts = []
        if conf >= 0.85:
            parts.append("High confidence.")
        elif conf >= 0.65:
            parts.append("Moderate confidence.")
        elif conf >= 0.4:
            parts.append("Reduced confidence.")
        else:
            parts.append("Low confidence.")

        gaps = trace.get("logical_gaps_detected", 0)
        if gaps:
            parts.append(f"{gaps} logical gap(s) detected.")

        if trace.get("self_critique_applied"):
            parts.append("Self-critique applied.")

        if trace.get("boundary_severity", 0) > 40:
            parts.append("Boundary severity elevated.")

        return " ".join(parts)

    def _fragility_explanation(self, score: float) -> str:
        """Generate fragility explanation."""
        if score >= 0.7:
            return "High fragility. Multiple destabilizing factors active. Conclusions may be unreliable."
        elif score >= 0.4:
            return "Moderate fragility. Some risk factors present. Conclusions require validation."
        elif score >= 0.2:
            return "Low fragility. Minor risk factors. Conclusions are reasonably stable."
        return "Minimal fragility. Session is stable."

    # ============================================================
    # PUBLIC ACCESSORS
    # ============================================================

    def get_session_state(self) -> Dict[str, Any]:
        """Get current session intelligence state."""
        return self.session.get_state_dict()

    def get_boundary_trend(self) -> Dict[str, Any]:
        """Get boundary severity trend."""
        return self.boundary.get_trend()

    def is_initialized(self) -> bool:
        """Check if the kernel has processed at least one message."""
        return self._initialized
