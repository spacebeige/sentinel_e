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

from core.mode_config import ModeConfig, Mode, SubMode
from core.session_intelligence import SessionIntelligence
from core.multipass_reasoning import MultiPassReasoningEngine
from core.omega_boundary import OmegaBoundaryEvaluator
from core.omega_formatter import OmegaOutputFormatter
from core.behavioral_analytics import BehavioralAnalyzer
from core.evidence_engine import EvidenceEngine
from core.debate_orchestrator import DebateOrchestrator
from core.confidence_engine import ConfidenceEngine, ConfidenceTrace
from core.stress_engine import StressEngine

# v4 Engine Modules — Production Cognitive Governance
from engines.aggregation_engine import AggregationEngine
from engines.forensic_evidence_engine import ForensicEvidenceEngine
from engines.blind_audit_engine import BlindAuditEngine
from engines.dynamic_boundary import DynamicBoundaryEngine
from engines.mode_controller import resolve_execution_mode, detect_evidence_trigger

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
        self.aggregation = None
        self.forensic = None
        self.audit = None
        self._initialized = False

        # Initialize model-dependent engines
        if self.sigma and hasattr(self.sigma, 'client'):
            self.debate = DebateOrchestrator(self.sigma.client)
            self.stress = StressEngine(self.sigma.client)
            # v4 Production Engines
            self.aggregation = AggregationEngine(self.sigma.client)
            self.forensic = ForensicEvidenceEngine(self.sigma.client)
            self.audit = BlindAuditEngine(self.sigma.client)

        logger.info("Omega Cognitive Kernel v4.0 initialized — Production engines active.")

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

        # 1b. Trigger Word Detection — Standard mode → Evidence override (Section 7)
        if config.is_standard and detect_evidence_trigger(config.text):
            logger.info(
                f"Evidence trigger detected in Standard mode — routing to Evidence. "
                f"Input: '{config.text[:60]}...'"
            )
            config = ModeConfig(
                text=config.text,
                mode=Mode.RESEARCH,
                sub_mode=SubMode.EVIDENCE,
                chat_id=config.chat_id,
                history=config.history,
                rounds=config.rounds,
                enable_shadow=config.enable_shadow,
                adaptive_learning=config.adaptive_learning,
            )

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
        STANDARD mode v4: TRUE parallel 3-model aggregation.

        Pipeline:
        1. Multi-pass pre-analysis
        2. Parallel execution: Groq + Llama 3.3 70B + Qwen via AggregationEngine
        3. Divergence computation + synthesis generation
        4. Multi-pass post-analysis
        5. Dynamic boundary evaluation (computed, not hardcoded)
        6. Confidence aggregation (mathematically defensible)
        7. Single formatted output with cross-model data
        
        NO debate. NO sequential single-model calls. NO stale state.
        """
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-generation analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_confidence = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_confidence, "Pre-analysis baseline")

        # PARALLEL 3-MODEL AGGREGATION (v4 engine)
        aggregation_result = None
        llm_text = ""
        aggregation_dict = {}

        if self.aggregation:
            aggregation_result = await self.aggregation.run_parallel_aggregation(
                query=config.text, history=config.history or []
            )
            llm_text = aggregation_result.synthesis
            aggregation_dict = aggregation_result.to_dict()
            
            trace.record_stage(
                "post_aggregation",
                aggregation_result.confidence_aggregation,
                f"Parallel aggregation: {aggregation_result.models_succeeded}/3 succeeded, "
                f"divergence={aggregation_result.divergence_score:.3f}"
            )
        elif self.sigma:
            # Fallback: legacy single-model if engine not available
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            sigma_config = SigmaV4Config(
                text=config.text, mode="conversational",
                enable_shadow=config.enable_shadow,
                rounds=config.rounds, chat_id=config.chat_id,
                history=config.history,
            )
            llm_result = await self.sigma.run_sentinel(sigma_config)
            llm_text = llm_result.data.get("priority_answer", "")
        else:
            llm_text = "[ANALYSIS-ONLY MODE: No LLM backend connected]"

        # PASS 5-9: Post-generation analysis
        post_analysis = self.reasoning.execute_post_passes(llm_text, pre_analysis)

        # DYNAMIC BOUNDARY (v4 — computed, not hardcoded)
        if aggregation_result:
            boundary_data = DynamicBoundaryEngine.compute_severity(
                evidence_confidence=aggregation_result.confidence_aggregation,
                model_divergence=aggregation_result.divergence_score,
                model_confidences=list(aggregation_result.confidence_per_model.values()),
                failed_models=aggregation_result.models_failed,
            )
        else:
            boundary_data = self.boundary.evaluate(
                config.text,
                context_observations=[llm_text[:500]] if llm_text else [],
                session_data=self.session.get_state_dict(),
            )

        trace.record_stage(
            "post_boundary",
            post_analysis.get("final_confidence", 0.5),
            f"Boundary severity: {boundary_data.get('severity_score', 0)}"
        )

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result=boundary_data,
        )
        session_state = self.session.get_state_dict()

        # Adaptive Learning — model weight
        model_weight = 1.0
        if config.adaptive_learning and self.knowledge_learner:
            model_weight = self.knowledge_learner.get_model_weight("groq")

        # Confidence (use aggregation confidence if available, else legacy)
        if aggregation_result:
            final_confidence = aggregation_result.confidence_aggregation
            # Apply boundary penalty
            boundary_penalty = boundary_data.get("severity_score", 0) / 200.0
            final_confidence = max(0.05, min(0.95, final_confidence - boundary_penalty))
        else:
            confidence_components = self.confidence_engine.compute_standard(
                base_confidence=post_analysis.get("final_confidence", 0.5),
                boundary_result=boundary_data,
                session_state=session_state,
                model_weight=model_weight,
            )
            final_confidence = confidence_components.final_confidence

        trace.record_stage("final", final_confidence, "After all adjustments")

        # Risk explanation
        risk_explanation = self._generate_risk_explanation(
            boundary_data,
            None,  # confidence_components may not exist
            session_state
        )

        # Decomposition
        decomposition = self._build_decomposition(pre_analysis)

        # Structured Output
        formatter_data = {
            "is_first_message": is_first,
            "chat_name": session_state.get("chat_name"),
            "executive_summary": llm_text[:1000] if llm_text else "No analysis generated.",
            "problem_decomposition": decomposition,
            "assumptions": pre_analysis.get("assumptions", {}),
            "logical_gaps": pre_analysis.get("gaps", {}),
            "solution": llm_text,
            "boundary": {
                "risk_level": boundary_data.get("risk_level", "LOW"),
                "severity_score": boundary_data.get("severity_score", 0),
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

        # Build omega_metadata with aggregation data for frontend
        omega_metadata = {}
        if aggregation_result:
            omega_metadata["aggregation_result"] = aggregation_dict
            omega_metadata["pipeline_steps"] = [
                {"step": "parallel_execution", "label": "Running 3 Models in Parallel", "status": "complete"},
                {"step": "divergence", "label": "Computing Cross-Model Divergence", "status": "complete"},
                {"step": "synthesis", "label": "Generating Synthesis", "status": "complete"},
                {"step": "boundary", "label": "Computing Dynamic Boundary", "status": "complete"},
            ]

        return {
            "formatted_output": formatted,
            "mode": "standard",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Standard Analysis"),
            "session_state": session_state,
            "boundary_result": boundary_data,
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_confidence,
            "omega_metadata": omega_metadata,
            "llm_result": None,
        }

    # ============================================================
    # DEBATE SUB-MODE — True iterative multi-agent discourse
    # ============================================================

    async def _process_debate(self, config: ModeConfig, is_first: bool) -> Dict[str, Any]:
        """
        DEBATE sub-mode: Named models (Qwen, Groq, Llama 3.3 70B), role assignment,
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
        GLASS sub-mode v4: Blind Cross-Model Forensic Audit.

        Steps:
        1. Each model generates independent response
        2. Each model blindly evaluates another's output (no identity revealed)
        3. Extract: coherence, hidden assumptions, bias, confidence inflation, persuasion
        4. Compute cross-model tactical map
        5. Dynamic boundary + trust scoring
        6. Formatted output with per-model forensic assessments

        When config.kill_override=True: KILL diagnostic (unchanged — inside GLASS only).
        """
        # --- KILL OVERRIDE (diagnostic inside Glass, unchanged) ---
        if config.is_kill:
            return await self._process_kill_diagnostic(config)

        # --- BLIND AUDIT (v4 engine) ---
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_conf, "Pre-analysis baseline")

        # BLIND CROSS-MODEL AUDIT
        audit_result = None
        audit_dict = {}
        llm_text = ""

        if self.audit:
            audit_result = await self.audit.run_blind_audit(
                query=config.text, history=config.history or []
            )
            audit_dict = audit_result.to_dict()
            
            # Build structured output from audit
            llm_text = "## 🧠 Blind Forensic Audit Results\n\n"
            
            for assessment in audit_result.assessments:
                llm_text += f"### 🧠 {assessment.auditor_name} Forensic Assessment\n"
                llm_text += f"**Subject**: {assessment.subject_name}\n\n"
                llm_text += f"| Metric | Score |\n|---|---|\n"
                llm_text += f"| Logical Coherence | {assessment.logical_coherence:.0%} |\n"
                llm_text += f"| Hidden Assumptions | {assessment.hidden_assumptions:.0%} |\n"
                llm_text += f"| Bias Patterns | {assessment.bias_patterns:.0%} |\n"
                llm_text += f"| Confidence Inflation | {assessment.confidence_inflation:.0%} |\n"
                llm_text += f"| Persuasion Tactics | {assessment.persuasion_tactics:.0%} |\n"
                llm_text += f"| Evidence Quality | {assessment.evidence_quality:.0%} |\n"
                llm_text += f"| Trust Score | {assessment.trust_score:.0%} |\n\n"
                
                if assessment.weak_points:
                    llm_text += f"**Weak Points**: {'; '.join(assessment.weak_points[:3])}\n\n"
                if assessment.risk_factors:
                    llm_text += f"**Risk Factors**: {'; '.join(assessment.risk_factors[:3])}\n\n"

            # Tactical map
            if audit_result.tactical_map.get("model_profiles"):
                llm_text += "### Cross-Model Tactical Map\n\n"
                for model_id, profile in audit_result.tactical_map["model_profiles"].items():
                    llm_text += f"**{profile.get('model_name', model_id)}**: "
                    llm_text += f"Trust={profile.get('avg_trust', 0):.0%}, "
                    llm_text += f"Bias={profile.get('avg_bias', 0):.0%}, "
                    llm_text += f"Inflation={profile.get('avg_confidence_inflation', 0):.0%}\n"

                if audit_result.tactical_map.get("highest_risk_model"):
                    llm_text += f"\n⚠️ **Highest Risk**: {audit_result.tactical_map['highest_risk_model']}\n"
                if audit_result.tactical_map.get("most_trustworthy_model"):
                    llm_text += f"✅ **Most Trustworthy**: {audit_result.tactical_map['most_trustworthy_model']}\n"

            trace.record_stage(
                "post_audit",
                audit_result.overall_trust,
                f"Blind audit: {len(audit_result.assessments)} assessments, "
                f"trust={audit_result.overall_trust:.3f}, risk={audit_result.consensus_risk}"
            )
        else:
            # Fallback: legacy glass behavior
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

        # POST-PASSES
        post_analysis = self.reasoning.execute_post_passes(llm_text, pre_analysis)

        # DYNAMIC BOUNDARY (v4)
        if audit_result:
            boundary_data = DynamicBoundaryEngine.compute_severity(
                evidence_confidence=audit_result.overall_trust,
                model_divergence=1.0 - audit_result.overall_trust,
                model_confidences=[a.trust_score for a in audit_result.assessments],
            )
        else:
            boundary_data = self.boundary.evaluate(
                config.text,
                context_observations=[llm_text[:500]] if llm_text else [],
                session_data=self.session.get_state_dict(),
            )

        trace.record_stage(
            "post_boundary",
            post_analysis.get("final_confidence", 0.5),
            f"Boundary severity: {boundary_data.get('severity_score', 0)}"
        )

        # Behavioral Analytics
        behavioral_profile = self.behavioral.analyze(
            text=config.text,
            llm_output=llm_text,
            shadow_result=None,
            context=self.session.get_state_dict(),
        )
        self.session.update_behavioral_risk(behavioral_profile.to_dict())

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result=boundary_data,
        )
        session_state = self.session.get_state_dict()

        # Confidence
        if audit_result:
            final_conf = audit_result.overall_trust
            boundary_penalty = boundary_data.get("severity_score", 0) / 200.0
            final_conf = max(0.05, min(0.95, final_conf - boundary_penalty))
        else:
            model_weight = 1.0
            if config.adaptive_learning and self.knowledge_learner:
                model_weight = self.knowledge_learner.get_model_weight("groq")
            confidence_components = self.confidence_engine.compute_standard(
                base_confidence=post_analysis.get("final_confidence", 0.5),
                boundary_result=boundary_data,
                session_state=session_state,
                model_weight=model_weight,
            )
            final_conf = confidence_components.final_confidence

        trace.record_stage("final", final_conf, "After all adjustments")

        # Format glass output
        formatter_data = {
            "session": {
                "primary_goal": session_state.get("primary_goal", "Analysis"),
                "inferred_domain": session_state.get("inferred_domain", "General"),
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
            },
            "boundary": {
                "risk_level": boundary_data.get("risk_level", "LOW"),
                "severity_score": boundary_data.get("severity_score", 0),
                "explanation": boundary_data.get("explanation", ""),
            },
            "behavioral_risk": behavioral_profile.to_dict(),
            "confidence_evolution": trace.to_dict(),
            "fragility": {
                "score": session_state.get("fragility_index", 0.0),
                "explanation": self._fragility_explanation(session_state.get("fragility_index", 0.0)),
            },
            "severity_trend": self.boundary.get_trend(),
            "synthesis": llm_text,
        }
        formatted = self.formatter.format_glass(formatter_data)

        # Build omega_metadata with audit data
        omega_metadata = {
            "audit_result": audit_dict,
            "pipeline_steps": [],
        }
        if audit_result:
            omega_metadata["pipeline_steps"] = [
                {"step": p.get("name", ""), "label": p.get("name", ""), "status": p.get("status", "")}
                for p in audit_result.phase_log
            ]

        return {
            "formatted_output": formatted,
            "mode": "research",
            "sub_mode": "glass",
            "kill_active": False,
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Blind Forensic Audit"),
            "session_state": session_state,
            "boundary_result": boundary_data,
            "behavioral_risk": behavioral_profile.to_dict(),
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_evolution": trace.to_dict(),
            "fragility_index": session_state.get("fragility_index", 0.0),
            "omega_metadata": omega_metadata,
            "llm_result": None,
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
        EVIDENCE sub-mode v4: 5-Phase Triangular Forensic Pipeline.

        Pipeline:
        1. Phase 1: Independent retrieval (3 models extract claims independently)
        2. Phase 2: Triangular cross-verification (blind, forensic framing)
        3. Phase 3: Algorithmic contradiction detection (deterministic)
        4. Phase 4: Bayesian confidence update (computed, not fixed)
        5. Phase 5: Verbatim citation mode (if triggered)
        6. Dynamic boundary evaluation
        7. Formatted output with structured evidence data
        """
        trace = ConfidenceTrace()

        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode.value, config.history
        )
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        trace.record_stage("initial", initial_conf, "Pre-analysis baseline")

        # 5-PHASE FORENSIC PIPELINE (v4 engine)
        forensic_result = None
        forensic_dict = {}
        llm_text = ""

        if self.forensic:
            forensic_result = await self.forensic.run_forensic_pipeline(
                query=config.text, history=config.history or []
            )
            forensic_dict = forensic_result.to_dict()
            
            # Build synthesis from claims
            if forensic_result.all_claims:
                top_claims = sorted(
                    forensic_result.all_claims,
                    key=lambda c: c.final_confidence,
                    reverse=True
                )[:5]
                llm_text = "## Forensic Evidence Analysis\n\n"
                for i, claim in enumerate(top_claims, 1):
                    conf_pct = claim.final_confidence * 100
                    llm_text += (
                        f"**{i}. [{claim.model_origin.upper()}]** {claim.statement}\n"
                        f"   - Confidence: {conf_pct:.1f}% | "
                        f"Verified by {claim.agreement_count} model(s) | "
                        f"{claim.contradiction_count} contradiction(s)\n\n"
                    )
                
                # Add verbatim citations if available
                if forensic_result.verbatim_citations:
                    llm_text += "\n## Verbatim Citations\n\n"
                    for cit in forensic_result.verbatim_citations[:5]:
                        llm_text += f"> \"{cit.get('quote', 'N/A')}\"\n"
                        llm_text += f"Source: {cit.get('source', 'Unknown')} "
                        llm_text += f"(Reliability: {cit.get('reliability', 0):.0%})\n\n"

                # Add contradiction report
                if forensic_result.contradictions:
                    llm_text += "\n## ⚠️ Contradictions Detected\n\n"
                    for c in forensic_result.contradictions[:5]:
                        llm_text += (
                            f"- **{c['type']}** between {c['model_a']} and {c['model_b']}: "
                            f"Severity {c['severity']:.0%}\n"
                        )

            trace.record_stage(
                "post_forensic",
                forensic_result.bayesian_confidence,
                f"Forensic: {forensic_result.to_dict().get('total_claims', 0)} claims, "
                f"{len(forensic_result.contradictions)} contradictions, "
                f"Bayesian confidence: {forensic_result.bayesian_confidence:.3f}"
            )
        else:
            # Fallback: legacy evidence engine
            llm_result = None
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

            evidence_result = await self.evidence.run_full_evidence_analysis(
                query=config.text, answer=llm_text, max_results=5
            )
            forensic_dict = {"legacy": True, "evidence": evidence_result.to_dict()}

        # POST-PASSES
        post_analysis = self.reasoning.execute_post_passes(llm_text, pre_analysis)

        # DYNAMIC BOUNDARY (v4 — computed from forensic data)
        if forensic_result and forensic_result.pipeline_succeeded:
            boundary_data = DynamicBoundaryEngine.compute_evidence_severity(
                bayesian_confidence=forensic_result.bayesian_confidence,
                agreement_score=forensic_result.agreement_score,
                contradiction_count=len(forensic_result.contradictions),
                source_reliability=forensic_result.source_reliability_avg,
                claims_total=len(forensic_result.all_claims),
                claims_verified=sum(
                    1 for c in forensic_result.all_claims
                    if c.agreement_count > 0
                ),
            )
        else:
            boundary_data = self.boundary.evaluate(
                config.text,
                context_observations=[llm_text[:500]] if llm_text else [],
                session_data=self.session.get_state_dict(),
            )

        trace.record_stage(
            "post_boundary",
            post_analysis.get("final_confidence", 0.5),
            f"Boundary severity: {boundary_data.get('severity_score', 0)}"
        )

        # Session Update
        self.session.update(
            config.text, config.mode.value,
            boundary_result=boundary_data,
        )
        session_state = self.session.get_state_dict()

        # Confidence (use forensic Bayesian confidence if available)
        if forensic_result and forensic_result.pipeline_succeeded:
            final_conf = forensic_result.bayesian_confidence
            # Apply boundary penalty
            boundary_penalty = boundary_data.get("severity_score", 0) / 200.0
            final_conf = max(0.05, min(0.95, final_conf - boundary_penalty))
        else:
            model_weight = 1.0
            if config.adaptive_learning and self.knowledge_learner:
                model_weight = self.knowledge_learner.get_model_weight("groq")
            base_components = self.confidence_engine.compute_standard(
                base_confidence=post_analysis.get("final_confidence", 0.5),
                boundary_result=boundary_data,
                session_state=session_state,
                model_weight=model_weight,
            )
            final_conf = base_components.final_confidence

        trace.record_stage("final", final_conf, "After evidence adjustment")

        # Format output
        formatter_data = {
            "session": {
                "primary_goal": session_state.get("primary_goal", "Analysis"),
                "inferred_domain": session_state.get("inferred_domain", "General"),
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
            },
            "evidence": forensic_dict,
            "boundary": {
                "risk_level": boundary_data.get("risk_level", "LOW"),
                "severity_score": boundary_data.get("severity_score", 0),
                "explanation": boundary_data.get("explanation", ""),
            },
            "confidence_evolution": trace.to_dict(),
            "synthesis": llm_text,
        }
        formatted = self.formatter.format_evidence(formatter_data)

        # Build omega_metadata with forensic pipeline data
        omega_metadata = {
            "forensic_result": forensic_dict,
            "pipeline_steps": [],
        }
        if forensic_result:
            omega_metadata["pipeline_steps"] = [
                {"step": p.get("name", ""), "label": p.get("name", ""), "status": p.get("status", "")}
                for p in forensic_result.phase_log
            ]

        return {
            "formatted_output": formatted,
            "mode": "research",
            "sub_mode": "evidence",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Forensic Evidence Analysis"),
            "session_state": session_state,
            "boundary_result": boundary_data,
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_evolution": trace.to_dict(),
            "fragility_index": session_state.get("fragility_index", 0.0),
            "omega_metadata": omega_metadata,
            "llm_result": None,
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
