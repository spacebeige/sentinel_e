"""
Omega Cognitive Kernel — Sentinel-E

Central orchestrator for the Omega Cognitive Kernel system.
Implements:
- 3-mode operation (STANDARD, EXPERIMENTAL, KILL)
- 9-pass multi-pass reasoning protocol
- Session intelligence with persistent state
- Boundary evaluation with trend analysis
- Structured output enforcement

Architecture:
  API Request → OmegaKernel.process()
    → SessionIntelligence.update()
    → MultiPassReasoning.pre_passes(1-4)
    → SigmaV4Orchestrator.run_sentinel() [LLM execution]
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

from core.session_intelligence import SessionIntelligence
from core.multipass_reasoning import MultiPassReasoningEngine
from core.omega_boundary import OmegaBoundaryEvaluator
from core.omega_formatter import OmegaOutputFormatter
from core.behavioral_analytics import BehavioralAnalyzer
from core.evidence_engine import EvidenceEngine
from core.debate_orchestrator import DebateOrchestrator

logger = logging.getLogger("Omega-Kernel")


# ============================================================
# VALID MODES
# ============================================================

VALID_MODES = {"standard", "experimental", "kill", "conversational", "forensic"}
OMEGA_MODES = {"standard", "experimental", "kill"}
VALID_SUB_MODES = {"debate", "glass", "evidence"}


@dataclass
class OmegaConfig:
    """Configuration for a single Omega Kernel execution."""
    text: str
    mode: str = "standard"
    sub_mode: str = "debate"           # debate | glass | evidence  (experimental sub-modes)
    kill: bool = False                 # Kill switch (only valid inside glass sub-mode)
    enable_shadow: bool = False
    rounds: int = 3
    chat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        # Normalize mode
        if self.mode not in VALID_MODES:
            logger.warning(f"Invalid mode '{self.mode}', defaulting to 'standard'")
            self.mode = "standard"
        # Normalize sub_mode
        if self.sub_mode not in VALID_SUB_MODES:
            self.sub_mode = "debate"
        # Kill is only valid inside glass
        if self.kill and self.sub_mode != "glass":
            logger.warning("Kill switch only valid in glass sub-mode. Forcing sub_mode=glass.")
            self.sub_mode = "glass"


class OmegaCognitiveKernel:
    """
    The Omega Cognitive Kernel.
    
    Wraps the existing SigmaV4 orchestrator with:
    - Session intelligence layer
    - Multi-pass reasoning protocol
    - Boundary evaluation system
    - Structured output enforcement
    
    Does NOT replace the LLM execution layer — enhances it.
    """

    def __init__(self, sigma_orchestrator=None):
        """
        Initialize the Omega Kernel.
        
        Args:
            sigma_orchestrator: The existing SentinelSigmaOrchestratorV4 instance.
                               If None, operates in analysis-only mode.
        """
        self.sigma = sigma_orchestrator
        self.session = SessionIntelligence()
        self.reasoning = MultiPassReasoningEngine()
        self.boundary = OmegaBoundaryEvaluator()
        self.formatter = OmegaOutputFormatter()
        self.behavioral = BehavioralAnalyzer()
        self.evidence = EvidenceEngine()
        self.debate = None  # Initialized lazily when sigma is available
        self._initialized = False

        # Initialize debate orchestrator if sigma has a client
        if self.sigma and hasattr(self.sigma, 'client'):
            self.debate = DebateOrchestrator(self.sigma.client)

        logger.info("Omega Cognitive Kernel v4.5 initialized.")

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
    def restore_from_session(cls, session_data: Dict[str, Any], sigma_orchestrator=None) -> "OmegaCognitiveKernel":
        """Restore kernel from persisted session data."""
        kernel = cls(sigma_orchestrator=sigma_orchestrator)
        session_payload = session_data.get("session", {})
        if session_payload:
            kernel.session = SessionIntelligence.from_dict(session_payload)
            kernel._initialized = session_data.get("initialized", False)
        # Debate orchestrator initialized in __init__ if sigma has client
        return kernel

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    async def process(self, config: OmegaConfig) -> Dict[str, Any]:
        """
        Main entry point for all Omega processing.
        Routes to mode-specific handlers after common preprocessing.
        
        v3.0: Experimental mode now routes to sub-mode handlers:
          - debate: Multi-model debate with hypothesis extraction
          - glass: Boundary signals + behavioral analytics + embedded kill
          - evidence: External search + source reliability + contradiction detection
        """
        logger.info(f"Omega processing: mode={config.mode}, sub_mode={config.sub_mode}, text_len={len(config.text)}")

        # 1. Session Intelligence Update
        if not self._initialized:
            self.session.initialize(config.text, config.mode)
            self._initialized = True
            is_first = True
        else:
            is_first = False

        # Track sub-mode usage
        if config.mode == "experimental":
            self.session.state.sub_mode_history.append(config.sub_mode)

        # 2. Mode Routing
        if config.mode == "kill":
            # Legacy kill route → redirect to glass+kill for backward compat
            config.sub_mode = "glass"
            config.kill = True
            return await self._process_glass(config, is_first)
        elif config.mode == "experimental":
            if config.sub_mode == "glass":
                return await self._process_glass(config, is_first)
            elif config.sub_mode == "evidence":
                return await self._process_evidence(config, is_first)
            else:
                # Default: debate
                return await self._process_debate(config, is_first)
        else:
            return await self._process_standard(config, is_first)

    # ============================================================
    # STANDARD MODE PROCESSOR
    # ============================================================

    async def _process_standard(self, config: OmegaConfig, is_first: bool) -> Dict[str, Any]:
        """
        STANDARD mode processing pipeline:
        1. Multi-pass pre-analysis (Passes 1-4)
        2. LLM execution via Sigma orchestrator
        3. Multi-pass post-analysis (Passes 5-9)
        4. Boundary evaluation
        5. Session update
        6. Structured output formatting
        """
        # PASS 1-4: Pre-generation analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode, config.history
        )
        logger.info(f"Pre-analysis complete. Initial confidence: {pre_analysis['initial_confidence']}")

        # LLM Execution (via existing Sigma orchestrator)
        llm_result = None
        llm_text = ""
        if self.sigma:
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            sigma_config = SigmaV4Config(
                text=config.text,
                mode="conversational" if config.mode == "standard" else config.mode,
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
        post_analysis = self.reasoning.execute_post_passes(
            llm_text, pre_analysis
        )
        logger.info(f"Post-analysis complete. Final confidence: {post_analysis['final_confidence']}")

        # Boundary Evaluation
        boundary_result = self.boundary.evaluate(
            config.text,
            context_observations=[llm_text[:500]] if llm_text else [],
            session_data=self.session.get_state_dict(),
        )

        # Session Update
        self.session.update(
            config.text, config.mode,
            boundary_result=boundary_result,
        )
        session_state = self.session.get_state_dict()

        # Problem Decomposition (from pre-analysis gaps/intent)
        decomposition = self._build_decomposition(pre_analysis)

        # Structured Output Assembly
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
                "explanation": boundary_result.get("explanation", ""),
            },
            "session": {
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
                "error_patterns": session_state.get("error_patterns", []),
                "reasoning_depth": session_state.get("reasoning_depth", "standard"),
            },
            "confidence": {
                "value": post_analysis.get("final_confidence", 0.5),
                "explanation": self._confidence_explanation(post_analysis),
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
            "confidence": post_analysis.get("final_confidence", 0.5),
            "llm_result": llm_result,
        }

    # ============================================================
    # DEBATE SUB-MODE PROCESSOR (formerly experimental)
    # ============================================================

    async def _process_debate(self, config: OmegaConfig, is_first: bool) -> Dict[str, Any]:
        """
        DEBATE sub-mode processing pipeline (v4.5 — True Adversarial Multi-Round):
        1. Multi-pass pre-analysis
        2. DebateOrchestrator: parallel 3-model multi-round debate
        3. Boundary evaluation across debate positions
        4. Session update with debate analysis
        5. Structured per-round per-model output formatting
        """
        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode, config.history
        )

        # True Multi-Round Debate (via DebateOrchestrator)
        debate_result = None
        if self.debate:
            debate_result = await self.debate.run_debate(
                query=config.text,
                rounds=config.rounds,
            )
        else:
            logger.warning("DebateOrchestrator not available — falling back to sigma")
            # Fallback: use sigma if debate orchestrator not available
            if self.sigma:
                from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
                sigma_config = SigmaV4Config(
                    text=config.text,
                    mode="experimental",
                    enable_shadow=config.enable_shadow,
                    rounds=config.rounds,
                    chat_id=config.chat_id,
                    history=config.history,
                )
                llm_result = await self.sigma.run_sentinel(sigma_config)
                llm_text = llm_result.data.get("priority_answer", "")
            else:
                llm_text = "[ANALYSIS-ONLY MODE]"

        # Build analysis text for post-passes
        if debate_result:
            # Use the analysis synthesis as the "LLM text" for post-analysis
            analysis_text = debate_result.analysis.synthesis if debate_result.analysis else ""
            debate_dict = debate_result.to_dict()
        else:
            analysis_text = llm_text if 'llm_text' in dir() else ""
            debate_dict = {}

        # PASS 5-9: Post-analysis on debate synthesis
        post_analysis = self.reasoning.execute_post_passes(analysis_text, pre_analysis)

        # Debate Boundary Analysis
        # Build model positions for boundary evaluation
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

        # Session Update with debate data
        self.session.update(
            config.text, config.mode,
            boundary_result={
                "severity_score": debate_boundary.get("compound_severity", 0),
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

        # Confidence evolution
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        debate_conf = debate_result.analysis.confidence_recalibration if debate_result and debate_result.analysis else 0.5
        post_stress_conf = post_analysis.get("final_confidence", 0.5)
        final_conf = round((debate_conf * 0.6 + post_stress_conf * 0.4), 4)

        # Format debate output with per-round per-model data
        formatter_data = {
            "session": {
                "primary_goal": session_state.get("primary_goal", "Analysis"),
                "inferred_domain": session_state.get("inferred_domain", "General"),
                "user_expertise_score": session_state.get("user_expertise_score", 0.5),
            },
            "debate": debate_dict,  # Full per-round per-model data
            "boundary": {
                "detected_count": len(debate_boundary.get("per_model_boundaries", [])),
                "severity_score": debate_boundary.get("compound_severity", 0),
                "explanation": debate_boundary.get("explanation", ""),
            },
            "confidence_evolution": {
                "initial": initial_conf,
                "post_debate": debate_conf,
                "post_stress": post_stress_conf,
                "final": final_conf,
            },
            "fragility": {
                "score": session_state.get("fragility_index", 0.0),
                "explanation": self._fragility_explanation(session_state.get("fragility_index", 0.0)),
            },
            "synthesis": debate_result.analysis.synthesis if debate_result and debate_result.analysis else analysis_text,
        }

        formatted = self.formatter.format_debate(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "experimental",
            "sub_mode": "debate",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Adversarial Debate"),
            "session_state": session_state,
            "boundary_result": debate_boundary,
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_evolution": {
                "initial": initial_conf,
                "post_debate": debate_conf,
                "post_stress": post_stress_conf,
                "final": final_conf,
            },
            "fragility_index": session_state.get("fragility_index", 0.0),
            "debate_result": debate_dict,
            "llm_result": None,
        }

    # ============================================================
    # GLASS SUB-MODE PROCESSOR (includes embedded kill)
    # ============================================================

    async def _process_glass(self, config: OmegaConfig, is_first: bool = False) -> Dict[str, Any]:
        """
        GLASS sub-mode: Research-grade boundary + behavioral analytics console.
        
        When config.kill=True: Kill mode — diagnostic snapshot only, no new reasoning.
        When config.kill=False: Full glass analysis with LLM + behavioral analytics.
        """
        # --- KILL SWITCH (inside Glass) ---
        if config.kill:
            diagnostic = self.session.get_kill_diagnostic()
            boundary_trend = self.boundary.get_trend()

            kill_data = {
                "session_state": {
                    "chat_name": diagnostic.get("chat_name", "Unknown"),
                    "primary_goal": diagnostic.get("primary_goal", "Unknown"),
                    "inferred_domain": diagnostic.get("inferred_domain", "Unknown"),
                    "user_expertise_score": diagnostic.get("user_expertise_score", 0.0),
                },
                "boundary_snapshot": {
                    "latest_severity": diagnostic.get("latest_boundary_severity", 0),
                    "trend": boundary_trend.get("trend", "insufficient_data"),
                },
                "disagreement_score": diagnostic.get("disagreement_score", 0.0),
                "fragility_index": diagnostic.get("fragility_index", 0.0),
                "error_patterns": diagnostic.get("recurring_error_patterns", []),
                "session_confidence": diagnostic.get("session_confidence", 0.5),
                "confidence_explanation": diagnostic.get("session_confidence_explanation", ""),
                "behavioral_risk_profile": self.session.state.behavioral_risk_profile,
            }

            formatted = self.formatter.format_kill(kill_data)

            return {
                "formatted_output": formatted,
                "mode": "experimental",
                "sub_mode": "glass",
                "kill_active": True,
                "chat_id": config.chat_id,
                "chat_name": diagnostic.get("chat_name", "Glass Kill Diagnostic"),
                "session_state": self.session.get_state_dict(),
                "diagnostic": diagnostic,
                "confidence": diagnostic.get("session_confidence", 0.5),
                "llm_result": None,
            }

        # --- GLASS ANALYSIS (non-kill) ---
        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode, config.history
        )

        # LLM Execution (standard reasoning for glass)
        llm_result = None
        llm_text = ""
        if self.sigma:
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            sigma_config = SigmaV4Config(
                text=config.text,
                mode="experimental",
                enable_shadow=config.enable_shadow,
                rounds=config.rounds,
                chat_id=config.chat_id,
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

        # Behavioral Analytics
        behavioral_profile = self.behavioral.analyze(
            text=config.text,
            llm_output=llm_text,
            shadow_result=None,
            context=self.session.get_state_dict(),
        )

        # Update session with behavioral risk
        self.session.state.behavioral_risk_profile = behavioral_profile.to_dict()

        # Session Update
        self.session.update(
            config.text, config.mode,
            boundary_result=boundary_result,
        )
        session_state = self.session.get_state_dict()

        # Confidence evolution
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        post_analysis_conf = post_analysis.get("final_confidence", 0.5)
        boundary_severity = boundary_result.get("severity_score", 0) / 100.0
        final_conf = round(max(0.0, post_analysis_conf - boundary_severity * 0.15), 4)

        # Format glass output
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
            "confidence_evolution": {
                "initial": initial_conf,
                "post_analysis": post_analysis_conf,
                "post_boundary": final_conf,
                "final": final_conf,
            },
            "fragility": {
                "score": session_state.get("fragility_index", 0.0),
                "explanation": self._fragility_explanation(session_state.get("fragility_index", 0.0)),
            },
            "severity_trend": self.boundary.get_trend(),
            "synthesis": llm_text,
        }

        formatted = self.formatter.format_glass(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "experimental",
            "sub_mode": "glass",
            "kill_active": False,
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Glass Analysis"),
            "session_state": session_state,
            "boundary_result": boundary_result,
            "behavioral_risk": behavioral_profile.to_dict(),
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_evolution": {
                "initial": initial_conf,
                "post_analysis": post_analysis_conf,
                "post_boundary": final_conf,
                "final": final_conf,
            },
            "fragility_index": session_state.get("fragility_index", 0.0),
            "llm_result": llm_result,
        }

    # ============================================================
    # EVIDENCE SUB-MODE PROCESSOR
    # ============================================================

    async def _process_evidence(self, config: OmegaConfig, is_first: bool) -> Dict[str, Any]:
        """
        EVIDENCE sub-mode: External search + source reliability + contradiction detection.
        
        Pipeline:
        1. Pre-analysis passes
        2. Evidence engine search (Tavily)
        3. LLM synthesis with evidence context
        4. Source reliability scoring
        5. Contradiction detection
        6. Boundary evaluation
        7. Structured output
        """
        # PASS 1-4: Pre-analysis
        pre_analysis = self.reasoning.execute_pre_passes(
            config.text, config.mode, config.history
        )

        # Evidence Search
        evidence_result = await self.evidence.search_evidence(
            query=config.text,
            max_results=5,
            search_depth="advanced",
        )

        # Build evidence context for LLM
        evidence_context = ""
        if evidence_result.sources:
            evidence_snippets = []
            for i, src in enumerate(evidence_result.sources, 1):
                evidence_snippets.append(
                    f"[Source {i}] ({src.domain}, reliability={src.reliability_score:.2f}): {src.content_snippet[:200]}"
                )
            evidence_context = "\n".join(evidence_snippets)

        # LLM Execution with evidence context
        llm_result = None
        llm_text = ""
        if self.sigma:
            from backend.sentinel.sentinel_sigma_v4 import SigmaV4Config
            enhanced_text = config.text
            if evidence_context:
                enhanced_text = f"{config.text}\n\n[Evidence Context]:\n{evidence_context}"
            
            sigma_config = SigmaV4Config(
                text=enhanced_text,
                mode="experimental",
                enable_shadow=config.enable_shadow,
                rounds=config.rounds,
                chat_id=config.chat_id,
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

        # Session Update
        self.session.update(
            config.text, config.mode,
            boundary_result=boundary_result,
        )
        session_state = self.session.get_state_dict()

        # Confidence with evidence factor
        initial_conf = pre_analysis.get("initial_confidence", 0.5)
        post_analysis_conf = post_analysis.get("final_confidence", 0.5)
        evidence_boost = evidence_result.evidence_confidence * 0.2  # Evidence increases confidence
        contradiction_penalty = len(evidence_result.contradictions) * 0.05
        final_conf = round(max(0.0, min(post_analysis_conf + evidence_boost - contradiction_penalty, 0.99)), 4)

        # Format evidence output
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
            "confidence_evolution": {
                "initial": initial_conf,
                "post_analysis": post_analysis_conf,
                "post_evidence": final_conf,
                "final": final_conf,
            },
            "synthesis": llm_text,
        }

        formatted = self.formatter.format_evidence(formatter_data)

        return {
            "formatted_output": formatted,
            "mode": "experimental",
            "sub_mode": "evidence",
            "chat_id": config.chat_id,
            "chat_name": session_state.get("chat_name", "Evidence Analysis"),
            "session_state": session_state,
            "boundary_result": boundary_result,
            "evidence_result": evidence_result.to_dict(),
            "reasoning_trace": post_analysis.get("trace_summary", {}),
            "confidence": final_conf,
            "confidence_evolution": {
                "initial": initial_conf,
                "post_analysis": post_analysis_conf,
                "post_evidence": final_conf,
                "final": final_conf,
            },
            "fragility_index": session_state.get("fragility_index", 0.0),
            "llm_result": llm_result,
        }

    # ============================================================
    # HELPER METHODS
    # ============================================================

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

    def _analyze_stress(self, pre: Dict, post: Dict, debate: Dict) -> Dict[str, Any]:
        """Analyze stress test results for experimental mode."""
        disagreements = debate.get("disagreements", [])
        gaps = pre.get("gaps", {}).get("gaps", [])
        critique = post.get("critique", {})

        weakest = []
        failures = []
        survivors = []

        # High-severity gaps are weak points
        for gap in gaps:
            if gap.get("severity") == "high":
                weakest.append(gap.get("description", ""))

        # Critique issues
        for issue in critique.get("issues", []):
            if issue.get("severity") == "high":
                failures.append(issue.get("description", ""))
            else:
                survivors.append(issue.get("description", ""))

        return {
            "weakest": ", ".join(weakest) if weakest else "No critical weaknesses identified",
            "failures": failures,
            "survivors": survivors if survivors else ["Core reasoning survived stress testing"],
        }

    def _build_error_profile(self, pre: Dict, session: Dict) -> Dict[str, Any]:
        """Build user error profile for experimental mode."""
        assumptions = pre.get("assumptions", {})
        gaps = pre.get("gaps", {})

        inconsistencies = []
        missing = []
        unsafe = []

        for gap in gaps.get("gaps", []):
            if gap.get("type") == "missing_constraint":
                missing.append(gap.get("description", ""))
            elif gap.get("type") in ("assumption_overload", "vagueness"):
                inconsistencies.append(gap.get("description", ""))

        for a in assumptions.get("implicit", []):
            if "unverified" in a.lower() or "unstated" in a.lower():
                unsafe.append(a)

        return {
            "logical_inconsistencies": inconsistencies,
            "missing_constraints": missing,
            "unsafe_assumptions": unsafe,
        }

    def _extract_hypotheses(self, perspectives: List[Dict], debate: Dict) -> List[str]:
        """Extract hypotheses from perspectives and debate data."""
        hypotheses = []
        seen = set()

        for p in perspectives:
            for kp in p.get("key_points", []):
                if kp and kp not in seen:
                    hypotheses.append(kp)
                    seen.add(kp)

        # Ensure minimum count
        if not hypotheses:
            hypotheses = ["Primary analysis perspective validated by debate consensus"]

        return hypotheses[:5]  # Cap at 5

    def _build_dependency_description(self, hypotheses: List[str]) -> str:
        """Build hypothesis dependency text."""
        if len(hypotheses) < 2:
            return "Single hypothesis — no dependency analysis required."

        return (
            f"{len(hypotheses)} hypotheses identified. "
            f"H1 serves as the central node. "
            f"H2–H{len(hypotheses)} are conditionally dependent on H1's validity."
        )

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
