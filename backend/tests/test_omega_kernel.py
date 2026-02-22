"""
Tests for the Omega Cognitive Kernel.

Covers:
- Session Intelligence lifecycle
- Multi-Pass Reasoning Engine (9-pass protocol)
- Omega Boundary Evaluator
- Structured Output Formatter (all 3 modes)
- OmegaCognitiveKernel integration (analysis-only mode)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

pytest_plugins = ('anyio',)

from backend.core.session_intelligence import SessionIntelligence, SessionState
from backend.core.multipass_reasoning import MultiPassReasoningEngine
from backend.core.omega_boundary import OmegaBoundaryEvaluator
from backend.core.omega_formatter import OmegaOutputFormatter
from backend.core.omega_kernel import OmegaCognitiveKernel, OmegaConfig


# ============================================================
# SESSION INTELLIGENCE TESTS
# ============================================================

class TestSessionIntelligence:

    def test_initialization(self):
        si = SessionIntelligence()
        state = si.initialize("How do I deploy a Python microservice to Kubernetes?", "standard")
        assert state.chat_name is not None
        assert len(state.chat_name) > 0
        assert state.inferred_domain == "software_engineering"
        assert state.message_count == 1
        assert 0 <= state.user_expertise_score <= 1

    def test_domain_inference_ml(self):
        si = SessionIntelligence()
        si.initialize("Train a transformer model with attention layers for NLP", "standard")
        assert si.state.inferred_domain == "machine_learning"

    def test_domain_inference_security(self):
        si = SessionIntelligence()
        si.initialize("How to detect zero-day vulnerabilities in a firewall?", "standard")
        assert si.state.inferred_domain == "cybersecurity"

    def test_expertise_high(self):
        si = SessionIntelligence()
        si.initialize(
            "The amortized complexity of this deterministic algorithm needs linearizable consensus with backpressure",
            "standard"
        )
        assert si.state.user_expertise_score >= 0.7

    def test_expertise_low(self):
        si = SessionIntelligence()
        si.initialize("What is a variable? Help me with a simple example please", "standard")
        assert si.state.user_expertise_score <= 0.4

    def test_session_update_increments_count(self):
        si = SessionIntelligence()
        si.initialize("Initial query", "standard")
        si.update("Follow up query", "standard")
        assert si.state.message_count == 2

    def test_error_pattern_detection(self):
        si = SessionIntelligence()
        si.initialize("Obviously this is the best approach", "standard")
        si.update("Everyone knows that Python is always faster than Java", "standard")
        patterns = [ep.pattern_type for ep in si.state.error_patterns]
        assert "assumption_without_evidence" in patterns or "hasty_generalization" in patterns

    def test_kill_diagnostic(self):
        si = SessionIntelligence()
        si.initialize("Test query for kill diagnostic", "standard")
        diag = si.get_kill_diagnostic()
        assert "chat_name" in diag
        assert "session_confidence" in diag
        assert "fragility_index" in diag

    def test_fragility_starts_low(self):
        si = SessionIntelligence()
        si.initialize("Simple clear query", "standard")
        assert si.state.fragility_index <= 0.3

    def test_reasoning_depth_experimental(self):
        si = SessionIntelligence()
        si.initialize("Analyze this", "experimental")
        assert si.state.reasoning_depth == "maximum"


# ============================================================
# MULTI-PASS REASONING ENGINE TESTS
# ============================================================

class TestMultiPassReasoning:

    def test_pre_passes_execute(self):
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes(
            "Deploy a REST API with authentication to AWS",
            "standard"
        )
        assert "intent" in result
        assert "assumptions" in result
        assert "gaps" in result
        assert "boundary" in result
        assert "initial_confidence" in result

    def test_intent_classification_debugging(self):
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes("Fix this crash in my Python code", "standard")
        assert result["intent"]["intent_type"] == "debugging"

    def test_intent_classification_creation(self):
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes("Build a REST API with FastAPI", "standard")
        assert result["intent"]["intent_type"] == "creation"

    def test_assumption_extraction(self):
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes(
            "Assuming we have a PostgreSQL database, the best approach is always to use an ORM",
            "standard"
        )
        assumptions = result["assumptions"]
        assert len(assumptions["explicit"]) > 0 or len(assumptions["implicit"]) > 0

    def test_gap_detection(self):
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes(
            "Make it scale to millions of users and be completely secure",
            "standard"
        )
        gaps = result["gaps"]["gaps"]
        assert len(gaps) > 0

    def test_post_passes_execute(self):
        engine = MultiPassReasoningEngine()
        pre = engine.execute_pre_passes("Test query", "standard")
        post = engine.execute_post_passes("This is the LLM response.", pre)
        assert "structured_draft" in post
        assert "critique" in post
        assert "final_confidence" in post
        assert "trace_summary" in post

    def test_confidence_decreases_with_issues(self):
        engine = MultiPassReasoningEngine()
        pre = engine.execute_pre_passes(
            "Make it scale to millions and always be secure with no constraints or criteria",
            "standard"
        )
        post = engine.execute_post_passes("Generic response", pre)
        # Confidence should be lower due to gaps
        assert post["final_confidence"] <= pre["initial_confidence"]

    def test_nine_passes_tracked(self):
        engine = MultiPassReasoningEngine()
        pre = engine.execute_pre_passes("Test", "standard")
        post = engine.execute_post_passes("Response", pre)
        trace = post["trace_summary"]
        assert trace["passes_executed"] == 9

    def test_coding_task_detection(self):
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes(
            "Write a Python function that sorts a list",
            "standard"
        )
        assert result["intent"]["requires_code"] is True

    def test_boundary_medical_escalation(self):
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes(
            "What medical treatment should a patient take for a disease with symptoms?",
            "standard"
        )
        assert result["boundary"]["preliminary_severity"] >= 20


# ============================================================
# OMEGA BOUNDARY EVALUATOR TESTS
# ============================================================

class TestOmegaBoundary:

    def test_basic_evaluation(self):
        evaluator = OmegaBoundaryEvaluator()
        result = evaluator.evaluate("Implement a REST API in Python")
        assert "risk_level" in result
        assert "severity_score" in result
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_medical_domain_high_risk(self):
        evaluator = OmegaBoundaryEvaluator()
        result = evaluator.evaluate("What drug should a patient take for a disease?")
        assert result["severity_score"] > 20

    def test_risk_dimensions_present(self):
        evaluator = OmegaBoundaryEvaluator()
        result = evaluator.evaluate("Analyze this data")
        assert "risk_dimensions" in result
        assert "epistemic_risk" in result["risk_dimensions"]

    def test_debate_boundary_evaluation(self):
        evaluator = OmegaBoundaryEvaluator()
        result = evaluator.evaluate_debate_boundaries(
            model_positions=[
                {"model": "A", "position": "Python is best for ML"},
                {"model": "B", "position": "Julia is better for numerical computing"},
            ],
            agreements=["Both support scientific computing"],
            disagreements=["Best language for ML workloads"],
        )
        assert "compound_severity" in result
        assert "per_model_boundaries" in result

    def test_trend_insufficient_data(self):
        evaluator = OmegaBoundaryEvaluator()
        trend = evaluator.get_trend()
        assert trend["trend"] == "insufficient_data"

    def test_trend_after_evaluations(self):
        evaluator = OmegaBoundaryEvaluator()
        evaluator.evaluate("First query")
        evaluator.evaluate("Second query about medical treatment for a patient")
        trend = evaluator.get_trend()
        assert trend["data_points"] == 2


# ============================================================
# OUTPUT FORMATTER TESTS
# ============================================================

class TestOmegaFormatter:

    def test_standard_format_has_all_sections(self):
        formatter = OmegaOutputFormatter()
        output = formatter.format_standard({
            "is_first_message": True,
            "chat_name": "Test Chat",
            "executive_summary": "This is a test summary.",
            "problem_decomposition": [{"component": "A", "description": "Test"}],
            "assumptions": {"explicit": ["A1"], "implicit": ["I1"]},
            "logical_gaps": {"gaps": [{"severity": "medium", "description": "Test gap"}]},
            "solution": "Test solution.",
            "boundary": {"risk_level": "LOW", "severity_score": 10, "explanation": "Safe"},
            "session": {"user_expertise_score": 0.7, "error_patterns": [], "reasoning_depth": "standard"},
            "confidence": {"value": 0.85, "explanation": "High confidence."},
        })
        assert "# Executive Summary" in output
        assert "# Problem Decomposition" in output
        assert "# Identified Assumptions" in output
        assert "# Logical Risk Assessment" in output
        assert "# Structured Solution" in output
        assert "# Boundary Evaluation" in output
        assert "# Session Adaptation Notes" in output
        assert "# Confidence Score" in output

    def test_experimental_format_has_all_sections(self):
        formatter = OmegaOutputFormatter()
        output = formatter.format_experimental({
            "session": {"primary_goal": "Analysis", "inferred_domain": "General", "user_expertise_score": 0.5},
            "perspectives": [{"model": "A", "position": "Test"}],
            "hypotheses": ["H1: Test hypothesis"],
            "hypothesis_dependencies": "H1 is central.",
            "debate_trace": {"disagreements": ["D1"], "resolutions": ["R1"]},
            "stress_testing": {"weakest": "H1", "failures": ["F1"], "survivors": ["S1"]},
            "boundary": {"detected_count": 1, "severity_score": 30, "explanation": "Moderate"},
            "user_error_profile": {"logical_inconsistencies": [], "missing_constraints": [], "unsafe_assumptions": []},
            "confidence_evolution": {"initial": 0.7, "post_debate": 0.65, "post_stress": 0.6, "final": 0.62},
            "fragility": {"score": 0.2, "explanation": "Low fragility"},
            "synthesis": "Test synthesis.",
        })
        assert "# Session Overview" in output
        assert "# Model Initial Perspectives" in output
        assert "# Extracted Hypotheses" in output
        assert "# Debate Trace" in output
        assert "# Stress Testing" in output
        assert "# Boundary Analysis" in output
        assert "# Confidence Evolution" in output
        assert "# Fragility Index" in output
        assert "# Executive Synthesis" in output

    def test_kill_format_has_all_sections(self):
        formatter = OmegaOutputFormatter()
        output = formatter.format_kill({
            "session_state": {
                "chat_name": "Test", "primary_goal": "Analysis",
                "inferred_domain": "General", "user_expertise_score": 0.5,
            },
            "boundary_snapshot": {"latest_severity": 20, "trend": "flat"},
            "disagreement_score": 0.1,
            "fragility_index": 0.15,
            "error_patterns": [],
            "session_confidence": 0.8,
            "confidence_explanation": "Stable session.",
        })
        assert "# Session Cognitive State" in output
        assert "# Boundary Snapshot" in output
        assert "# Disagreement Score" in output
        assert "# Fragility Index" in output
        assert "# Session Confidence" in output


# ============================================================
# OMEGA KERNEL INTEGRATION TESTS (ANALYSIS-ONLY MODE)
# ============================================================

class TestOmegaKernel:

    @pytest.mark.anyio
    async def test_standard_mode_analysis_only(self):
        """Test standard mode without LLM backend (analysis-only)."""
        kernel = OmegaCognitiveKernel(sigma_orchestrator=None)
        config = OmegaConfig(text="Explain how Kubernetes autoscaling works", mode="standard")
        result = await kernel.process(config)
        assert result["mode"] == "standard"
        assert "formatted_output" in result
        assert "# Executive Summary" in result["formatted_output"]
        assert "# Boundary Evaluation" in result["formatted_output"]
        assert "# Confidence Score" in result["formatted_output"]

    @pytest.mark.anyio
    async def test_kill_mode(self):
        """Test KILL mode returns diagnostic only."""
        kernel = OmegaCognitiveKernel(sigma_orchestrator=None)
        # Initialize with a message first
        await kernel.process(OmegaConfig(text="Initial query", mode="standard"))
        # Then kill
        result = await kernel.process(OmegaConfig(text="", mode="kill"))
        assert result["mode"] == "kill"
        assert "# Session Cognitive State" in result["formatted_output"]
        assert result["llm_result"] is None  # Kill mode never calls LLMs

    @pytest.mark.anyio
    async def test_session_persistence(self):
        """Test session state persists across interactions."""
        kernel = OmegaCognitiveKernel(sigma_orchestrator=None)
        await kernel.process(OmegaConfig(text="First query about Python", mode="standard"))
        await kernel.process(OmegaConfig(text="Second query about machine learning models", mode="standard"))
        state = kernel.get_session_state()
        assert state["message_count"] >= 2

    @pytest.mark.anyio
    async def test_experimental_mode_analysis_only(self):
        """Test experimental mode without LLM backend."""
        kernel = OmegaCognitiveKernel(sigma_orchestrator=None)
        config = OmegaConfig(text="Compare Python vs Rust for systems programming", mode="experimental")
        result = await kernel.process(config)
        assert result["mode"] == "experimental"
        assert "# Session Overview" in result["formatted_output"]
        assert "# Fragility Index" in result["formatted_output"]

    @pytest.mark.anyio
    async def test_confidence_tracked(self):
        """Test confidence is tracked in results."""
        kernel = OmegaCognitiveKernel(sigma_orchestrator=None)
        result = await kernel.process(OmegaConfig(text="Simple query", mode="standard"))
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    @pytest.mark.anyio
    async def test_coding_task_formatting(self):
        """Test coding tasks get extended output."""
        kernel = OmegaCognitiveKernel(sigma_orchestrator=None)
        result = await kernel.process(
            OmegaConfig(text="Write a Python function to implement binary search", mode="standard")
        )
        # Should have standard sections
        assert "# Executive Summary" in result["formatted_output"]


# ============================================================
# PRIORITY ORDER VALIDATION
# ============================================================

class TestPriorityOrder:
    """Validates the protocol priority order:
    1. Logical Consistency
    2. Structural Clarity
    3. Boundary Safety
    4. Transparency
    5. Adaptivity
    6. Fluency
    """

    def test_logical_consistency_gaps_detected(self):
        """Priority 1: Logical gaps must always be detected."""
        engine = MultiPassReasoningEngine()
        result = engine.execute_pre_passes(
            "Make it always perfect with no tradeoffs and scale infinitely",
            "standard"
        )
        assert result["gaps"]["total_count"] > 0

    def test_structural_clarity_all_sections_present(self):
        """Priority 2: All required sections must be present in output."""
        formatter = OmegaOutputFormatter()
        output = formatter.format_standard({
            "executive_summary": "Test",
            "solution": "Test",
            "boundary": {"risk_level": "LOW", "severity_score": 0, "explanation": ""},
            "session": {"user_expertise_score": 0.5, "error_patterns": [], "reasoning_depth": "standard"},
            "confidence": {"value": 0.5, "explanation": ""},
        })
        required_sections = [
            "# Executive Summary",
            "# Problem Decomposition",
            "# Identified Assumptions",
            "# Logical Risk Assessment",
            "# Structured Solution",
            "# Boundary Evaluation",
            "# Session Adaptation Notes",
            "# Confidence Score",
        ]
        for section in required_sections:
            assert section in output, f"Missing required section: {section}"

    def test_boundary_safety_high_risk_flagged(self):
        """Priority 3: High-risk domains must be flagged."""
        evaluator = OmegaBoundaryEvaluator()
        result = evaluator.evaluate("What medical treatment should patient take for a disease?")
        assert result["severity_score"] > 15
