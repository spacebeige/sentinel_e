"""
Tests that the sub-mode pipeline adapter and pipeline functions
produce the correct data shapes for frontend views.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_glass_pipeline_data_contract():
    """Glass pipeline output must match GlassView.js data contract."""
    from core.glass_pipeline import build_glass_result

    # Create mock results using the adapter pattern from main.py
    class MockOutput:
        def __init__(self, model_name, raw_output, success=True):
            self.model_name = model_name
            self.raw_output = raw_output
            self.success = success

    class MockScore:
        def __init__(self, model_name, final_score):
            self.model_name = model_name
            self.final_score = final_score
            self.topic_alignment = 0.7
            self.knowledge_grounding = 0.6
            self.specificity = 0.65
            self.confidence_calibration = 0.7
            self.drift_penalty = 0.05

    class MockResult:
        def __init__(self, model_name, raw_output, score_val):
            self.output = MockOutput(model_name, raw_output)
            self.score = MockScore(model_name, score_val)

    results = [
        MockResult("llama33-70b", "AI is transforming healthcare through early diagnosis, drug discovery, and personalized treatment plans.", 0.85),
        MockResult("mixtral-8x7b", "Machine learning in healthcare enables predictive analytics and improved patient outcomes.", 0.78),
        MockResult("llama4-scout", "Healthcare AI applications range from imaging analysis to clinical decision support systems.", 0.72),
    ]
    scoring = [MockScore(r.output.model_name, r.score.final_score) for r in results]

    result = build_glass_result(
        all_results=results,
        scoring_breakdown=scoring,
        divergence_metrics={"max_divergence": 0.15},
        aggregated_answer="AI is transforming healthcare...",
        winning_model="llama33-70b",
        drift_score=0.05,
        volatility_score=0.1,
    )

    # Verify top-level keys match GlassView expectations
    assert "assessments" in result
    assert "tactical_map" in result
    assert "overall_trust" in result
    assert "consensus_risk" in result
    assert "reasoning_graph" in result
    assert len(result["assessments"]) == 3
    assert result["consensus_risk"] in ("LOW", "MEDIUM", "HIGH")
    assert 0 < result["overall_trust"] < 1

    # Verify per-assessment structure
    a = result["assessments"][0]
    assert "auditor_name" in a
    assert "subject_name" in a
    assert "trust_score" in a
    assert "logical_coherence" in a
    assert "hidden_assumptions" in a
    assert "bias_patterns" in a
    assert "confidence_inflation" in a
    assert "persuasion_tactics" in a
    assert "evidence_quality" in a
    assert "completeness" in a
    assert "strong_points" in a
    assert "weak_points" in a
    assert "risk_factors" in a  # Frontend uses risk_factors, not red_flags
    assert "overall_assessment" in a  # Frontend renders this narrative
    assert isinstance(a["strong_points"], list)

    # Verify tactical map
    tm = result["tactical_map"]
    assert "model_profiles" in tm
    for model_id, profile in tm["model_profiles"].items():
        assert "model_name" in profile  # Frontend accesses profile.model_name
        assert "trust" in profile
        assert "bias_risk" in profile

    # Verify reasoning graph
    rg = result["reasoning_graph"]
    assert "nodes" in rg
    assert "edges" in rg
    assert len(rg["nodes"]) > 0
    assert len(rg["edges"]) > 0


def test_evidence_pipeline_claim_format():
    """Claims from evidence pipeline must use statement/model_origin keys."""
    from core.evidence_pipeline import build_evidence_result
    import asyncio

    class MockOutput:
        def __init__(self, model_name, raw_output, success=True):
            self.model_name = model_name
            self.raw_output = raw_output
            self.success = success

    class MockScore:
        def __init__(self, final_score):
            self.final_score = final_score

    class MockResult:
        def __init__(self, model_name, raw_output, score_val):
            self.output = MockOutput(model_name, raw_output)
            self.score = MockScore(score_val)

    results = [
        MockResult("llama33-70b", "The Earth orbits the Sun at a distance of approximately 93 million miles. This results in a year being about 365 days long.", 0.85),
    ]
    scoring = [MockScore(0.85)]

    result = asyncio.get_event_loop().run_until_complete(
        build_evidence_result(
            query="How far is Earth from the Sun?",
            all_results=results,
            scoring_breakdown=scoring,
            aggregated_answer="Earth is about 93 million miles from the Sun.",
            winning_model="llama33-70b",
        )
    )

    # Verify top-level keys
    assert "all_claims" in result
    assert "contradictions" in result
    assert "bayesian_confidence" in result
    assert "agreement_score" in result
    assert "source_reliability_avg" in result
    assert "sources" in result
    assert "phase_log" in result
    assert "verbatim_citations" in result

    # Verify claim uses frontend-expected keys
    if result["all_claims"]:
        claim = result["all_claims"][0]
        assert "statement" in claim, f"Expected 'statement', got keys: {claim.keys()}"
        assert "model_origin" in claim, f"Expected 'model_origin', got keys: {claim.keys()}"
        assert "final_confidence" in claim
        assert "verifications" in claim
        # Must NOT have old keys
        assert "claim" not in claim, "'claim' key should be renamed to 'statement'"
        assert "model" not in claim, "'model' key should be renamed to 'model_origin'"


def test_synthesis_pipeline_data_contract():
    """Synthesis pipeline output must match SynthesisView.js data contract."""
    from core.synthesis_engine import build_synthesis_result

    class MockOutput:
        def __init__(self, model_name, raw_output, success=True):
            self.model_name = model_name
            self.raw_output = raw_output
            self.success = success

    class MockScore:
        def __init__(self, model_name, final_score):
            self.model_name = model_name
            self.final_score = final_score

    class MockResult:
        def __init__(self, model_name, raw_output, score_val):
            self.output = MockOutput(model_name, raw_output)
            self.score = MockScore(model_name, score_val)

    results = [
        MockResult("llama33-70b", "The primary approach to solving climate change involves renewable energy adoption, carbon capture technology, and policy reform.", 0.85),
        MockResult("mixtral-8x7b", "Climate solutions must address both mitigation through clean energy and adaptation through infrastructure resilience.", 0.78),
        MockResult("llama4-scout", "A holistic approach combining technological innovation with behavioral change is essential for climate action.", 0.72),
    ]
    scoring = [MockScore(r.output.model_name, r.score.final_score) for r in results]

    result = build_synthesis_result(
        all_results=results,
        scoring_breakdown=scoring,
        divergence_metrics={"max_divergence": 0.15},
        aggregated_answer="Climate change requires a multi-faceted approach...",
        winning_model="llama33-70b",
    )

    # Verify top-level keys match SynthesisView contract
    assert "draft" in result
    assert "draft_model" in result
    assert "draft_score" in result
    assert "revisions" in result
    assert "consensus_score" in result
    assert "improvement_delta" in result
    assert "models_participated" in result
    assert "synthesis_graph" in result

    assert result["models_participated"] == 3
    assert result["draft_model"] == "llama33-70b"
    assert 0 <= result["consensus_score"] <= 1
    assert len(result["revisions"]) == 2  # 3 models - 1 primary = 2 reviewers

    # Verify revision structure
    rev = result["revisions"][0]
    assert "model" in rev
    assert "type" in rev
    assert rev["type"] in ("endorsement", "refinement", "alternative")
    assert "agreement" in rev
    assert "comment" in rev
    assert "key_additions" in rev
    assert "output_preview" in rev

    # Verify graph
    graph = result["synthesis_graph"]
    assert "nodes" in graph
    assert "edges" in graph


def test_pipeline_adapter():
    """The adapter classes bridge StructuredModelOutput to pipeline interface."""
    # Simulate what main.py does
    class FakeStructuredModelOutput:
        def __init__(self):
            self.model_id = "llama33-70b"
            self.model_name = "Llama 3.3 70B"
            self.raw_output = "Test output content for pipeline processing."
            self.confidence = 0.85
            self.error = None
            self.position = "Test position"

        @property
        def succeeded(self):
            return self.error is None and bool(self.position)

    # Import adapter classes from our main module indirectly (simulate)
    class _PipelineScore:
        __slots__ = ("model_name", "topic_alignment", "knowledge_grounding",
                     "specificity", "confidence_calibration", "drift_penalty", "final_score")
        def __init__(self, model_name, confidence):
            self.model_name = model_name
            self.final_score = confidence
            self.topic_alignment = confidence
            self.knowledge_grounding = confidence * 0.9
            self.specificity = confidence * 0.85
            self.confidence_calibration = confidence
            self.drift_penalty = 0.0

    class _PipelineOutput:
        __slots__ = ("success", "raw_output", "model_name")
        def __init__(self, succeeded, raw_output, model_name):
            self.success = succeeded
            self.raw_output = raw_output
            self.model_name = model_name

    class _PipelineResult:
        __slots__ = ("output", "score")
        def __init__(self, smo):
            self.output = _PipelineOutput(smo.succeeded, smo.raw_output, smo.model_name)
            self.score = _PipelineScore(smo.model_name, smo.confidence)

    smo = FakeStructuredModelOutput()
    adapted = _PipelineResult(smo)

    assert adapted.output.success is True
    assert adapted.output.raw_output == "Test output content for pipeline processing."
    assert adapted.output.model_name == "Llama 3.3 70B"
    assert adapted.score.final_score == 0.85
    assert adapted.score.model_name == "Llama 3.3 70B"


def test_no_gemma_in_registry():
    """Verify gemma-7b key has been renamed to llama4-scout."""
    from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY, MODEL_DEBATE_TIERS, MODEL_FALLBACK_MAP

    assert "llama4-scout" in COGNITIVE_MODEL_REGISTRY, "llama4-scout should be in registry"
    assert "gemma-7b" not in COGNITIVE_MODEL_REGISTRY, "gemma-7b should be removed from registry"

    assert "llama4-scout" in MODEL_DEBATE_TIERS
    assert "gemma-7b" not in MODEL_DEBATE_TIERS

    assert "llama4-scout" in MODEL_FALLBACK_MAP
    assert "gemma-7b" not in MODEL_FALLBACK_MAP
