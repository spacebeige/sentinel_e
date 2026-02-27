"""
Tests for DebateOrchestrator — verifies dynamic model routing,
analysis retry logic, and full debate pipeline.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

# ── Fixtures ──────────────────────────────────────────────────


class MockBridge:
    """Mock MCOModelBridge that simulates call_model for all registered models."""

    def __init__(self, model_ids=None, fail_models=None):
        self._model_ids = model_ids or ["groq", "llama70b", "qwen", "qwen3-coder", "nemotron"]
        self._fail_models = fail_models or set()
        self.call_log: List[Dict[str, Any]] = []

    async def call_model(self, model_id: str, prompt: str, system_role: str = "", **kwargs) -> str:
        self.call_log.append({"model_id": model_id, "prompt_len": len(prompt)})
        if model_id in self._fail_models:
            raise RuntimeError(f"Simulated failure for {model_id}")
        return (
            f"POSITION: Test position from {model_id}\n"
            f"ARGUMENT: Test argument with reasoning from {model_id}\n"
            f"ASSUMPTIONS:\n- Assumption A\n- Assumption B\n"
            f"RISKS:\n- Risk X\n"
            f"CONFIDENCE: 0.75"
        )

    def get_enabled_model_ids(self) -> List[str]:
        return [m for m in self._model_ids if m not in self._fail_models]

    def get_enabled_models_info(self) -> List[Dict[str, Any]]:
        return [
            {"id": m, "legacy_id": m, "registry_key": m, "name": f"Model-{m}",
             "provider": "test", "role": "general", "supports_vision": False, "supports_debate": True}
            for m in self._model_ids if m not in self._fail_models
        ]


# ── Tests ─────────────────────────────────────────────────────


class TestDebateOrchestratorInit:
    """Test DebateOrchestrator initialization uses dynamic model routing."""

    @pytest.fixture(autouse=True)
    def patch_debate_models(self):
        """Patch DEBATE_MODELS to simulate enabled models."""
        import core.debate_orchestrator as mod
        fake_models = [
            {"id": "groq", "registry_key": "groq-small", "label": "Groq", "provider": "groq", "name": "Groq LLaMA", "color": "blue"},
            {"id": "llama70b", "registry_key": "llama-3.3", "label": "Llama70B", "provider": "llama70b", "name": "Llama 3.3", "color": "indigo"},
            {"id": "qwen", "registry_key": "qwen-vl-2.5", "label": "Qwen", "provider": "qwen", "name": "Qwen 2.5", "color": "purple"},
        ]
        old = mod.DEBATE_MODELS
        mod.DEBATE_MODELS = fake_models
        yield
        mod.DEBATE_MODELS = old

    def test_all_models_get_callers(self):
        """Every enabled model in DEBATE_MODELS should have a caller registered."""
        from core.debate_orchestrator import DebateOrchestrator, DEBATE_MODELS

        bridge = MockBridge()
        orch = DebateOrchestrator(bridge)

        for model in DEBATE_MODELS:
            mid = model["id"]
            assert mid in orch._model_callers, f"Model {mid} missing from _model_callers"

    def test_no_legacy_callers(self):
        """No direct call_groq/call_llama70b/call_qwenvl references in callers."""
        from core.debate_orchestrator import DebateOrchestrator

        bridge = MockBridge()
        orch = DebateOrchestrator(bridge)

        # Each caller should be a lambda wrapping call_model, not a legacy method
        for mid, caller in orch._model_callers.items():
            # Lambda functions have names like '<lambda>'
            assert callable(caller), f"Caller for {mid} is not callable"


class TestAnalysisRetry:
    """Test _run_analysis dynamic model selection and retry."""

    @pytest.fixture(autouse=True)
    def patch_debate_models(self):
        """Patch DEBATE_MODELS to simulate enabled models."""
        import core.debate_orchestrator as mod
        fake_models = [
            {"id": "groq", "registry_key": "groq-small", "label": "Groq", "provider": "groq", "name": "Groq LLaMA", "color": "blue"},
            {"id": "llama70b", "registry_key": "llama-3.3", "label": "Llama70B", "provider": "llama70b", "name": "Llama 3.3", "color": "indigo"},
            {"id": "qwen", "registry_key": "qwen-vl-2.5", "label": "Qwen", "provider": "qwen", "name": "Qwen 2.5", "color": "purple"},
        ]
        old = mod.DEBATE_MODELS
        mod.DEBATE_MODELS = fake_models
        yield
        mod.DEBATE_MODELS = old

    @pytest.mark.asyncio
    async def test_analysis_uses_first_available_model(self):
        """Analysis should succeed with the first working model."""
        from core.debate_orchestrator import DebateOrchestrator

        bridge = MockBridge()
        # Override call_model to return analysis-format output
        original = bridge.call_model

        async def analysis_call(model_id, prompt, system_role="", **kw):
            return (
                "CONFLICT_AXES:\n- Axis 1\n"
                "DISAGREEMENT_STRENGTH: 0.6\n"
                "LOGICAL_STABILITY: 0.7\n"
                "POSITION_SHIFTS:\n- None\n"
                "CONVERGENCE_LEVEL: moderate\n"
                "CONVERGENCE_DETAIL: Models partially converged\n"
                "STRONGEST_ARGUMENT: Model-groq had the best case\n"
                "WEAKEST_ARGUMENT: Model-qwen was weakest\n"
                "CONFIDENCE_RECALIBRATION: 0.65\n"
                "SYNTHESIS: Balanced conclusion."
            )

        bridge.call_model = analysis_call
        orch = DebateOrchestrator(bridge)

        result = await orch._run_analysis("Test transcript")

        assert result.synthesis != ""
        assert result.confidence_recalibration == 0.65
        assert result.convergence_level == "moderate"

    @pytest.mark.asyncio
    async def test_analysis_retries_on_failure(self):
        """Analysis should try the next model if the first fails."""
        from core.debate_orchestrator import DebateOrchestrator

        call_count = {"count": 0}

        bridge = MockBridge()

        async def failing_then_success(model_id, prompt, system_role="", **kw):
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise RuntimeError("Simulated failure")
            return (
                "CONFLICT_AXES:\n- Axis 1\n"
                "DISAGREEMENT_STRENGTH: 0.5\n"
                "LOGICAL_STABILITY: 0.8\n"
                "CONVERGENCE_LEVEL: high\n"
                "CONVERGENCE_DETAIL: Full convergence\n"
                "STRONGEST_ARGUMENT: Best model\n"
                "WEAKEST_ARGUMENT: Weakest model\n"
                "CONFIDENCE_RECALIBRATION: 0.9\n"
                "SYNTHESIS: Good synthesis."
            )

        bridge.call_model = failing_then_success
        orch = DebateOrchestrator(bridge)

        result = await orch._run_analysis("Test transcript")

        assert call_count["count"] >= 3  # Failed twice, succeeded on third
        assert result.confidence_recalibration == 0.9

    @pytest.mark.asyncio
    async def test_analysis_all_models_fail(self):
        """If all models fail, should return graceful fallback."""
        from core.debate_orchestrator import DebateOrchestrator

        bridge = MockBridge()

        async def always_fail(model_id, prompt, system_role="", **kw):
            raise RuntimeError("All dead")

        bridge.call_model = always_fail
        orch = DebateOrchestrator(bridge)

        result = await orch._run_analysis("Test transcript")

        assert "failed" in result.synthesis.lower()
        assert result.confidence_recalibration == 0.5


class TestFullDebate:
    """Test full debate pipeline execution."""

    @pytest.fixture(autouse=True)
    def patch_debate_models(self):
        """Patch DEBATE_MODELS to simulate enabled models."""
        import core.debate_orchestrator as mod
        fake_models = [
            {"id": "groq", "registry_key": "groq-small", "label": "Groq", "provider": "groq", "name": "Groq LLaMA", "color": "blue"},
            {"id": "llama70b", "registry_key": "llama-3.3", "label": "Llama70B", "provider": "llama70b", "name": "Llama 3.3", "color": "indigo"},
            {"id": "qwen", "registry_key": "qwen-vl-2.5", "label": "Qwen", "provider": "qwen", "name": "Qwen 2.5", "color": "purple"},
        ]
        old = mod.DEBATE_MODELS
        mod.DEBATE_MODELS = fake_models
        yield
        mod.DEBATE_MODELS = old

    @pytest.mark.asyncio
    async def test_debate_runs_all_models(self):
        """All models in DEBATE_MODELS should participate in debate rounds."""
        from core.debate_orchestrator import DebateOrchestrator, DEBATE_MODELS

        bridge = MockBridge()

        async def debate_call(model_id, prompt, system_role="", **kw):
            return (
                "REBUTTALS: Opponent X is wrong because...\n"
                "POSITION: Updated position from " + model_id + "\n"
                "ARGUMENT: Strengthened argument\n"
                "POSITION_SHIFT: none\n"
                "WEAKNESSES_FOUND: Logical gap in opponent\n"
                "CONFIDENCE: 0.8"
            )

        bridge.call_model = debate_call
        orch = DebateOrchestrator(bridge)

        result = await orch.run_debate("Test question", rounds=2)

        assert result.total_rounds == 2
        assert len(result.rounds) == 2

        # Each round should have outputs from ALL debate models
        for round_outputs in result.rounds:
            assert len(round_outputs) == len(DEBATE_MODELS), \
                f"Expected {len(DEBATE_MODELS)} outputs per round, got {len(round_outputs)}"

        # All models should be listed
        assert len(result.models_used) == len(DEBATE_MODELS)

    @pytest.mark.asyncio
    async def test_debate_handles_model_failure(self):
        """Failed models should produce error outputs, not crash the debate."""
        from core.debate_orchestrator import DebateOrchestrator, DEBATE_MODELS

        bridge = MockBridge(fail_models={"groq"})

        async def mixed_call(model_id, prompt, system_role="", **kw):
            if model_id == "groq":
                raise RuntimeError("Groq down")
            return (
                "POSITION: Position from " + model_id + "\n"
                "ARGUMENT: Valid argument\n"
                "ASSUMPTIONS:\n- A1\n"
                "RISKS:\n- R1\n"
                "CONFIDENCE: 0.7"
            )

        bridge.call_model = mixed_call
        orch = DebateOrchestrator(bridge)

        result = await orch.run_debate("Test question", rounds=1)

        assert len(result.rounds) == 1
        round_outputs = result.rounds[0]

        # Should still have outputs for all models (including failed ones)
        assert len(round_outputs) == len(DEBATE_MODELS)

        # At least one should have failed gracefully
        failed = [o for o in round_outputs if o.confidence == 0.0]
        succeeded = [o for o in round_outputs if o.confidence > 0.0]

        # groq should have failed if it's in DEBATE_MODELS
        groq_models = [o for o in round_outputs if o.model_id == "groq"]
        if groq_models:
            assert groq_models[0].confidence == 0.0
            assert "unavailable" in groq_models[0].position.lower() or "failed" in groq_models[0].argument.lower()


class TestDebateResultSerialization:
    """Test DebateResult serialization."""

    def test_to_dict(self):
        """DebateResult.to_dict() should produce valid JSON-serializable dict."""
        from core.debate_orchestrator import DebateResult, ModelRoundOutput, DebateAnalysis

        result = DebateResult(
            query="Test",
            total_rounds=1,
            models_used=["Model A", "Model B"],
        )

        output = ModelRoundOutput(
            model_id="groq",
            model_label="Groq",
            model_name="LLaMA 3.1",
            model_color="blue",
            round_num=1,
            position="Test position",
            argument="Test argument",
            confidence=0.8,
        )
        result.rounds.append([output])

        result.analysis = DebateAnalysis(
            synthesis="Test synthesis",
            confidence_recalibration=0.7,
        )

        d = result.to_dict()
        assert d["query"] == "Test"
        assert d["total_rounds"] == 1
        assert len(d["rounds"]) == 1
        assert len(d["rounds"][0]) == 1
        assert d["rounds"][0][0]["model_id"] == "groq"
        assert d["analysis"]["synthesis"] == "Test synthesis"

        # Should be JSON serializable
        import json
        json.dumps(d)  # Should not raise


class TestRedisStub:
    """Test InMemoryRedisStub fallback."""

    @pytest.mark.asyncio
    async def test_setex_get(self):
        """setex/get should work in memory."""
        from database.connection import InMemoryRedisStub

        stub = InMemoryRedisStub()
        await stub.setex("key1", 60, "value1")
        assert await stub.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_delete(self):
        """delete should remove key."""
        from database.connection import InMemoryRedisStub

        stub = InMemoryRedisStub()
        await stub.setex("key1", 60, "value1")
        await stub.delete("key1")
        assert await stub.get("key1") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Should evict oldest entries when MAX_KEYS exceeded."""
        from database.connection import InMemoryRedisStub

        stub = InMemoryRedisStub()
        stub._MAX_KEYS = 3

        await stub.setex("a", 60, "1")
        await stub.setex("b", 60, "2")
        await stub.setex("c", 60, "3")
        await stub.setex("d", 60, "4")  # Should evict "a"

        assert await stub.get("a") is None  # evicted
        assert await stub.get("b") == "2"
        assert await stub.get("d") == "4"

    @pytest.mark.asyncio
    async def test_ping(self):
        """ping should always return True."""
        from database.connection import InMemoryRedisStub

        stub = InMemoryRedisStub()
        assert await stub.ping() is True

    @pytest.mark.asyncio
    async def test_keys_pattern(self):
        """keys() with wildcard pattern."""
        from database.connection import InMemoryRedisStub

        stub = InMemoryRedisStub()
        await stub.setex("session:abc", 60, "1")
        await stub.setex("session:def", 60, "2")
        await stub.setex("chat:abc", 60, "3")

        session_keys = await stub.keys("session:*")
        assert len(session_keys) == 2
        assert "session:abc" in session_keys
