"""Validate adversarial prompt wiring across both debate engines."""
import sys
sys.path.insert(0, ".")

from core.ensemble_schemas import DebatePosition, DebateResult
from core.structured_debate_engine import StructuredDebateEngine, STRUCTURED_ROUND_1, STRUCTURED_ROUND_N
from core.debate_orchestrator import DebateOrchestrator, ROUND_1_SYSTEM, ROUND_N_SYSTEM, ANALYSIS_SYSTEM, ROLE_INSTRUCTIONS

errors = []

# --- Schema checks ---
try:
    dp = DebatePosition(
        model_id="test", model_name="Test", round_number=1,
        position="test", argument="test",
        risks=["risk1"], weaknesses_found=["w1", "w2"],
    )
    assert isinstance(dp.risks, list), "risks should be list"
    assert isinstance(dp.weaknesses_found, list), "weaknesses_found should be list"
    assert len(dp.weaknesses_found) == 2
    print("[OK] DebatePosition schema accepts risks + weaknesses_found as lists")
except Exception as e:
    errors.append(f"Schema: {e}")

# --- Structured debate engine prompts ---
try:
    assert "adversarial" in STRUCTURED_ROUND_1.lower(), "ROUND_1 missing adversarial"
    assert "RISKS" in STRUCTURED_ROUND_1, "ROUND_1 missing RISKS"
    assert "ARGUMENT" in STRUCTURED_ROUND_1, "ROUND_1 missing ARGUMENT"
    assert "ASSUMPTIONS" in STRUCTURED_ROUND_1, "ROUND_1 missing ASSUMPTIONS"
    print("[OK] STRUCTURED_ROUND_1 has adversarial format")
except Exception as e:
    errors.append(f"STRUCTURED_ROUND_1: {e}")

try:
    assert "WEAKNESSES_FOUND" in STRUCTURED_ROUND_N, "ROUND_N missing WEAKNESSES_FOUND"
    assert "ARGUMENT" in STRUCTURED_ROUND_N, "ROUND_N missing ARGUMENT"
    assert "REBUTTALS" in STRUCTURED_ROUND_N, "ROUND_N missing REBUTTALS"
    assert "POSITION_SHIFTED" in STRUCTURED_ROUND_N, "ROUND_N missing POSITION_SHIFTED"
    assert "SHIFT_REASON" in STRUCTURED_ROUND_N, "ROUND_N missing SHIFT_REASON"
    print("[OK] STRUCTURED_ROUND_N has adversarial rebuttal format")
except Exception as e:
    errors.append(f"STRUCTURED_ROUND_N: {e}")

# --- Debate orchestrator prompts ---
try:
    assert "adversarial" in ROUND_1_SYSTEM.lower(), "orchestrator ROUND_1 missing adversarial"
    assert "ARGUMENT" in ROUND_1_SYSTEM, "orchestrator ROUND_1 missing ARGUMENT"
    assert "RISKS" in ROUND_1_SYSTEM, "orchestrator ROUND_1 missing RISKS"
    print("[OK] ROUND_1_SYSTEM (orchestrator) has adversarial format")
except Exception as e:
    errors.append(f"ROUND_1_SYSTEM: {e}")

try:
    assert "WEAKNESSES_FOUND" in ROUND_N_SYSTEM, "orchestrator ROUND_N missing WEAKNESSES_FOUND"
    assert "POSITION_SHIFT" in ROUND_N_SYSTEM, "orchestrator ROUND_N missing POSITION_SHIFT"
    print("[OK] ROUND_N_SYSTEM (orchestrator) has adversarial format")
except Exception as e:
    errors.append(f"ROUND_N_SYSTEM: {e}")

try:
    assert "DRIFT_ESTIMATE" in ANALYSIS_SYSTEM, "ANALYSIS missing DRIFT"
    assert "RIFT_ESTIMATE" in ANALYSIS_SYSTEM, "ANALYSIS missing RIFT"
    assert "SYNTHESIS" in ANALYSIS_SYSTEM, "ANALYSIS missing SYNTHESIS"
    print("[OK] ANALYSIS_SYSTEM has drift/rift/synthesis format")
except Exception as e:
    errors.append(f"ANALYSIS_SYSTEM: {e}")

# --- Role instructions ---
try:
    assert "PROPONENT" in ROLE_INSTRUCTIONS["for"]
    assert "OPPONENT" in ROLE_INSTRUCTIONS["against"]
    assert "JUDGE" in ROLE_INSTRUCTIONS["judge"]
    assert "INDEPENDENT ANALYST" in ROLE_INSTRUCTIONS["neutral"]
    print("[OK] ROLE_INSTRUCTIONS are adversarial/personality-driven")
except Exception as e:
    errors.append(f"ROLE_INSTRUCTIONS: {e}")

# --- Summary ---
if errors:
    print(f"\nFAILED ({len(errors)} errors):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nALL PROMPT WIRING TESTS PASSED")
