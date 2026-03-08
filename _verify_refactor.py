#!/usr/bin/env python3
"""Stability verification — confirms all refactored modules are clean."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

ok = True

# ── chat_routes ───────────────────────────────────────────────
print("1. gateway/chat_routes.py")
try:
    from gateway.chat_routes import router as chat_router
    paths = [r.path for r in chat_router.routes]
    print(f"   ✓ {len(paths)} routes: {paths}")
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── evaluation/routes.py ──────────────────────────────────────
print("2. evaluation/routes.py")
try:
    from evaluation.routes import router as battle_router
    paths = [r.path for r in battle_router.routes]
    debate = [p for p in paths if 'debate' in p]
    print(f"   ✓ {len(paths)} routes total; debate: {debate}")
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── cognitive_gateway ─────────────────────────────────────────
print("3. cognitive_gateway ensemble")
try:
    from metacognitive.cognitive_gateway import (
        COGNITIVE_MODEL_REGISTRY, MODEL_DEBATE_TIERS, get_tiered_models_for_debate
    )
    print(f"   Registry ({len(COGNITIVE_MODEL_REGISTRY)}): {sorted(COGNITIVE_MODEL_REGISTRY.keys())}")
    print(f"   Tiers: {MODEL_DEBATE_TIERS}")
    sel = get_tiered_models_for_debate()
    print(f"   Default selection (general): {sel}")
    legacy = [k for k in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b',
                           'qwen-2.5-32b','deepseek-chat','deepseek-coder','mixtral-8x7b')
              if k in COGNITIVE_MODEL_REGISTRY]
    if legacy:
        print(f"   ✗ Legacy models still in registry: {legacy}")
        ok = False
    else:
        print("   ✓ No legacy model refs in registry")
    # Verify new models present
    expected_new = {'llama31-8b', 'gemma2-9b', 'mistral-7b', 'phi3-mini',
                    'gemma2-2b', 'llama31-instant', 'phi3-small'}
    present = expected_new & set(COGNITIVE_MODEL_REGISTRY.keys())
    missing = expected_new - present
    if missing:
        print(f"   ✗ Missing new models: {missing}")
        ok = False
    else:
        print(f"   ✓ All 7 free-tier models present")
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── model_registry ────────────────────────────────────────────
print("4. core/model_registry.py")
try:
    from core.model_registry import _LEGACY_ID_MAP, _TIER_MAP, _COST_MAP
    print(f"   _LEGACY_ID_MAP: {sorted(_LEGACY_ID_MAP.keys())}")
    print(f"   _TIER_MAP: {sorted(_TIER_MAP.keys())}")
    bad = [k for k in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b',
                       'qwen-2.5-32b','deepseek-chat','deepseek-coder','mixtral-8x7b')
           if k in _TIER_MAP]
    if bad:
        print(f"   ✗ Legacy entries in _TIER_MAP: {bad}")
        ok = False
    else:
        print("   ✓ No legacy entries in _TIER_MAP")
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── cost_governor ─────────────────────────────────────────────
print("5. optimization/cost_governor.py")
try:
    from optimization.cost_governor import MODEL_COSTS
    bad = [k for k in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b',
                       'qwen-2.5-32b','deepseek-chat','deepseek-coder','mixtral-8x7b')
           if k in MODEL_COSTS]
    if bad:
        print(f"   ✗ Legacy entries in MODEL_COSTS: {bad}")
        ok = False
    else:
        print(f"   ✓ MODEL_COSTS clean ({sorted(MODEL_COSTS.keys())})")
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── mco_bridge ────────────────────────────────────────────────
print("6. models/mco_bridge.py")
try:
    from models.mco_bridge import LEGACY_TO_REGISTRY
    print(f"   LEGACY_TO_REGISTRY: {LEGACY_TO_REGISTRY}")
    bad_targets = [k for k,v in LEGACY_TO_REGISTRY.items()
                   if v in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b',
                            'qwen-2.5-32b','deepseek-chat','deepseek-coder','mixtral-8x7b')]
    if bad_targets:
        print(f"   ✗ Entries targeting removed models: {bad_targets}")
        ok = False
    else:
        print("   ✓ All legacy IDs map to active models")
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

print()

# ── reasoning analytics ──────────────────────────────────────
print("7. analysis/reasoning_analytics.py")
try:
    from analysis.reasoning_analytics import ReasoningAnalyticsEngine, ReasoningStepExtractor
    engine = ReasoningAnalyticsEngine()
    steps = ReasoningStepExtractor.extract_steps(
        "First we identify the claim, then analyze evidence from multiple sources, "
        "compare alternative approaches, and conclude that the hypothesis holds."
    )
    if len(steps) >= 3:
        print(f"   ✓ ReasoningStepExtractor works ({len(steps)} steps extracted)")
    else:
        print(f"   ✗ Expected ≥3 steps, got {len(steps)}")
        ok = False
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── evidence_debate_pipeline ─────────────────────────────────
print("8. core/evidence_debate_pipeline.py")
try:
    from core.evidence_debate_pipeline import EvidenceContext, gather_evidence
    ctx = EvidenceContext(summary="test", sources=[], source_count=0)
    prompt = ctx.to_prompt_injection()
    if prompt == "":
        print("   ✓ EvidenceContext works (empty sources → empty prompt)")
    else:
        print(f"   ✗ Unexpected prompt: {prompt[:80]}")
        ok = False
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── BattleVisualizationPayload analytics field ───────────────
print("9. ensemble_schemas.py — reasoning_analytics field")
try:
    from core.ensemble_schemas import BattleVisualizationPayload
    p = BattleVisualizationPayload(prompt="test", models_selected=["a"])
    if hasattr(p, 'reasoning_analytics'):
        print("   ✓ reasoning_analytics field present on BattleVisualizationPayload")
    else:
        print("   ✗ reasoning_analytics field missing")
        ok = False
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

print()
print("=" * 50)
if ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
