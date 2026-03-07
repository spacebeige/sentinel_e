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
    legacy = [k for k in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b')
              if k in COGNITIVE_MODEL_REGISTRY]
    if legacy:
        print(f"   ✗ Legacy models still in registry: {legacy}")
        ok = False
    else:
        print("   ✓ No legacy model refs in registry")
except Exception as e:
    print(f"   ✗ {e}")
    ok = False

# ── model_registry ────────────────────────────────────────────
print("4. core/model_registry.py")
try:
    from core.model_registry import _LEGACY_ID_MAP, _TIER_MAP, _COST_MAP
    print(f"   _LEGACY_ID_MAP: {sorted(_LEGACY_ID_MAP.keys())}")
    print(f"   _TIER_MAP: {sorted(_TIER_MAP.keys())}")
    bad = [k for k in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b')
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
    bad = [k for k in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b')
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
                   if v in ('qwen-vl-2.5','nemotron-30b-free','mistral-small-24b','llama-3.2-3b')]
    if bad_targets:
        print(f"   ✗ Entries targeting removed models: {bad_targets}")
        ok = False
    else:
        print("   ✓ All legacy IDs map to active models")
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
