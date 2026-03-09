"""Validate that OpenRouter has been completely removed from the runtime system."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))

from dotenv import load_dotenv
load_dotenv('.env')

errors = []

# 1. Check OpenRouter key is commented out
or_key = os.getenv('OPENROUTER_API_KEY', '')
if or_key:
    errors.append(f"OPENROUTER_API_KEY still active: {or_key[:8]}...")
else:
    print("OK: OPENROUTER_API_KEY is commented out")

# 2. Import and check registry
from metacognitive.cognitive_gateway import (
    COGNITIVE_MODEL_REGISTRY, MODEL_DEBATE_TIERS, MODEL_FALLBACK_MAP,
    get_tiered_models_for_debate,
)

print(f"OK: Registry has {len(COGNITIVE_MODEL_REGISTRY)} models")
for key, spec in COGNITIVE_MODEL_REGISTRY.items():
    print(f"  {key}: provider={spec.provider}, model={spec.model_id}, enabled={spec.enabled}")

# 3. Check NO OpenRouter providers remain
or_models = [k for k, s in COGNITIVE_MODEL_REGISTRY.items() if s.provider == 'openrouter']
if or_models:
    errors.append(f"OpenRouter models still in registry: {or_models}")
else:
    print("OK: No OpenRouter models in registry")

# 4. Check old model keys are gone
old_keys = ['gemma2-9b', 'mistral-7b', 'phi3-mini', 'gemma2-2b', 'phi3-small']
remaining = [k for k in old_keys if k in COGNITIVE_MODEL_REGISTRY]
if remaining:
    errors.append(f"Old model keys still present: {remaining}")
else:
    print("OK: All old OpenRouter model keys removed")

# 5. Check new model keys exist
new_keys = ['llama31-8b', 'mixtral-8x7b', 'gemma-7b', 'qwen-2.5-vl', 'gemini-flash', 'llama31-instant']
missing = [k for k in new_keys if k not in COGNITIVE_MODEL_REGISTRY]
if missing:
    errors.append(f"New model keys missing: {missing}")
else:
    print("OK: All 6 new model keys present")

# 6. Check debate tiers cover all models
uncovered = [k for k in COGNITIVE_MODEL_REGISTRY if k not in MODEL_DEBATE_TIERS]
if uncovered:
    errors.append(f"Models missing from debate tiers: {uncovered}")
else:
    print("OK: All models have tier assignments")

# 7. Check fallback map covers all models
no_fallback = [k for k in COGNITIVE_MODEL_REGISTRY if k not in MODEL_FALLBACK_MAP]
if no_fallback:
    errors.append(f"Models missing from fallback map: {no_fallback}")
else:
    print("OK: All models have fallback assignments")

# 8. Check model_registry.py
from core.model_registry import get_all_models, get_enabled_models
all_m = get_all_models()
enabled_m = get_enabled_models()
print(f"OK: model_registry: {len(all_m)} total, {len(enabled_m)} enabled")

# 9. Check mco_bridge
from models.mco_bridge import LEGACY_TO_REGISTRY
# Ensure no legacy mapping points to a removed key
for legacy_id, reg_key in LEGACY_TO_REGISTRY.items():
    if reg_key not in COGNITIVE_MODEL_REGISTRY:
        errors.append(f"mco_bridge: '{legacy_id}' maps to missing key '{reg_key}'")
print(f"OK: mco_bridge has {len(LEGACY_TO_REGISTRY)} legacy mappings")

# 10. Check ensemble_schemas
from core.ensemble_schemas import MAX_DEBATE_MODELS
if MAX_DEBATE_MODELS != 6:
    errors.append(f"MAX_DEBATE_MODELS is {MAX_DEBATE_MODELS}, expected 6")
else:
    print(f"OK: MAX_DEBATE_MODELS = {MAX_DEBATE_MODELS}")

# 11. Check tiered model selection works
selected = get_tiered_models_for_debate("general")
print(f"OK: get_tiered_models_for_debate('general') = {selected}")

# 12. Check gateway config
from gateway.config import get_settings
settings = get_settings()
if hasattr(settings, 'OPENROUTER_API_KEY') and settings.OPENROUTER_API_KEY:
    errors.append("gateway.config still has active OPENROUTER_API_KEY")

# Summary
print()
if errors:
    print(f"FAILED with {len(errors)} errors:")
    for e in errors:
        print(f"  ERROR: {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED - OpenRouter completely removed from runtime.")
    sys.exit(0)
