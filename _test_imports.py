#!/usr/bin/env python3
"""Quick import validation for compressed pipeline v2."""
import sys
sys.path.insert(0, "backend")

errors = []

try:
    from compressed.model_clients import RoleBasedRouter, GroqClient, GeminiFlashClient, QwenClient, MODELS_REGISTRY
    print("OK model_clients")
    print(f"   Models: {list(MODELS_REGISTRY.keys())}")
except Exception as e:
    errors.append(f"model_clients: {e}")
    print(f"FAIL model_clients: {e}")

try:
    from compressed.search_engine import DualSearchEngine, TavilySearch, SerperSearch
    print("OK search_engine")
except Exception as e:
    errors.append(f"search_engine: {e}")
    print(f"FAIL search_engine: {e}")

try:
    from compressed.role_engine import RoleEngine, ReasoningState
    print("OK role_engine")
except Exception as e:
    errors.append(f"role_engine: {e}")
    print(f"FAIL role_engine: {e}")

try:
    from compressed.token_governor import TokenGovernor, TokenBudget, MAX_PROMPT_TOKENS
    print(f"OK token_governor (MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS})")
except Exception as e:
    errors.append(f"token_governor: {e}")
    print(f"FAIL token_governor: {e}")

try:
    from compressed.report_generator import ReportGenerator
    print("OK report_generator")
except Exception as e:
    errors.append(f"report_generator: {e}")
    print(f"FAIL report_generator: {e}")

try:
    from compressed.pipeline import run_compressed_pipeline, PipelineState
    print("OK pipeline")
except Exception as e:
    errors.append(f"pipeline: {e}")
    print(f"FAIL pipeline: {e}")

try:
    router = RoleBasedRouter()
    models = router.list_models()
    print(f"OK RoleBasedRouter ({len(models)} models)")
    for m in models:
        c = router.get_client_by_id(m["id"])
        avail = c.available if c else "NOT FOUND"
        print(f"   {m['id']}: available={avail}")
except Exception as e:
    errors.append(f"RoleBasedRouter: {e}")
    print(f"FAIL RoleBasedRouter: {e}")

# Test role routing
try:
    import asyncio
    async def test_roles():
        r = RoleBasedRouter()
        for role in ["analysis", "synthesis", "critique_a", "critique_b", "critique_c", "verification", "summarize"]:
            resp = await r.generate(role=role, prompt="test", max_tokens=10)
            assert resp is not None, f"None response for {role}"
        print("OK role routing (all 7 roles respond)")
    asyncio.run(test_roles())
except Exception as e:
    errors.append(f"role routing: {e}")
    print(f"FAIL role routing: {e}")

# Test individual model clients
try:
    for mid in ["llama-3.3-70b", "llama-3.1-8b", "mixtral-8x7b", "gemma-7b", "gemini-flash", "qwen-2.5-vl"]:
        client = router.get_client_by_id(mid)
        assert client is not None, f"client not found for {mid}"
        assert hasattr(client, "generate"), f"{mid} missing generate()"
        assert hasattr(client, "available"), f"{mid} missing available"
    print("OK individual model lookup (all 6 models)")
except Exception as e:
    errors.append(f"model lookup: {e}")
    print(f"FAIL model lookup: {e}")

if errors:
    print(f"\n{len(errors)} FAILURES")
    sys.exit(1)
else:
    print("\nAll imports and checks OK")
