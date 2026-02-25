"""Quick verification of the optimization layer."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimization import (
    get_token_optimizer,
    get_response_cache,
    get_fallback_router,
    get_cost_governor,
    get_observability_hub,
)

def main():
    # Verify singletons create without error
    to = get_token_optimizer()
    rc = get_response_cache()
    fr = get_fallback_router()
    cg = get_cost_governor()
    oh = get_observability_hub()
    print("[OK] All 5 optimization singletons initialized")

    # Token optimizer
    result = to.optimize(
        query="What is 2+2?",
        system_prompt="You are helpful.",
        history=[{"role": "user", "content": "What is 2+2?"}],
        context_window=4096,
    )
    depth = result.get("depth_assessment")
    print(f"[OK] TokenOptimizer: complexity={depth.complexity}, compression={result.get('compression_applied')}")

    # Response cache
    rc.store("What is 2+2?", "standard", {"formatted_output": "Four", "confidence": 0.9})
    hit = rc.lookup("What is 2+2?", "standard")
    assert hit.hit, "Cache hit expected"
    print("[OK] ResponseCache: store + lookup (exact match)")

    stats = rc.stats
    assert stats["tier_1"]["hits"] >= 1
    print(f"[OK] ResponseCache stats: {stats['tier_1']['hits']} exact hits")

    # Cost governor
    decision = cg.check_budget("test-session", "standard")
    assert decision.allowed, "Budget should be allowed"
    cg.record_usage("test-session", "llama-3.1-8b", 100, 50)
    budget = cg.get_session_budget("test-session")
    assert budget["total_tokens"] == 150
    print("[OK] CostGovernor: budget check + usage recording")

    # Observability
    tracer = oh.start_request("test-session", "req-001")
    tracer.start_span("test")
    time.sleep(0.01)
    lat = tracer.end_span("test")
    assert lat > 0
    tracer.record_model_call("llama-3.1-8b", 100, 50, 150.0, 0.0001)
    summary = tracer.finalize()
    oh.record(summary)
    m = oh.get_metrics()
    assert m["counters"]["total_requests"] >= 1
    print("[OK] Observability: tracer + metrics aggregation")

    # Memory check
    import resource
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports in bytes, Linux in KB
    mem_mb = mem_kb / (1024 * 1024) if sys.platform == "darwin" else mem_kb / 1024
    print(f"[OK] Memory after optimization layer: {mem_mb:.1f} MB")

    print()
    print("ALL OPTIMIZATION MODULES VERIFIED")


if __name__ == "__main__":
    main()
