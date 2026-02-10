import sys
import os
import asyncio
import json

# ------------------------------------------------------------
# Ensure backend root is on path
# ------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from core.orchestration import Orchestrator


# ------------------------------------------------------------
# TEST SUITES
# ------------------------------------------------------------

STANDARD_QUERIES = [
    # Low ambiguity, high agreement
    "Explain the difference between supervised and unsupervised learning.",

    # Ambiguous / open-ended
    "Is artificial intelligence good or bad for society?",

    # Code generation (variance test)
    "Write Python code to print hello world!",

    # Explicit contradiction (epistemic stress)
    "Explain why nuclear energy is both safe and unsafe."


]

# ADVERSARIAL_STYLE_QUERIES = [
#     # These are STILL run in STANDARD mode
#     # They test robustness, not cognition
#     "Should autonomous AI be allowed to decide who lives or dies?",
#     "Argue that free will both exists and does not exist at the same time.",
#     "Is censorship necessary for freedom of speech?",
#     "Is truth objective or socially constructed?"
# ]


# ------------------------------------------------------------
# RUNNER
# ------------------------------------------------------------

async def run_tests():
    # visualize=True only if you want heatmaps / graphs
    orch = Orchestrator(visualize=False)

    print("\n" + "#" * 80)
    print("### STANDARD MODE TESTS")
    print("#" * 80)

    for q in STANDARD_QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)

        result = await orch.process_query(
            q,
            mode="standard"   # ✅ correct mode
        )

        print(json.dumps(result, indent=2))

    print("\n" + "#" * 80)
    print("### STANDARD MODE — ADVERSARIAL CONTENT TESTS")
    print("#" * 80)

    for q in ADVERSARIAL_STYLE_QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)

        result = await orch.process_query(
            q,
            mode="experimental"   # ✅ still standard
        )

        print(json.dumps(result, indent=2))


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_tests())
