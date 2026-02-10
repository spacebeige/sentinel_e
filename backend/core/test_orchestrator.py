
import sys
import os

# Add PROJECT ROOT (PolyMath/backend) to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import asyncio
from orchestration import Orchestrator

# async def test():
#     orch = Orchestrator()
#     result = await orch.process_query(
#         "Explain the difference between supervised and unsupervised learning."
#     )
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(test())

# ============================================================
import sys
import os
import asyncio

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the upgraded Orchestrator
# Note: Ensure the file name matches 'sentinel_orchestrator.py'
from orchestration import Orchestrator

async def run_diagnostic():
    print("--- Sentinel-X V2.0 Diagnostic Starting ---")
    orch = Orchestrator()
    
    # Test 1: Standard Reasoning
    print("\n[Test 1] Standard Query (Supervised vs Unsupervised)")
    result = await orch.process_query(
        "Explain the difference between supervised and unsupervised learning."
    )
    
    print(f"Mode Used: {result.get('mode')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Answer Sample: {result.get('aggregate_answer')[:150]}...")

    # Test 2: Dialectic Mode (Complex Debate)
    print("\n[Test 2] Dialectic Mode (AI Ethics)")
    dialectic_result = await orch.process_query(
        "Should AI be given legal personhood?",
        mode="dialectic"
    )
    print(f"Synthesis Complete: {dialectic_result.get('executive_decision')}")
    print(f"Synthesized Answer: {dialectic_result.get('aggregate_answer')[:150]}...")

    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    try:
        asyncio.run(run_diagnostic())
    except Exception as e:
        print(f"FATAL: Test Failed with error: {e}")