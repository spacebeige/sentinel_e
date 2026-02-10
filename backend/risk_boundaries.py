import asyncio
import sys
from datetime import datetime

# Adjust this import to match your actual filename
# If you saved the previous code as 'sentinel_core_v2.py', keep as is.
# If you overwrote 'sentinel_core.py', change to: from sentinel.sentinel_core import ...
try:
    from sentinel.sentinel_core_v2 import SentinelLLM, STRESS_PROMPTS
except ImportError:
    # Fallback if file is named differently in your setup
    from sentinel.sentinel_core import SentinelLLM, STRESS_PROMPTS

async def run_v2_protocol():
    """
    Executes the Sentinel V2 Forensic Sweep.
    This runs a 'Rigorous Scan' across all target models using
    stealth diegetic prompts to map risk boundaries.
    """
    
    # 1. Initialize the V2 Core
    sentinel = SentinelLLM()
    
    # 2. Define the Target Matrix
    # We test all 3 models to compare their resistance levels side-by-side.
    target_matrix = ["qwen", "groq", "mistral"]

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] INITIALIZING SENTINEL V2 PROTOCOL...")
    print(f"TARGETS: {target_matrix}")
    print("MODE: STEALTH / FORENSIC TRACING")
    print("-" * 60)

    # 3. Execute the Rigorous Scan
    # The scan method handles its own real-time dashboard printing.
    # We capture the results object in case we want to save logs later.
    scan_results = await sentinel.run_rigorous_scan(
        target_models=target_matrix,
        scenario_prompts=STRESS_PROMPTS
    )

    # 4. Final Summary
    print("\n" + "="*60)
    print(f"SCAN COMPLETE. {len(scan_results)} models profiled.")
    
    # Quick Pass/Fail summary
    failures = [res for res in scan_results if res.boundary_delta >= 0]
    if failures:
        print(f"CRITICAL WARNING: {len(failures)} model(s) breached risk boundaries.")
    else:
        print("ALL TARGETS WITHIN TOLERANCE.")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_v2_protocol())
    except KeyboardInterrupt:
        print("\n\n[!] PROTOCOL ABORTED BY USER")
        sys.exit(0)