import asyncio
import argparse
import sys
import os
import json

# Add project root to path if running from backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add backend folder explicitly if needed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


try:
    from backend.sentinel.sentinel_sigma_v4 import SentinelSigmaOrchestratorV4, SigmaV4Config
except ImportError:
    # Try importing directly if we are inside the package structure weirdly
    from sentinel.sentinel_sigma_v4 import SentinelSigmaOrchestratorV4, SigmaV4Config

async def main():
    parser = argparse.ArgumentParser(description="Run Sentinel-Sigma V4")
    parser.add_argument("topic", nargs="?", type=str, help="Input topic")
    parser.add_argument("--mode", type=str, default="conversational", help="conversational | forensic | experimental")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds")
    parser.add_argument("--kill-switch", action="store_true", help="Enable kill switch")
    parser.add_argument("--execute", action="store_true", help="Force execute=True even without topic")
    
    args = parser.parse_args()
    
    # Logic: if topic is provided, execute=True. If --execute is provided, execute=True.
    execute = bool(args.topic) or args.execute
    
    config = SigmaV4Config(
        execute=execute,
        mode=args.mode,
        topic=args.topic if args.topic else "",
        rounds=args.rounds,
        kill_switch_enabled=args.kill_switch
    )
    
    orchestrator = SentinelSigmaOrchestratorV4()
    result = await orchestrator.run_sentinel(config)
    
    # Print Human Layer
    if "human_layer" in result:
        print("\n" + "="*40)
        print(" SENTINEL-Σ v4 HUMAN LAYER")
        print("="*40 + "\n")
        print(result["human_layer"])
    
    # Print Machine Layer
    if "machine_layer" in result:
        print("\n" + "="*40)
        print(" SENTINEL-Σ v4 MACHINE LAYER")
        print("="*40 + "\n")
        print(json.dumps(result["machine_layer"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
