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
    parser.add_argument("topic", nargs="?", type=str, help="Input topic/text")
    parser.add_argument("--mode", type=str, default="conversational", help="conversational | forensic | experimental")
    parser.add_argument("--rounds", type=int, default=1, help="Number of debate rounds (experimental mode)")
    parser.add_argument("--shadow", action="store_true", help="Enable shadow evaluation")
    parser.add_argument("--chat-id", type=str, default=None, help="Optional chat UUID")
    
    args = parser.parse_args()
    
    text = args.topic if args.topic else input("Enter topic/query: ")
    
    config_kwargs = {
        "text": text,
        "mode": args.mode,
        "rounds": args.rounds,
        "enable_shadow": args.shadow,
    }
    if args.chat_id:
        config_kwargs["chat_id"] = args.chat_id
    
    config = SigmaV4Config(**config_kwargs)
    
    orchestrator = SentinelSigmaOrchestratorV4()
    result = await orchestrator.run_sentinel(config)
    
    # Print Human-Readable Layer (priority_answer from data)
    priority_answer = result.data.get("priority_answer") if result.data else None
    if priority_answer:
        print("\n" + "="*40)
        print(" SENTINEL-Σ v4 RESPONSE")
        print("="*40 + "\n")
        print(priority_answer)
    
    # Print Machine Metadata
    if result.metadata:
        print("\n" + "="*40)
        print(" SENTINEL-Σ v4 METADATA")
        print("="*40 + "\n")
        print(json.dumps(result.metadata.model_dump(), indent=2))
    
    # Print Full Data Payload
    if result.data:
        print("\n" + "="*40)
        print(" SENTINEL-Σ v4 DATA PAYLOAD")
        print("="*40 + "\n")
        print(json.dumps(result.data, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
