import os
import asyncio
from dotenv import load_dotenv
from backend.common.model_interface import ModelInterface

load_dotenv("backend/.env")

async def test_keys():
    print("--- ENV CHECK ---")
    print(f"GROQ: {os.getenv('GROQ_API_KEY')[:5]}...")
    print(f"OPENROUTER: {os.getenv('OPENROUTER_API_KEY')[:5]}...")
    
    mi = ModelInterface()
    print("\n--- MODEL CHECK ---")
    print("Testing Groq (LLaMA 3.1 8B)...")
    res_groq = await mi.call_groq("Hello")
    print(f"Groq Result: {res_groq[:50]}")
    
    print("\nTesting Llama 3.3 70B (via Groq)...")
    res_llama = await mi.call_llama70b("Hello")
    print(f"Llama 3.3 70B Result: {res_llama[:50]}")
    
    print("\nTesting OpenRouter (Qwen)...")
    res_or = await mi.call_openrouter("Hello")
    print(f"OpenRouter Result: {res_or[:50]}")

if __name__ == "__main__":
    asyncio.run(test_keys())
