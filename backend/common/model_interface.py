import os
import json
import asyncio
import requests
from dotenv import load_dotenv
# Local LLM engine import
from backend.models.local_engine import LocalLLMEngine

# ============================================================
# ENV LOADING
# ============================================================

# Robustly find the .env file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, ".env") # backend/.env
load_dotenv(env_path)

class ModelInterface:
    """
    Unified interface for raw model calls.
    Uses requests (wrapped in threads) for reliability over aiohttp.
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        # Local model - always initialize (will work in simulated mode if no model files present)
        self.local_llm = None
        try:
            self.local_llm = LocalLLMEngine()
            self.local_llm.load_model()
            print("[Local LLM] LocalLLMEngine initialized (simulated mode if no models found)")
        except Exception as e:
            print(f"[Local LLM] Failed to initialize: {e}")
    async def call_local(self, prompt: str, system_role: str = "You are a helpful local assistant.") -> str:
        """
        Calls the local LLM if available, otherwise returns a message.
        """
        if not self.local_llm:
            return "Local LLM not configured or failed to load."
        # For now, just concatenate system_role and prompt for context
        try:
            # Synchronous call, wrap in thread for async
            return await asyncio.to_thread(self.local_llm.determine_intent, f"{system_role}\n{prompt}")
        except Exception as e:
            return f"Local LLM Exception: {str(e)}"

    async def call_groq(self, prompt: str, system_role: str = "You are a fast, concise analytical assistant.", temperature: float = 0.3) -> str:
        """
        Calls Groq hosted LLaMA 3.1.
        """
        return await asyncio.to_thread(self._sync_request, 
            "https://api.groq.com/openai/v1/chat/completions",
            self.groq_api_key,
            "llama-3.1-8b-instant",
            prompt,
            system_role,
            temperature
        )

    async def call_mistral(self, prompt: str, system_role: str = "You are a helpful assistant.", temperature: float = 0.2) -> str:
        """
        Calls the official Mistral API.
        """
        return await asyncio.to_thread(self._sync_request,
            "https://api.mistral.ai/v1/chat/completions",
            self.mistral_api_key,
            "mistral-small-latest",
            prompt,
            system_role,
            temperature
        )

    async def call_openrouter(self, prompt: str, system_role: str = "You are a helpful assistant.") -> str:
        """
        Calls OpenRouter (Qwen 2.5 7B).
        """
        return await asyncio.to_thread(self._sync_request,
            "https://openrouter.ai/api/v1/chat/completions",
            self.openrouter_api_key,
            "qwen/qwen-2.5-7b-instruct",
            prompt,
            system_role,
            0.2, # Temp
            {"HTTP-Referer": "http://localhost:3000", "X-Title": "Sentinel-E"}
        )

    def _sync_request(self, url, api_key, model, prompt, system_role, temperature, extra_headers=None):
        if not api_key:
            return f"API Key missing for {model}"
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                print(f"Error calling {model}: {resp.status_code} - {resp.text}")
                return f"Error {resp.status_code}: {resp.text}"
            
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Exception calling {model}: {e}")
            return f"Exception: {str(e)}"
        # Placeholder for local execution
        # In a real scenario this might call llama.cpp python bindings
        return "Local model execution not fully implemented in this refactor."
