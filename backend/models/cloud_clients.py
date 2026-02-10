import os
import json
import aiohttp
from dotenv import load_dotenv

# ============================================================
# ENV LOADING (EXPLICIT, ROBUST)
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)


class CloudModelClient:
    """
    Research-grade cloud LLM client.
    Supports:
      - Groq (LLaMA-3.1 via Groq)
      - Mistral (official API)
      - Qwen 2.5 via OpenRouter (FREE tier) - Text Only for now
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # one-time sanity check (safe)
        print(
            "ENV CHECK →",
            bool(self.groq_api_key),
            bool(self.mistral_api_key),
            bool(self.openrouter_api_key),
        )

    # ============================================================
    # GROQ
    # ============================================================

    async def call_groq(self, prompt: str, system_role: str = "You are a fast, concise analytical assistant.") -> str:
        """
        Calls Groq hosted LLaMA 3.1 (free/quick).
        """
        if not self.groq_api_key:
            return "Groq API Key missing"

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            # Stable, supported model identifier
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return f"Groq Error {resp.status}: {text}"
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Groq Exception: {str(e)}"

    # ============================================================
    # MISTRAL
    # ============================================================

    async def call_mistral(self, prompt: str, system_role: str = "You are a helpful assistant.") -> str:
        """
        Calls the official Mistral API.
        """
        if not self.mistral_api_key:
            return "Mistral API Key missing"

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return f"Mistral Error {resp.status}: {text}"
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Mistral Exception: {str(e)}"

    # ============================================================
    # QWEN 2.5 7B (FREE) via OPENROUTER
    # ============================================================

    async def call_qwenvl(self, prompt: str, system_role: str = "You are a careful, analytical assistant.") -> str:
        """
        Calls Qwen 2.5 7B Instruct via OpenRouter (text-only).
        Uses free tier as available.
        """
        if not self.openrouter_api_key:
            return "Qwen/OpenRouter API Key missing"

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",

            # Must use HTTPS and a valid public referer
            "HTTP-Referer": "https://github.com/spacebeige/sentinel-llm",
            "X-Title": "Sentinel-Sigma",
        }
        payload = {
            "model": "qwen/qwen-2.5-7b-instruct",
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1024,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return f"OpenRouter/Qwen Error {resp.status}: {text}"
                    data = json.loads(text)
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Qwen/OpenRouter Exception: {str(e)}"
