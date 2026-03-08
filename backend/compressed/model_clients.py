"""
Gemini Flash 2.0 integration + local model fallback.
Provides a unified async interface for the compressed debate chain.
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("compressed.models")


@dataclass
class ModelResponse:
    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.content) and not self.error


class GeminiFlashClient:
    """Async client for Gemini Flash 2.0 (free tier) via google-genai."""

    MODEL_ID = "gemini-2.0-flash"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> ModelResponse:
        if not self._api_key:
            return ModelResponse(content="", model=self.MODEL_ID, error="GEMINI_API_KEY not set")

        # Use REST API directly for async compatibility
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_ID}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self._api_key}

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        if system_instruction:
            body = {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
        else:
            body = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        err_msg = data.get("error", {}).get("message", f"HTTP {resp.status}")
                        return ModelResponse(content="", model=self.MODEL_ID, error=err_msg)

                    # Extract text from response
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ModelResponse(content="", model=self.MODEL_ID, error="No candidates returned")

                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)

                    # Token usage
                    usage = data.get("usageMetadata", {})
                    tokens_in = usage.get("promptTokenCount", 0)
                    tokens_out = usage.get("candidatesTokenCount", 0)

                    return ModelResponse(
                        content=text.strip(),
                        model=self.MODEL_ID,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
        except asyncio.TimeoutError:
            return ModelResponse(content="", model=self.MODEL_ID, error="Timeout")
        except Exception as e:
            return ModelResponse(content="", model=self.MODEL_ID, error=str(e))


class GroqClient:
    """Lightweight async Groq client for local-speed inference."""

    def __init__(self, api_key: Optional[str] = None, model_id: str = "llama-3.1-8b-instant"):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model_id = model_id
        self._url = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> ModelResponse:
        if not self._api_key:
            return ModelResponse(content="", model=self._model_id, error="GROQ_API_KEY not set")

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        err_msg = data.get("error", {}).get("message", f"HTTP {resp.status}")
                        return ModelResponse(content="", model=self._model_id, error=err_msg)

                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return ModelResponse(
                        content=content.strip(),
                        model=self._model_id,
                        tokens_in=usage.get("prompt_tokens", 0),
                        tokens_out=usage.get("completion_tokens", 0),
                    )
        except asyncio.TimeoutError:
            return ModelResponse(content="", model=self._model_id, error="Timeout")
        except Exception as e:
            return ModelResponse(content="", model=self._model_id, error=str(e))


class ModelRouter:
    """Routes requests to the best available model.

    Priority:
      1. Gemini Flash 2.0 (primary — free tier, high quality)
      2. Groq llama-3.1-8b-instant (fast fallback)
    """

    def __init__(self):
        self.gemini = GeminiFlashClient()
        self.groq = GroqClient()
        self._call_count = 0

    @property
    def primary_available(self) -> bool:
        return self.gemini.available

    @property
    def any_available(self) -> bool:
        return self.gemini.available or self.groq.available

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        prefer_local: bool = False,
    ) -> ModelResponse:
        """Generate with automatic fallback."""
        self._call_count += 1

        # If prefer_local and Groq available, use Groq for intermediate steps
        if prefer_local and self.groq.available:
            result = await self.groq.generate(prompt, system_instruction, min(max_tokens, 1024), temperature)
            if result.ok:
                return result

        # Primary: Gemini Flash
        if self.gemini.available:
            result = await self.gemini.generate(prompt, system_instruction, max_tokens, temperature)
            if result.ok:
                return result
            logger.warning(f"Gemini failed: {result.error}, falling back to Groq")

        # Fallback: Groq
        if self.groq.available:
            result = await self.groq.generate(prompt, system_instruction, min(max_tokens, 1500), temperature)
            if result.ok:
                return result

        return ModelResponse(content="", model="none", error="No models available")

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_call_count(self):
        self._call_count = 0
