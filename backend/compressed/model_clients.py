"""
Multi-provider model clients for role-based reasoning pipeline.

Providers:
  - Gemini Flash 2.0 (Synthesis primary)
  - Groq (Llama-3.3-70B, Llama-3.1-8B, Mixtral-8x7B, Llama 4 Scout)
  - Qwen 2.5 VL Instruct (DashScope — free tier)

Role assignments:
  Analysis:     Groq Llama-3.3-70B → Gemini Flash
  Critique A:   Groq Mixtral-8x7B → Groq 8B
  Critique B:   Groq Llama 4 Scout → Groq 8B
  Critique C:   Qwen 2.5 VL → Groq 8B
  Synthesis:    Gemini Flash 2.0 → Groq Llama-3.3-70B
  Verification: Groq Llama-3.1-8B → Gemini Flash
  Summarize:    Groq Llama-3.1-8B → Gemini Flash
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
    """Async client for Gemini Flash 2.0 via REST API."""

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

                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ModelResponse(content="", model=self.MODEL_ID, error="No candidates returned")

                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)

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
    """Async Groq client (OpenAI-compatible API)."""

    def __init__(self, api_key: Optional[str] = None, model_id: str = "llama-3.1-8b-instant", api_key_env: str = "GROQ_API_KEY"):
        self._api_key = api_key or os.getenv(api_key_env, "") or os.getenv("GROQ_API_KEY", "")
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


class QwenClient:
    """Async client for Qwen models via DashScope (OpenAI-compatible API)."""

    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(
        self,
        model_id: str = "qwen2.5-vl-7b-instruct",
        api_key: Optional[str] = None,
    ):
        self._model_id = model_id
        self._api_key = api_key or os.getenv("QWEN_API_KEY", "")

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
            return ModelResponse(content="", model=self._model_id, error="QWEN_API_KEY not set")

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
                async with session.post(
                    self.API_URL, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
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


class RoleBasedRouter:
    """Routes requests to specific models based on reasoning role.

    Role assignments (v2 — no OpenRouter):
      analysis:     Groq Llama-3.3-70B → Gemini Flash
      critique_a:   Groq Mixtral-8x7B → Groq 8B
      critique_b:   Groq Llama 4 Scout → Groq 8B
      critique_c:   Qwen 2.5 VL → Groq 8B
      synthesis:    Gemini Flash 2.0 → Groq Llama-3.3-70B
      verification: Groq Llama-3.1-8B → Gemini Flash
      summarize:    Groq Llama-3.1-8B → Gemini Flash
    """

    def __init__(self):
        self.gemini = GeminiFlashClient()
        self.groq_70b = GroqClient(model_id="llama-3.3-70b-versatile", api_key_env="GROQ_LLAMA70B_KEY")
        self.groq_8b = GroqClient(model_id="llama-3.1-8b-instant", api_key_env="GROQ_LLAMA8B_KEY")
        self.groq_mixtral = GroqClient(model_id="mixtral-8x7b-32768", api_key_env="GROQ_MIXTRAL_KEY")
        self.groq_gemma = GroqClient(model_id="meta-llama/llama-4-scout-17b-16e-instruct", api_key_env="GROQ_GEMMA_KEY")
        self.qwen = QwenClient()

        self._call_count = 0

        # Client lookup by public model ID
        self._clients_by_id = {
            "llama-3.3-70b": self.groq_70b,
            "llama-3.1-8b": self.groq_8b,
            "mixtral-8x7b": self.groq_mixtral,
            "llama4-scout": self.groq_gemma,
            "gemini-flash": self.gemini,
            "qwen-2.5-vl": self.qwen,
        }

    @property
    def any_available(self) -> bool:
        return self.gemini.available or self.groq_70b.available

    def list_models(self) -> List[Dict[str, Any]]:
        """Return list of available models with metadata."""
        return [
            {"id": mid, "available": client.available}
            for mid, client in self._clients_by_id.items()
        ]

    def get_client_by_id(self, model_id: str):
        """Get a client by its public model ID."""
        return self._clients_by_id.get(model_id)

    async def generate(
        self,
        role: str,
        prompt: str,
        system_instruction: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> ModelResponse:
        """Generate with role-based routing and automatic fallback."""
        self._call_count += 1

        # ── Analysis: Groq 70B → Gemini Flash ──
        if role == "analysis":
            if self.groq_70b.available:
                result = await self.groq_70b.generate(prompt, system_instruction, max_tokens, temperature)
                if result.ok:
                    return result
                logger.warning(f"Groq-70B failed for analysis: {result.error}")
            if self.gemini.available:
                return await self.gemini.generate(prompt, system_instruction, max_tokens, temperature)

        # ── Synthesis: Gemini Flash → Groq 70B ──
        elif role == "synthesis":
            if self.gemini.available:
                result = await self.gemini.generate(prompt, system_instruction, max_tokens, temperature)
                if result.ok:
                    return result
                logger.warning(f"Gemini failed for synthesis: {result.error}")
            if self.groq_70b.available:
                return await self.groq_70b.generate(prompt, system_instruction, min(max_tokens, 1500), temperature)

        # ── Critique A: Mixtral-8x7B → Groq 8B ──
        elif role == "critique_a":
            if self.groq_mixtral.available:
                result = await self.groq_mixtral.generate(prompt, system_instruction, max_tokens, temperature)
                if result.ok:
                    return result
                logger.warning(f"Mixtral failed: {result.error}")
            if self.groq_8b.available:
                return await self.groq_8b.generate(prompt, system_instruction, max_tokens, temperature)

        # ── Critique B: Llama 4 Scout → Groq 8B ──
        elif role == "critique_b":
            if self.groq_gemma.available:
                result = await self.groq_gemma.generate(prompt, system_instruction, max_tokens, temperature)
                if result.ok:
                    return result
                logger.warning(f"Llama 4 Scout failed: {result.error}")
            if self.groq_8b.available:
                return await self.groq_8b.generate(prompt, system_instruction, max_tokens, temperature)

        # ── Critique C: Qwen 2.5 VL → Groq 8B ──
        elif role == "critique_c":
            if self.qwen.available:
                result = await self.qwen.generate(prompt, system_instruction, max_tokens, temperature)
                if result.ok:
                    return result
                logger.warning(f"Qwen failed: {result.error}")
            if self.groq_8b.available:
                return await self.groq_8b.generate(prompt, system_instruction, max_tokens, temperature)

        # ── Verification: Groq 8B → Gemini Flash ──
        elif role == "verification":
            if self.groq_8b.available:
                result = await self.groq_8b.generate(prompt, system_instruction, max_tokens, temperature)
                if result.ok:
                    return result
                logger.warning(f"Groq-8B failed for verification: {result.error}")
            if self.gemini.available:
                return await self.gemini.generate(prompt, system_instruction, max_tokens, temperature)

        # ── Summarize: Groq 8B → Gemini Flash ──
        elif role == "summarize":
            if self.groq_8b.available:
                result = await self.groq_8b.generate(prompt, system_instruction, max_tokens, temperature)
                if result.ok:
                    return result
            if self.gemini.available:
                return await self.gemini.generate(prompt, system_instruction, max_tokens, temperature)

        return ModelResponse(content="", model="none", error=f"No model available for role '{role}'")

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_call_count(self):
        self._call_count = 0


# ── Model Registry (for individual model mode) ──

MODELS_REGISTRY = {
    "llama-3.3-70b": {"provider": "groq", "model_id": "llama-3.3-70b-versatile", "key_env": "GROQ_LLAMA70B_KEY", "role": "analysis"},
    "llama-3.1-8b": {"provider": "groq", "model_id": "llama-3.1-8b-instant", "key_env": "GROQ_LLAMA8B_KEY", "role": "verification/summarize"},
    "mixtral-8x7b": {"provider": "groq", "model_id": "mixtral-8x7b-32768", "key_env": "GROQ_MIXTRAL_KEY", "role": "critique_a"},
    "llama4-scout": {"provider": "groq", "model_id": "meta-llama/llama-4-scout-17b-16e-instruct", "key_env": "GROQ_GEMMA_KEY", "role": "critique_b"},
    "gemini-flash": {"provider": "gemini", "model_id": "gemini-2.0-flash", "key_env": "GEMINI_API_KEY", "role": "synthesis"},
    "qwen-2.5-vl": {"provider": "dashscope", "model_id": "qwen2.5-vl-7b-instruct", "key_env": "QWEN_API_KEY", "role": "critique_c"},
}
