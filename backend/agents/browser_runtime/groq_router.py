from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GroqRouteResult:
    ok: bool
    content: str = ""
    json_data: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    error: str = ""
    model: str = ""


class GroqBrowserRouter:
    """Small Groq-only reasoning adapter for browser planning/reflection."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROQ_LLAMA_INSTANT_KEY")
        self.model = model or os.getenv("BROWSER_RUNTIME_GROQ_MODEL", "llama-3.1-8b-instant")
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 700,
        temperature: float = 0.1,
    ) -> GroqRouteResult:
        result = await self.complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not result.ok:
            return result
        result.json_data = self._extract_json(result.content)
        if result.json_data is None:
            result.ok = False
            result.error = "Groq response did not contain a JSON object."
        return result

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 700,
        temperature: float = 0.1,
    ) -> GroqRouteResult:
        if not self.api_key:
            return GroqRouteResult(ok=False, error="GROQ_API_KEY is not configured.", model=self.model)

        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    latency_ms = (time.monotonic() - started) * 1000
                    if response.status != 200:
                        text = await response.text()
                        return GroqRouteResult(
                            ok=False,
                            error=f"Groq error {response.status}: {text[:180]}",
                            latency_ms=latency_ms,
                            model=self.model,
                        )
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return GroqRouteResult(
                        ok=bool(content.strip()),
                        content=content.strip(),
                        latency_ms=latency_ms,
                        model=self.model,
                        error="" if content.strip() else "Empty Groq response.",
                    )
        except Exception as exc:
            return GroqRouteResult(
                ok=False,
                error=str(exc),
                latency_ms=(time.monotonic() - started) * 1000,
                model=self.model,
            )

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None


def compact_json(data: Dict[str, Any], max_chars: int = 6000) -> str:
    text = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '"...truncated"}'
