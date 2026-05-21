from __future__ import annotations

from typing import Any, Dict, Optional

from agents.browser_runtime.groq_router import GroqBrowserRouter, compact_json


REFLECTION_SYSTEM_PROMPT = """You reflect on a browser action result.
Return only JSON: {"success":true,"retry":false,"retry_strategy":"","done":false,"summary":""}
Use the compact state and error only. Be concise."""


class BrowserReflection:
    def __init__(self, groq: Optional[GroqBrowserRouter] = None):
        self.groq = groq or GroqBrowserRouter()

    async def reflect(
        self,
        *,
        task: str,
        action: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "task": task[:600],
            "action": action,
            "result": {
                "ok": result.get("ok"),
                "error": result.get("error", ""),
                "permission": result.get("permission"),
                "state": result.get("state"),
            },
        }
        response = await self.groq.complete_json(
            system=REFLECTION_SYSTEM_PROMPT,
            user=compact_json(payload),
            max_tokens=420,
            temperature=0.1,
        )
        if response.ok and response.json_data:
            return {
                "ok": True,
                "success": bool(response.json_data.get("success", False)),
                "retry": bool(response.json_data.get("retry", False)),
                "retry_strategy": str(response.json_data.get("retry_strategy", ""))[:240],
                "done": bool(response.json_data.get("done", False)),
                "summary": str(response.json_data.get("summary", ""))[:400],
                "model": response.model,
                "latency_ms": round(response.latency_ms, 1),
            }

        return {
            "ok": False,
            "success": bool(result.get("ok")),
            "retry": not bool(result.get("ok")),
            "retry_strategy": "Retry with a different selector or extract state for user review.",
            "done": bool(result.get("ok")),
            "summary": result.get("error", "")[:300] or "Reflection unavailable.",
            "error": response.error,
            "model": response.model,
            "latency_ms": round(response.latency_ms, 1),
        }
