from __future__ import annotations

from typing import Any, Dict, Optional

from agents.browser_runtime.groq_router import GroqBrowserRouter, compact_json


PLANNER_SYSTEM_PROMPT = """You plan browser actions for an embedded runtime.
Return only JSON: {"action":{"type":"","url":"","selector":"","value":"","key":"","direction":""},"rationale":"","done":false}
Allowed action types: goto, click, fill, press, scroll, wait, extract.
Never plan purchases, sends, deletes, or form submissions without requiring confirmation.
Use compact selectors from page state. Prefer reading/extracting when uncertain."""


class BrowserPlanner:
    def __init__(self, groq: Optional[GroqBrowserRouter] = None):
        self.groq = groq or GroqBrowserRouter()

    async def plan(
        self,
        *,
        task: str,
        state: Dict[str, Any],
        memory_hints: Optional[Dict[str, Any]] = None,
        previous_error: str = "",
    ) -> Dict[str, Any]:
        user_payload = {
            "task": task[:600],
            "state": state,
            "memory_hints": memory_hints or {},
            "previous_error": previous_error[:300],
        }
        result = await self.groq.complete_json(
            system=PLANNER_SYSTEM_PROMPT,
            user=compact_json(user_payload),
            max_tokens=550,
            temperature=0.1,
        )
        if result.ok and result.json_data:
            action = result.json_data.get("action") or {}
            return {
                "ok": True,
                "action": action,
                "rationale": str(result.json_data.get("rationale", ""))[:300],
                "done": bool(result.json_data.get("done", False)),
                "model": result.model,
                "latency_ms": round(result.latency_ms, 1),
            }

        return {
            "ok": False,
            "action": {"type": "extract"},
            "rationale": "Planning failed; extract current browser state for caller review.",
            "done": False,
            "error": result.error,
            "model": result.model,
            "latency_ms": round(result.latency_ms, 1),
        }
