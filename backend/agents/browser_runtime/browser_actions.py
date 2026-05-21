from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agents.browser_runtime.browser_observer import BrowserObserver
from agents.browser_runtime.memory_manager import BrowserMemoryManager, get_browser_memory_manager
from agents.browser_runtime.permissions import BrowserPermissionPolicy, PermissionDecision

logger = logging.getLogger("BrowserRuntime.Actions")


@dataclass
class BrowserActionResult:
    ok: bool
    action: Dict[str, Any]
    state: Optional[Dict[str, Any]] = None
    permission: Optional[Dict[str, Any]] = None
    error: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "state": self.state,
            "permission": self.permission,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "metadata": self.metadata,
        }


class BrowserActionExecutor:
    """Executes a small set of governance-aware Playwright actions."""

    def __init__(
        self,
        observer: Optional[BrowserObserver] = None,
        permissions: Optional[BrowserPermissionPolicy] = None,
        memory: Optional[BrowserMemoryManager] = None,
    ):
        self.observer = observer or BrowserObserver()
        self.permissions = permissions or BrowserPermissionPolicy()
        self.memory = memory or get_browser_memory_manager()

    async def execute(self, page, action: Dict[str, Any], *, page_state: Optional[Dict[str, Any]] = None) -> BrowserActionResult:
        started = time.monotonic()
        action_type = str(action.get("type", "")).lower().strip()
        decision = self.permissions.evaluate(action, page_state)
        if not decision.allowed:
            return BrowserActionResult(
                ok=False,
                action=action,
                state=page_state,
                permission=decision.to_dict(),
                error=decision.reason,
                latency_ms=(time.monotonic() - started) * 1000,
            )

        try:
            await self._execute_raw(page, action_type, action)
            state = (await self.observer.observe(page)).to_dict()
            selector = str(action.get("selector", "") or "")
            if selector:
                self.memory.record_selector(page.url, selector, action_type, True)
            return BrowserActionResult(
                ok=True,
                action=action,
                state=state,
                permission=decision.to_dict(),
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except Exception as exc:
            selector = str(action.get("selector", "") or "")
            if selector:
                self.memory.record_selector(page.url, selector, action_type, False, note=str(exc))
            logger.debug("Browser action failed: %s", exc)
            return BrowserActionResult(
                ok=False,
                action=action,
                state=page_state,
                permission=decision.to_dict(),
                error=str(exc),
                latency_ms=(time.monotonic() - started) * 1000,
            )

    async def _execute_raw(self, page, action_type: str, action: Dict[str, Any]) -> None:
        timeout = int(action.get("timeout_ms") or 15000)
        if action_type == "goto":
            url = action.get("url")
            if not url:
                raise ValueError("goto requires url")
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            return

        if action_type == "click":
            selector = self._require_selector(action)
            await page.locator(selector).first.click(timeout=timeout)
            return

        if action_type == "fill":
            selector = self._require_selector(action)
            await page.locator(selector).first.fill(str(action.get("value", "")), timeout=timeout)
            return

        if action_type == "press":
            selector = action.get("selector")
            key = str(action.get("key") or "Enter")
            if selector:
                await page.locator(selector).first.press(key, timeout=timeout)
            else:
                await page.keyboard.press(key)
            return

        if action_type == "press_enter":
            selector = action.get("selector")
            if selector:
                await page.locator(selector).first.press("Enter", timeout=timeout)
            else:
                await page.keyboard.press("Enter")
            return

        if action_type == "submit":
            selector = self._require_selector(action)
            await page.locator(selector).first.press("Enter", timeout=timeout)
            return

        if action_type == "scroll":
            direction = str(action.get("direction") or "down").lower()
            distance = int(action.get("distance") or 700)
            signed_distance = -abs(distance) if direction == "up" else abs(distance)
            await page.mouse.wheel(0, signed_distance)
            return

        if action_type == "wait":
            await page.wait_for_timeout(int(action.get("duration_ms") or 1000))
            return

        if action_type == "extract":
            return

        raise ValueError(f"Unsupported action type: {action_type}")

    @staticmethod
    def _require_selector(action: Dict[str, Any]) -> str:
        selector = str(action.get("selector", "") or "").strip()
        if not selector:
            raise ValueError(f"{action.get('type')} requires selector")
        return selector
