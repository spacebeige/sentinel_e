from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DEFAULT_MEMORY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "browser_runtime_memory.json")
)


@dataclass
class BrowserRuntimeMemory:
    successful_selectors: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    failed_selectors: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    retry_strategies: List[Dict[str, Any]] = field(default_factory=list)
    successful_workflows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "successful_selectors": self.successful_selectors,
            "failed_selectors": self.failed_selectors,
            "retry_strategies": self.retry_strategies[-100:],
            "successful_workflows": self.successful_workflows[-100:],
        }


class BrowserMemoryManager:
    """Compact JSON memory for selectors and retry strategies only."""

    def __init__(self, path: str = DEFAULT_MEMORY_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._memory = self._load()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._memory.to_dict()

    def selector_hints(self, domain: str) -> Dict[str, Any]:
        domain_key = self._domain_key(domain)
        with self._lock:
            return {
                "successful_selectors": self._memory.successful_selectors.get(domain_key, [])[-20:],
                "failed_selectors": self._memory.failed_selectors.get(domain_key, [])[-20:],
                "retry_strategies": self._memory.retry_strategies[-10:],
            }

    def record_selector(self, domain: str, selector: str, action_type: str, success: bool, note: str = "") -> None:
        if not selector:
            return
        domain_key = self._domain_key(domain)
        entry = {
            "selector": selector,
            "action_type": action_type,
            "note": note[:160],
            "timestamp": self._now(),
        }
        with self._lock:
            target = self._memory.successful_selectors if success else self._memory.failed_selectors
            target.setdefault(domain_key, []).append(entry)
            target[domain_key] = target[domain_key][-50:]
            self._save()

    def record_retry_strategy(self, task: str, failed_action: Dict[str, Any], strategy: str) -> None:
        with self._lock:
            self._memory.retry_strategies.append({
                "task": task[:200],
                "failed_action": failed_action,
                "strategy": strategy[:240],
                "timestamp": self._now(),
            })
            self._memory.retry_strategies = self._memory.retry_strategies[-100:]
            self._save()

    def record_workflow(self, task: str, steps: List[Dict[str, Any]], success: bool) -> None:
        if not success:
            return
        compact_steps = [
            {
                "type": step.get("type"),
                "selector": step.get("selector"),
                "url": step.get("url"),
            }
            for step in steps[-12:]
        ]
        with self._lock:
            self._memory.successful_workflows.append({
                "task": task[:200],
                "steps": compact_steps,
                "timestamp": self._now(),
            })
            self._memory.successful_workflows = self._memory.successful_workflows[-100:]
            self._save()

    def _load(self) -> BrowserRuntimeMemory:
        if not os.path.exists(self.path):
            return BrowserRuntimeMemory()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return BrowserRuntimeMemory(
                successful_selectors=data.get("successful_selectors", {}),
                failed_selectors=data.get("failed_selectors", {}),
                retry_strategies=data.get("retry_strategies", []),
                successful_workflows=data.get("successful_workflows", []),
            )
        except Exception:
            return BrowserRuntimeMemory()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._memory.to_dict(), handle, indent=2)
        os.replace(tmp_path, self.path)

    @staticmethod
    def _domain_key(domain: Optional[str]) -> str:
        return (domain or "unknown").lower().strip()[:120]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


_memory_manager: Optional[BrowserMemoryManager] = None


def get_browser_memory_manager() -> BrowserMemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = BrowserMemoryManager()
    return _memory_manager
