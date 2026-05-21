from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


RISKY_ACTION_TYPES = {"submit", "press_enter", "upload_file"}
SAFE_ACTION_TYPES = {"goto", "scroll", "wait", "extract", "click", "fill", "press"}

RISKY_KEYWORDS = (
    "send",
    "submit",
    "confirm",
    "purchase",
    "buy",
    "pay",
    "checkout",
    "delete",
    "remove",
    "archive",
    "unsubscribe",
    "transfer",
    "publish",
    "post",
    "share",
    "email",
    "message",
)


@dataclass
class PermissionDecision:
    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""
    risk_level: str = "low"
    action: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "requires_confirmation": self.requires_confirmation,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "action": self.action,
        }


class BrowserPermissionPolicy:
    """
    Workspace governance boundary for browser actions.

    Safe read/navigation actions can run automatically. Irreversible or
    externally-visible actions require caller confirmation before execution.
    """

    def __init__(
        self,
        allowed_domains: Optional[Iterable[str]] = None,
        blocked_domains: Optional[Iterable[str]] = None,
    ):
        self.allowed_domains = {d.lower().strip() for d in (allowed_domains or []) if d}
        self.blocked_domains = {d.lower().strip() for d in (blocked_domains or []) if d}

    def evaluate(self, action: Dict[str, Any], page_state: Optional[Dict[str, Any]] = None) -> PermissionDecision:
        action_type = str(action.get("type", "")).lower().strip()
        selector = str(action.get("selector", "") or "")
        value = str(action.get("value", "") or "")
        label = str(action.get("label", "") or "")
        url = str(action.get("url", "") or "")
        text = " ".join([action_type, selector, value, label, url]).lower()

        if action_type not in SAFE_ACTION_TYPES and action_type not in RISKY_ACTION_TYPES:
            return PermissionDecision(
                allowed=False,
                requires_confirmation=True,
                reason=f"Unsupported browser action: {action_type or 'missing'}",
                risk_level="high",
                action=action,
            )

        domain_decision = self._evaluate_domain(url)
        if domain_decision is not None:
            domain_decision.action = action
            return domain_decision

        if action.get("confirmed") is True:
            return PermissionDecision(
                allowed=True,
                requires_confirmation=False,
                reason="User confirmation supplied.",
                risk_level="confirmed",
                action=action,
            )

        if action_type in RISKY_ACTION_TYPES or any(keyword in text for keyword in RISKY_KEYWORDS):
            return PermissionDecision(
                allowed=False,
                requires_confirmation=True,
                reason="Action may submit, send, delete, purchase, or otherwise change external state.",
                risk_level="high",
                action=action,
            )

        if action_type == "click" and self._click_targets_risky_element(action, page_state):
            return PermissionDecision(
                allowed=False,
                requires_confirmation=True,
                reason="Click target appears to be a submit or irreversible control.",
                risk_level="high",
                action=action,
            )

        return PermissionDecision(
            allowed=True,
            requires_confirmation=False,
            reason="Read/navigation action allowed.",
            risk_level="low",
            action=action,
        )

    def _evaluate_domain(self, url: str) -> Optional[PermissionDecision]:
        if not url:
            return None

        lowered = url.lower()
        if any(blocked in lowered for blocked in self.blocked_domains):
            return PermissionDecision(
                allowed=False,
                requires_confirmation=False,
                reason="URL matches a blocked domain.",
                risk_level="blocked",
            )

        if self.allowed_domains and not any(allowed in lowered for allowed in self.allowed_domains):
            return PermissionDecision(
                allowed=False,
                requires_confirmation=True,
                reason="URL is outside the configured browser domain allowlist.",
                risk_level="medium",
            )

        return None

    @staticmethod
    def _click_targets_risky_element(action: Dict[str, Any], page_state: Optional[Dict[str, Any]]) -> bool:
        if not page_state:
            return False

        selector = str(action.get("selector", "") or "")
        if not selector:
            return False

        candidates: List[Dict[str, Any]] = []
        candidates.extend(page_state.get("buttons") or [])
        candidates.extend(page_state.get("inputs") or [])

        for item in candidates:
            if item.get("selector") != selector:
                continue
            text = " ".join(
                str(item.get(key, "") or "")
                for key in ("text", "label", "placeholder", "name", "type")
            ).lower()
            if any(keyword in text for keyword in RISKY_KEYWORDS):
                return True
        return False
