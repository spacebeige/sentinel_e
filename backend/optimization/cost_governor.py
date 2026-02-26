"""
============================================================
Sentinel-E — Cost Governance Engine
============================================================
Token budget enforcement, cost tracking, and automatic model
downgrade when thresholds are exceeded.

Features:
  - Per-session token cap
  - Per-request token tracking
  - Hard ceiling guardrail
  - Automatic model tier downgrade
  - Cost estimation per model
  - Usage analytics (internal only)

Memory safety: stdlib only. Bounded data structures.
============================================================
"""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger("CostGovernor")


# ============================================================
# COST RATES (per 1K tokens)
# ============================================================

# Model cost table — aligned with COGNITIVE_MODEL_REGISTRY
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "groq-small": {"input": 0.00005, "output": 0.00008},
    "llama-3.3": {"input": 0.00059, "output": 0.00079},
    "qwen-vl-2.5": {"input": 0.00000, "output": 0.00000},  # OpenRouter free tier
    "qwen3-coder": {"input": 0.00000, "output": 0.00000},  # OpenRouter free tier
    "qwen3-vl": {"input": 0.00000, "output": 0.00000},      # OpenRouter free tier
    "nemotron-nano": {"input": 0.00000, "output": 0.00000},  # OpenRouter free tier
    "kimi-2.5": {"input": 0.00000, "output": 0.00000},       # OpenRouter free tier
    # Legacy aliases (backward compat for recorded usage)
    "llama-3.1-8b": {"input": 0.00005, "output": 0.00008},
    "llama-3.3-70b": {"input": 0.00059, "output": 0.00079},
    "qwen-2.5-7b": {"input": 0.00000, "output": 0.00000},
}

# Tier cost multipliers (relative)
TIER_COST_WEIGHTS = {
    "budget": 0.3,
    "standard": 0.5,
    "premium": 1.0,
}


def estimate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for a single request."""
    costs = MODEL_COSTS.get(model_id, {"input": 0.0005, "output": 0.0008})
    return (
        (input_tokens / 1000) * costs["input"]
        + (output_tokens / 1000) * costs["output"]
    )


# ============================================================
# SESSION BUDGET TRACKER
# ============================================================

@dataclass
class SessionBudget:
    """Tracks token/cost usage for a single session."""
    session_id: str
    max_tokens: int = 50000        # Hard cap per session
    max_cost: float = 0.10         # Hard cost cap per session ($)
    warning_threshold: float = 0.8 # Warn at 80% of budget
    created_at: float = field(default_factory=time.time)

    # Tracked state
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    downgrade_triggered: bool = False

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def token_utilization(self) -> float:
        return self.total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    @property
    def cost_utilization(self) -> float:
        return self.total_cost / self.max_cost if self.max_cost > 0 else 0.0

    @property
    def budget_exceeded(self) -> bool:
        return self.total_tokens >= self.max_tokens or self.total_cost >= self.max_cost

    @property
    def budget_warning(self) -> bool:
        return (
            self.token_utilization >= self.warning_threshold
            or self.cost_utilization >= self.warning_threshold
        )

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ):
        """Record token usage for this session."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.request_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "total_cost": round(self.total_cost, 6),
            "max_cost": self.max_cost,
            "token_utilization": round(self.token_utilization, 3),
            "cost_utilization": round(self.cost_utilization, 3),
            "request_count": self.request_count,
            "budget_exceeded": self.budget_exceeded,
            "budget_warning": self.budget_warning,
            "downgrade_triggered": self.downgrade_triggered,
        }


# ============================================================
# COST GOVERNOR
# ============================================================

@dataclass
class GovernanceDecision:
    """Result of cost governance check."""
    allowed: bool
    recommended_model: Optional[str] = None
    recommended_tier: Optional[str] = None
    downgraded: bool = False
    reason: str = ""
    budget_status: Optional[Dict[str, Any]] = None


class CostGovernor:
    """
    Cost governance engine.

    Enforces:
      - Per-session token caps
      - Per-session cost caps
      - Automatic model downgrade when budget is high
      - Hard ceiling guardrail (block request if exceeded)
      - Global cost tracking

    Usage:
        governor = CostGovernor()
        decision = governor.check_budget("session_123", "premium")
        if not decision.allowed:
            return error_response(decision.reason)
        if decision.downgraded:
            use decision.recommended_model instead
    """

    def __init__(
        self,
        session_max_tokens: int = 50000,
        session_max_cost: float = 0.10,
        max_sessions: int = 1000,
        warning_threshold: float = 0.8,
    ):
        self.session_max_tokens = session_max_tokens
        self.session_max_cost = session_max_cost
        self.warning_threshold = warning_threshold
        self.max_sessions = max_sessions

        # Session budgets (bounded)
        self._sessions: OrderedDict[str, SessionBudget] = OrderedDict()

        # Global tracking
        self._global_tokens: int = 0
        self._global_cost: float = 0.0
        self._global_requests: int = 0
        self._request_log: List[Dict[str, Any]] = []
        self._max_log_size = 500

    def _get_session(self, session_id: str) -> SessionBudget:
        """Get or create session budget tracker."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBudget(
                session_id=session_id,
                max_tokens=self.session_max_tokens,
                max_cost=self.session_max_cost,
                warning_threshold=self.warning_threshold,
            )
            # Evict oldest sessions
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return self._sessions[session_id]

    def check_budget(
        self,
        session_id: str,
        requested_tier: str = "standard",
        estimated_tokens: int = 0,
    ) -> GovernanceDecision:
        """
        Pre-request budget check.

        Returns GovernanceDecision with:
          - allowed: whether to proceed
          - recommended_model: which model to use (may be downgraded)
          - downgraded: whether model was auto-downgraded
        """
        budget = self._get_session(session_id)

        # Hard ceiling: block if budget exhausted
        if budget.budget_exceeded:
            return GovernanceDecision(
                allowed=False,
                reason=f"Session budget exceeded (tokens: {budget.total_tokens}/{budget.max_tokens}, cost: ${budget.total_cost:.4f}/${budget.max_cost})",
                budget_status=budget.to_dict(),
            )

        # Check if this request would exceed budget
        projected = budget.total_tokens + estimated_tokens
        if projected > budget.max_tokens:
            return GovernanceDecision(
                allowed=False,
                reason=f"Request would exceed token budget ({projected} > {budget.max_tokens})",
                budget_status=budget.to_dict(),
            )

        # Auto-downgrade if approaching budget
        recommended_tier = requested_tier
        recommended_model = None
        downgraded = False

        if budget.budget_warning and requested_tier == "premium":
            # Downgrade premium → standard when budget is warning
            recommended_tier = "budget"
            recommended_model = "groq-small"
            downgraded = True
            budget.downgrade_triggered = True
            logger.info(
                f"Session {session_id}: auto-downgrade premium → budget "
                f"(utilization: tokens={budget.token_utilization:.1%}, cost={budget.cost_utilization:.1%})"
            )
        elif budget.token_utilization >= 0.6 and requested_tier == "premium":
            # Downgrade premium → standard at 60%
            recommended_tier = "standard"
            recommended_model = "qwen-vl-2.5"
            downgraded = True
            budget.downgrade_triggered = True

        return GovernanceDecision(
            allowed=True,
            recommended_model=recommended_model,
            recommended_tier=recommended_tier,
            downgraded=downgraded,
            reason="ok" if not downgraded else f"auto-downgraded to {recommended_tier}",
            budget_status=budget.to_dict(),
        )

    def record_usage(
        self,
        session_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        cache_hit: bool = False,
    ):
        """
        Record usage after a request completes.
        """
        cost = estimate_cost(model_id, input_tokens, output_tokens)

        # Session tracking
        budget = self._get_session(session_id)
        budget.record_usage(input_tokens, output_tokens, cost)

        # Global tracking
        self._global_tokens += input_tokens + output_tokens
        self._global_cost += cost
        self._global_requests += 1

        # Request log
        entry = {
            "session_id": session_id,
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": round(cost, 8),
            "latency_ms": round(latency_ms, 1),
            "cache_hit": cache_hit,
            "timestamp": time.time(),
            "session_utilization": round(budget.token_utilization, 3),
        }
        self._request_log.append(entry)
        if len(self._request_log) > self._max_log_size:
            self._request_log = self._request_log[-self._max_log_size:]

        logger.debug(
            f"Usage: model={model_id}, tokens={input_tokens}+{output_tokens}, "
            f"cost=${cost:.6f}, session_util={budget.token_utilization:.1%}"
        )

    def get_session_budget(self, session_id: str) -> Dict[str, Any]:
        """Get budget status for a session."""
        if session_id in self._sessions:
            return self._sessions[session_id].to_dict()
        return {"session_id": session_id, "status": "no_data"}

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cost/usage statistics."""
        # Compute per-model breakdown
        model_breakdown = {}
        for entry in self._request_log:
            model = entry.get("model_id", "unknown")
            if model not in model_breakdown:
                model_breakdown[model] = {
                    "requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "cache_hits": 0,
                }
            b = model_breakdown[model]
            b["requests"] += 1
            b["total_tokens"] += entry.get("input_tokens", 0) + entry.get("output_tokens", 0)
            b["total_cost"] += entry.get("cost", 0.0)
            if entry.get("cache_hit"):
                b["cache_hits"] += 1

        # Round costs
        for model in model_breakdown:
            model_breakdown[model]["total_cost"] = round(
                model_breakdown[model]["total_cost"], 6
            )

        return {
            "global_tokens": self._global_tokens,
            "global_cost": round(self._global_cost, 6),
            "global_requests": self._global_requests,
            "active_sessions": len(self._sessions),
            "model_breakdown": model_breakdown,
            "recent_requests": self._request_log[-10:] if self._request_log else [],
        }

    def reset_session(self, session_id: str):
        """Reset budget for a session."""
        self._sessions.pop(session_id, None)


# ── Module-level singleton ──
_governor: Optional[CostGovernor] = None


def get_cost_governor() -> CostGovernor:
    """Get or create the singleton CostGovernor."""
    global _governor
    if _governor is None:
        _governor = CostGovernor()
    return _governor
