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

# Model cost table — aligned with COGNITIVE_MODEL_REGISTRY (v4 free-tier ensemble)
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # Tier 1 Anchor
    "llama33-70b": {"input": 0.00005, "output": 0.00008},
    # Tier 2 Debate
    "mixtral-8x7b": {"input": 0.0, "output": 0.0},
    "llama4-scout": {"input": 0.0, "output": 0.0},
    "qwen-2.5-vl": {"input": 0.0, "output": 0.0},
    "kimi-k2-thinking": {"input": 0.0001, "output": 0.0002},
    # Tier 3 Synthesis + Verification
    "gemini-flash": {"input": 0.0, "output": 0.0},
    "llama31-8b": {"input": 0.00005, "output": 0.00008},
    # NVIDIA Models
    "mistral-large-675b": {"input": 0.00008, "output": 0.00024},
    "kimi-k2-thinking": {"input": 0.0001, "output": 0.0002},
    # Legacy aliases (backward compat for recorded usage)
    "llama-3.1-8b": {"input": 0.00005, "output": 0.00008},
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
# API KEY TOKENIZATION TRACKER
# ============================================================

@dataclass
class APIKeyTokenization:
    """Tracks token usage per shared API key (e.g., NVIDIA_API_KEY)."""
    api_key_name: str  # e.g., "NVIDIA_API_KEY"
    max_tokens_per_day: int = 100000  # Daily quota per shared key
    max_tokens_per_hour: int = 10000  # Hourly quota per shared key
    created_at: float = field(default_factory=time.time)
    
    # Tracked state
    total_tokens_today: int = 0
    total_tokens_this_hour: int = 0
    tokens_by_model: Dict[str, int] = field(default_factory=dict)
    last_hour_timestamp: float = field(default_factory=time.time)
    request_count: int = 0
    
    @property
    def tokens_remaining_today(self) -> int:
        return max(0, self.max_tokens_per_day - self.total_tokens_today)
    
    @property
    def tokens_remaining_hour(self) -> int:
        return max(0, self.max_tokens_per_hour - self.total_tokens_this_hour)
    
    @property
    def daily_utilization(self) -> float:
        return self.total_tokens_today / self.max_tokens_per_day if self.max_tokens_per_day > 0 else 0.0
    
    @property
    def hourly_utilization(self) -> float:
        return self.total_tokens_this_hour / self.max_tokens_per_hour if self.max_tokens_per_hour > 0 else 0.0
    
    @property
    def is_daily_quota_exceeded(self) -> bool:
        return self.total_tokens_today >= self.max_tokens_per_day
    
    @property
    def is_hourly_quota_exceeded(self) -> bool:
        return self.total_tokens_this_hour >= self.max_tokens_per_hour
    
    def reset_hourly_window(self):
        """Reset hourly token counter (call when new hour starts)."""
        now = time.time()
        if now - self.last_hour_timestamp >= 3600:  # 1 hour has passed
            self.total_tokens_this_hour = 0
            self.last_hour_timestamp = now
    
    def record_tokens(self, model_id: str, input_tokens: int, output_tokens: int):
        """Record token usage for a model using this API key."""
        total_tokens = input_tokens + output_tokens
        self.total_tokens_today += total_tokens
        self.total_tokens_this_hour += total_tokens
        self.tokens_by_model[model_id] = self.tokens_by_model.get(model_id, 0) + total_tokens
        self.request_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key_name": self.api_key_name,
            "total_tokens_today": self.total_tokens_today,
            "max_tokens_per_day": self.max_tokens_per_day,
            "daily_utilization": round(self.daily_utilization, 3),
            "tokens_remaining_today": self.tokens_remaining_today,
            "total_tokens_this_hour": self.total_tokens_this_hour,
            "max_tokens_per_hour": self.max_tokens_per_hour,
            "hourly_utilization": round(self.hourly_utilization, 3),
            "tokens_remaining_hour": self.tokens_remaining_hour,
            "tokens_by_model": self.tokens_by_model,
            "request_count": self.request_count,
            "daily_quota_exceeded": self.is_daily_quota_exceeded,
            "hourly_quota_exceeded": self.is_hourly_quota_exceeded,
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

        # API Key tokenization tracking (per shared API key)
        self._api_key_tokens: Dict[str, APIKeyTokenization] = {}

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
            recommended_model = "qwen2.5-32b"
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

    # ── API Key Tokenization Methods ─────────────────────────
    
    def record_api_key_usage(
        self,
        api_key_name: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record token usage for a shared API key (e.g., NVIDIA_API_KEY)."""
        if api_key_name not in self._api_key_tokens:
            self._api_key_tokens[api_key_name] = APIKeyTokenization(api_key_name=api_key_name)
        
        tracker = self._api_key_tokens[api_key_name]
        tracker.reset_hourly_window()  # Auto-reset if hour has passed
        tracker.record_tokens(model_id, input_tokens, output_tokens)
    
    def get_api_key_usage(self, api_key_name: str) -> Dict[str, Any]:
        """Get current token usage stats for a shared API key."""
        if api_key_name not in self._api_key_tokens:
            return {"error": f"No usage tracked for {api_key_name}"}
        
        tracker = self._api_key_tokens[api_key_name]
        tracker.reset_hourly_window()
        return tracker.to_dict()
    
    def check_api_key_quota(self, api_key_name: str) -> Dict[str, Any]:
        """Check if API key quota is exceeded."""
        if api_key_name not in self._api_key_tokens:
            return {"allowed": True, "quota_name": api_key_name, "reason": "No usage yet"}
        
        tracker = self._api_key_tokens[api_key_name]
        tracker.reset_hourly_window()
        
        if tracker.is_daily_quota_exceeded:
            return {
                "allowed": False,
                "quota_name": api_key_name,
                "reason": f"Daily quota exceeded ({tracker.total_tokens_today}/{tracker.max_tokens_per_day})",
                "daily_utilization": round(tracker.daily_utilization, 3),
            }
        
        if tracker.is_hourly_quota_exceeded:
            return {
                "allowed": False,
                "quota_name": api_key_name,
                "reason": f"Hourly quota exceeded ({tracker.total_tokens_this_hour}/{tracker.max_tokens_per_hour})",
                "hourly_utilization": round(tracker.hourly_utilization, 3),
            }
        
        return {
            "allowed": True,
            "quota_name": api_key_name,
            "daily_utilization": round(tracker.daily_utilization, 3),
            "hourly_utilization": round(tracker.hourly_utilization, 3),
        }
    
    def get_all_api_key_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all tracked API keys."""
        for key_name in self._api_key_tokens:
            self._api_key_tokens[key_name].reset_hourly_window()
        
        return {
            key_name: tracker.to_dict()
            for key_name, tracker in self._api_key_tokens.items()
        }


# ── Module-level singleton ──
_governor: Optional[CostGovernor] = None


def get_cost_governor() -> CostGovernor:
    """Get or create the singleton CostGovernor."""
    global _governor
    if _governor is None:
        _governor = CostGovernor()
    return _governor
