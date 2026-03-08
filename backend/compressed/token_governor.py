"""
Token governance — estimate, track, compress, and enforce budgets.
Uses tiktoken for accurate counting and SQLite for persistence.
"""

import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

import tiktoken

logger = logging.getLogger("compressed.tokens")

# Use cl100k_base (GPT-4 / Gemini approximate)
_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (accurate for GPT/Gemini family)."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Count tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # message overhead
        total += count_tokens(msg.get("content", ""))
        total += count_tokens(msg.get("role", ""))
    total += 2  # reply priming
    return total


@dataclass
class TokenBudget:
    """Token budget for a single pipeline run."""
    total_limit: int = 16000      # Total input+output budget for entire run
    per_call_input: int = 4000    # Max input tokens per API call
    per_call_output: int = 2048   # Max output tokens per API call
    search_context: int = 2000    # Max tokens for web search context
    history_context: int = 1500   # Max tokens for session history
    reserve: int = 500            # Reserved for system overhead

    tokens_used_in: int = 0
    tokens_used_out: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_limit - self.tokens_used_in - self.tokens_used_out)

    @property
    def exhausted(self) -> bool:
        return self.remaining < self.reserve

    def record(self, tokens_in: int, tokens_out: int):
        self.tokens_used_in += tokens_in
        self.tokens_used_out += tokens_out

    def allowed_output(self) -> int:
        """How many output tokens are allowed for the next call."""
        return min(self.per_call_output, self.remaining - self.reserve)


class TokenGovernor:
    """Controls token flow through the pipeline."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        self.budget = budget or TokenBudget()

    def estimate_prompt(self, prompt: str) -> int:
        return count_tokens(prompt)

    def compress_context(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        tokens = count_tokens(text)
        if tokens <= max_tokens:
            return text
        # Binary search for the right cutoff point
        enc = _get_encoding()
        encoded = enc.encode(text)
        truncated = enc.decode(encoded[:max_tokens])
        logger.info(f"Compressed context from {tokens} to {max_tokens} tokens")
        return truncated

    def compress_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Trim history to fit within budget."""
        total = count_messages_tokens(messages)
        if total <= self.budget.history_context:
            return messages

        # Keep most recent messages, drop oldest
        result = []
        running = 0
        for msg in reversed(messages):
            msg_tokens = count_tokens(msg.get("content", "")) + 6
            if running + msg_tokens > self.budget.history_context:
                break
            result.insert(0, msg)
            running += msg_tokens

        logger.info(f"Compressed history from {len(messages)} to {len(result)} messages")
        return result

    def compress_search_context(self, search_text: str) -> str:
        """Compress search results to fit budget."""
        return self.compress_context(search_text, self.budget.search_context)

    def check_budget(self) -> bool:
        """Returns True if budget allows another API call."""
        return not self.budget.exhausted

    def record_usage(self, tokens_in: int, tokens_out: int):
        self.budget.record(tokens_in, tokens_out)

    def get_summary(self) -> Dict:
        return {
            "total_limit": self.budget.total_limit,
            "tokens_in": self.budget.tokens_used_in,
            "tokens_out": self.budget.tokens_used_out,
            "remaining": self.budget.remaining,
            "exhausted": self.budget.exhausted,
        }
