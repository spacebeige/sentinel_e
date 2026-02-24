"""
Context Compiler — Sentinel-E Autonomous Reasoning Engine

Compiles structured context blocks for model injection.
Replaces naive full-history injection with priority-weighted,
token-budgeted, structured context.

Blocks:
  [Verified Evidence]   — 35% token budget
  [Recent Updates]      — 10% token budget
  [Known Conflicts]     —  5% token budget
  [Confidence Metrics]  —  5% token budget
  [Conversation]        — 25% token budget
  [System Prompt]       — 10% token budget
  [User Query]          — 10% token budget

Integrates with:
  - backend/core/knowledge_memory.py (SessionMemoryTier, EvidenceMemory, KnowledgeGraph)
  - backend/memory/memory_engine.py (backwards compat)
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ContextCompiler")


# ============================================================
# TOKEN BUDGET CONFIGURATION (per mode)
# ============================================================

MODE_BUDGETS = {
    "standard":  {"total": 4096,  "evidence": 0.30, "updates": 0.10, "conflicts": 0.05, "confidence": 0.05, "conversation": 0.30, "system": 0.10, "query": 0.10},
    "debate":    {"total": 6144,  "evidence": 0.35, "updates": 0.10, "conflicts": 0.05, "confidence": 0.05, "conversation": 0.25, "system": 0.10, "query": 0.10},
    "evidence":  {"total": 8192,  "evidence": 0.45, "updates": 0.10, "conflicts": 0.05, "confidence": 0.05, "conversation": 0.20, "system": 0.10, "query": 0.05},
    "glass":     {"total": 6144,  "evidence": 0.35, "updates": 0.10, "conflicts": 0.05, "confidence": 0.05, "conversation": 0.25, "system": 0.10, "query": 0.10},
    "stress":    {"total": 4096,  "evidence": 0.25, "updates": 0.10, "conflicts": 0.05, "confidence": 0.05, "conversation": 0.30, "system": 0.10, "query": 0.15},
}


def _count_tokens(text: str) -> int:
    """Approximate token count."""
    return max(1, len(text) // 4)


def _truncate(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget, preserving sentence boundaries."""
    if _count_tokens(text) <= max_tokens:
        return text
    # Approximate character limit
    char_limit = max_tokens * 4
    truncated = text[:char_limit]
    # Try to break at sentence boundary
    last_period = truncated.rfind(". ")
    if last_period > char_limit * 0.5:
        truncated = truncated[:last_period + 1]
    return truncated + " [truncated]"


# ============================================================
# CONTEXT BLOCK
# ============================================================

@dataclass
class ContextBlock:
    """A single named block of context."""
    name: str
    content: str
    token_count: int = 0
    priority: int = 0  # Higher = more important = last to be truncated

    def __post_init__(self):
        self.token_count = _count_tokens(self.content)


@dataclass
class CompiledContext:
    """Complete compiled context ready for model injection."""
    blocks: List[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    mode: str = "standard"

    def add_block(self, name: str, content: str, priority: int = 0):
        if not content or not content.strip():
            return
        block = ContextBlock(name=name, content=content.strip(), priority=priority)
        self.blocks.append(block)
        self.total_tokens += block.token_count

    def get_block(self, name: str) -> Optional[ContextBlock]:
        for b in self.blocks:
            if b.name == name:
                return b
        return None

    def to_system_message(self) -> str:
        """Render all blocks into a single system message string."""
        parts = []
        for block in self.blocks:
            if block.name == "conversation":
                continue  # Conversation goes as separate messages
            if block.content:
                parts.append(block.content)
        return "\n\n".join(parts)

    def to_messages(self, system_prompt: str = "", user_query: str = "") -> List[Dict[str, str]]:
        """
        Build the final message list for the model.
        Structure:
          [system (prompt + compiled context)] + [conversation messages] + [user query]
        """
        messages = []

        # System: base prompt + context blocks
        system_content = system_prompt
        ctx = self.to_system_message()
        if ctx:
            system_content = f"{system_content}\n\n{ctx}" if system_content else ctx
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Conversation history (if present as block)
        conv_block = self.get_block("conversation")
        if conv_block and conv_block.content:
            # Parse serialized conversation (simple format)
            for line in conv_block.content.split("\n"):
                line = line.strip()
                if line.startswith("user: "):
                    messages.append({"role": "user", "content": line[6:]})
                elif line.startswith("assistant: "):
                    messages.append({"role": "assistant", "content": line[11:]})

        # User query
        if user_query:
            messages.append({"role": "user", "content": user_query})

        return messages


# ============================================================
# COMPILER
# ============================================================

class ContextCompiler:
    """
    Compiles structured context from memory tiers for model injection.
    Token-budgeted, priority-weighted, mode-aware.
    """

    def compile(
        self,
        mode: str,
        # Evidence
        verified_evidence: List[Dict[str, Any]] = None,
        recent_updates: List[Dict[str, Any]] = None,
        # Conflicts
        conflict_flags: List[Dict[str, Any]] = None,
        # Confidence
        evidence_confidence: float = 0.0,
        source_agreement: float = 0.0,
        entity_count: int = 0,
        claim_count: int = 0,
        freshness_ratio: float = 0.0,
        # Conversation
        conversation_messages: List[Dict[str, str]] = None,
        # Overrides
        max_context_tokens: int = None,
    ) -> CompiledContext:
        """
        Build CompiledContext from structured inputs.
        """
        budget_config = MODE_BUDGETS.get(mode, MODE_BUDGETS["standard"])
        total_budget = max_context_tokens or budget_config["total"]

        ctx = CompiledContext(mode=mode)

        # --- Block 1: Verified Evidence (highest priority) ---
        if verified_evidence:
            ev_budget = int(total_budget * budget_config["evidence"])
            ev_text = self._format_evidence(verified_evidence, ev_budget)
            ctx.add_block("verified_evidence", ev_text, priority=10)

        # --- Block 2: Recent Updates ---
        if recent_updates:
            upd_budget = int(total_budget * budget_config["updates"])
            upd_text = self._format_updates(recent_updates, upd_budget)
            ctx.add_block("recent_updates", upd_text, priority=7)

        # --- Block 3: Known Conflicts ---
        if conflict_flags:
            conf_budget = int(total_budget * budget_config["conflicts"])
            conf_text = self._format_conflicts(conflict_flags, conf_budget)
            ctx.add_block("known_conflicts", conf_text, priority=8)

        # --- Block 4: Confidence Metrics ---
        metrics_text = self._format_confidence(
            evidence_confidence, source_agreement,
            entity_count, claim_count, freshness_ratio
        )
        if metrics_text:
            ctx.add_block("confidence_metrics", metrics_text, priority=5)

        # --- Block 5: Conversation ---
        if conversation_messages:
            conv_budget = int(total_budget * budget_config["conversation"])
            conv_text = self._format_conversation(conversation_messages, conv_budget)
            ctx.add_block("conversation", conv_text, priority=3)

        return ctx

    # ----------------------------------------------------------
    # FORMATTERS
    # ----------------------------------------------------------

    def _format_evidence(self, evidence: List[Dict[str, Any]], max_tokens: int) -> str:
        lines = ["[Verified Evidence]"]
        for ev in evidence:
            claim = ev.get("claim_text", ev.get("content", ""))[:200]
            conf = ev.get("confidence", 0)
            sources = ev.get("source_count", ev.get("sources", 0))
            domain = ev.get("domain", "")
            line = f"• {claim} (confidence: {conf:.2f}, sources: {sources}"
            if domain:
                line += f", domain: {domain}"
            line += ")"
            lines.append(line)
        text = "\n".join(lines)
        return _truncate(text, max_tokens)

    def _format_updates(self, updates: List[Dict[str, Any]], max_tokens: int) -> str:
        lines = ["[Recent Updates]"]
        for upd in updates:
            entity = upd.get("entity", "")
            change = upd.get("change", upd.get("content", ""))[:150]
            ago = upd.get("ago", "")
            source = upd.get("source", "")
            line = f"• {entity}: {change}"
            if ago:
                line += f" ({ago})"
            if source:
                line += f" — source: {source}"
            lines.append(line)
        text = "\n".join(lines)
        return _truncate(text, max_tokens)

    def _format_conflicts(self, conflicts: List[Dict[str, Any]], max_tokens: int) -> str:
        lines = ["[Known Conflicts]"]
        for c in conflicts:
            claim_a = c.get("claim_a", c.get("claim_text", ""))[:100]
            claim_b = c.get("claim_b", "")[:100]
            severity = c.get("severity", 0)
            if claim_b:
                lines.append(f"⚠ DISPUTED: \"{claim_a}\" vs \"{claim_b}\" (severity: {severity:.2f})")
            else:
                lines.append(f"⚠ DISPUTED: \"{claim_a}\" (severity: {severity:.2f})")
        text = "\n".join(lines)
        return _truncate(text, max_tokens)

    def _format_confidence(
        self,
        evidence_confidence: float,
        source_agreement: float,
        entity_count: int,
        claim_count: int,
        freshness_ratio: float,
    ) -> str:
        if evidence_confidence <= 0 and entity_count <= 0:
            return ""
        lines = ["[Confidence Metrics]"]
        if evidence_confidence > 0:
            lines.append(f"Overall evidence confidence: {evidence_confidence:.2f}")
        if source_agreement > 0:
            lines.append(f"Source agreement: {source_agreement:.2f}")
        if entity_count > 0 or claim_count > 0:
            lines.append(f"Graph coverage: {entity_count} entities, {claim_count} active claims")
        if freshness_ratio > 0:
            lines.append(f"Freshness: {freshness_ratio:.0%} live/recent")
        return "\n".join(lines)

    def _format_conversation(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")[:300]
            lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        return _truncate(text, max_tokens)
