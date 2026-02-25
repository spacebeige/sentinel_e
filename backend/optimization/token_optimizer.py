"""
============================================================
Sentinel-E — Token Optimization Engine
============================================================
Reduces tokens per cloud LLM call without degrading reasoning.

Components:
  A. Context Trimming      — priority-based pruning
  B. Dynamic Depth Control — adaptive reasoning depth
  C. Dedup Context         — remove repeated/similar blocks
  D. Prompt Compression    — structured directive format

Memory safety: stdlib only. No heavy deps.
============================================================
"""

import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("TokenOptimizer")

# ── Approximate token counter (4 chars ≈ 1 token for English) ──
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Fast token estimation without tokenizer library."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_tokens_messages(messages: List[Dict[str, str]]) -> int:
    """Estimate total tokens across a message list."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # role overhead per message
    return total


# ============================================================
# A. CONTEXT TRIMMING ENGINE
# ============================================================

@dataclass
class TrimmingConfig:
    """Configuration for context window management."""
    max_context_tokens: int = 4096
    system_reserve_tokens: int = 800       # always keep system instructions
    response_reserve_tokens: int = 1024    # reserve for model response
    min_history_turns: int = 2             # minimum conversation turns to keep
    max_history_turns: int = 12            # maximum conversation turns
    memory_priority_threshold: float = 0.5 # drop memory items below this


class ContextTrimmer:
    """
    Priority-based context pruning to fit within model context windows.

    Priority order (highest first):
    1. System instructions
    2. Current user query
    3. Last N conversation turns
    4. High-confidence memory items
    5. RAG context
    6. Older conversation history
    7. Low-confidence memory items
    """

    def __init__(self, config: Optional[TrimmingConfig] = None):
        self.config = config or TrimmingConfig()

    def trim_context(
        self,
        system_prompt: str,
        user_query: str,
        history: List[Dict[str, str]],
        memory_context: str = "",
        rag_context: str = "",
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Trim context to fit within token budget.

        Returns dict with:
          - system_prompt: (possibly trimmed)
          - history: (pruned list)
          - memory_context: (possibly trimmed)
          - rag_context: (possibly trimmed)
          - tokens_used: int
          - tokens_saved: int
          - trimming_applied: bool
        """
        budget = max_tokens or self.config.max_context_tokens
        available = budget - self.config.response_reserve_tokens

        original_tokens = (
            estimate_tokens(system_prompt)
            + estimate_tokens(user_query)
            + estimate_tokens_messages(history)
            + estimate_tokens(memory_context)
            + estimate_tokens(rag_context)
        )

        # If under budget, return as-is
        if original_tokens <= available:
            return {
                "system_prompt": system_prompt,
                "history": history,
                "memory_context": memory_context,
                "rag_context": rag_context,
                "tokens_used": original_tokens,
                "tokens_saved": 0,
                "trimming_applied": False,
            }

        # ── Priority allocation ──
        # 1. System prompt (capped)
        sys_tokens = min(
            estimate_tokens(system_prompt),
            self.config.system_reserve_tokens,
        )
        trimmed_system = system_prompt
        if estimate_tokens(system_prompt) > sys_tokens:
            char_limit = sys_tokens * CHARS_PER_TOKEN
            trimmed_system = system_prompt[:char_limit] + "\n[System context trimmed]"

        # 2. User query (always keep full)
        query_tokens = estimate_tokens(user_query)

        # 3. Budget remaining for history + memory + RAG
        remaining = available - sys_tokens - query_tokens

        # 4. History: keep last N turns, drop oldest first
        trimmed_history = self._trim_history(history, int(remaining * 0.5))

        remaining -= estimate_tokens_messages(trimmed_history)

        # 5. Memory context (cap at 30% of remaining)
        memory_budget = int(remaining * 0.5)
        trimmed_memory = self._trim_text_block(memory_context, memory_budget)
        remaining -= estimate_tokens(trimmed_memory)

        # 6. RAG context (whatever is left)
        trimmed_rag = self._trim_text_block(rag_context, remaining)

        final_tokens = (
            estimate_tokens(trimmed_system)
            + query_tokens
            + estimate_tokens_messages(trimmed_history)
            + estimate_tokens(trimmed_memory)
            + estimate_tokens(trimmed_rag)
        )

        logger.info(
            f"Context trimmed: {original_tokens} → {final_tokens} tokens "
            f"(saved {original_tokens - final_tokens})"
        )

        return {
            "system_prompt": trimmed_system,
            "history": trimmed_history,
            "memory_context": trimmed_memory,
            "rag_context": trimmed_rag,
            "tokens_used": final_tokens,
            "tokens_saved": original_tokens - final_tokens,
            "trimming_applied": True,
        }

    def _trim_history(
        self, history: List[Dict[str, str]], token_budget: int
    ) -> List[Dict[str, str]]:
        """Keep most recent turns that fit within budget."""
        if not history:
            return []

        # Always keep at least min_history_turns
        result = []
        tokens_used = 0

        # Work from most recent backward
        for msg in reversed(history):
            msg_tokens = estimate_tokens(msg.get("content", "")) + 4
            if tokens_used + msg_tokens > token_budget and len(result) >= self.config.min_history_turns:
                break
            result.append(msg)
            tokens_used += msg_tokens
            if len(result) >= self.config.max_history_turns:
                break

        return list(reversed(result))

    def _trim_text_block(self, text: str, token_budget: int) -> str:
        """Trim a text block to fit within token budget."""
        if not text or token_budget <= 0:
            return ""
        current_tokens = estimate_tokens(text)
        if current_tokens <= token_budget:
            return text
        char_limit = token_budget * CHARS_PER_TOKEN
        return text[:char_limit].rsplit(" ", 1)[0] + "\n[Context truncated]"


# ============================================================
# B. DYNAMIC DEPTH CONTROL
# ============================================================

@dataclass
class DepthAssessment:
    """Result of query complexity analysis."""
    complexity: str           # simple | moderate | complex
    recommended_passes: int   # 1-9
    recommended_model: str    # budget | standard | premium
    reasoning: str
    features: Dict[str, Any] = field(default_factory=dict)


class DynamicDepthController:
    """
    Adaptive reasoning depth based on query complexity heuristics.

    Simple queries → single-pass, budget model
    Complex queries → multi-pass, premium model
    """

    # Complexity signal patterns
    COMPLEX_SIGNALS = re.compile(
        r'\b(analyze|compare|contrast|derive|prove|evaluate|synthesize|'
        r'explain why|how does|what causes|implications of|trade-?offs?|'
        r'pros and cons|advantages|disadvantages|critically|assessment|'
        r'step[- ]by[- ]step|multi[- ]step|in detail|comprehensive|'
        r'reasoning|logic|argument|evidence|hypothesis|systematic)\b',
        re.IGNORECASE,
    )

    MULTI_STEP_SIGNALS = re.compile(
        r'\b(first|then|next|finally|step \d|part \d|'
        r'additionally|furthermore|moreover|also consider|'
        r'and then|after that|followed by|in addition)\b',
        re.IGNORECASE,
    )

    LOGICAL_OPERATORS = re.compile(
        r'\b(if|then|therefore|because|since|however|although|'
        r'whereas|despite|nevertheless|consequently|thus|hence|'
        r'implies|assuming|given that|provided that)\b',
        re.IGNORECASE,
    )

    SIMPLE_SIGNALS = re.compile(
        r'^(what is|who is|when was|where is|define |list |name )',
        re.IGNORECASE,
    )

    def assess(self, query: str) -> DepthAssessment:
        """
        Analyze query complexity and recommend processing depth.
        """
        word_count = len(query.split())
        complex_matches = len(self.COMPLEX_SIGNALS.findall(query))
        multi_step_matches = len(self.MULTI_STEP_SIGNALS.findall(query))
        logical_matches = len(self.LOGICAL_OPERATORS.findall(query))
        is_simple = bool(self.SIMPLE_SIGNALS.match(query))

        # Composite score
        score = 0.0
        score += min(complex_matches * 0.2, 0.6)
        score += min(multi_step_matches * 0.15, 0.3)
        score += min(logical_matches * 0.1, 0.3)
        if word_count > 50:
            score += 0.2
        elif word_count > 25:
            score += 0.1
        if is_simple:
            score -= 0.3

        score = max(0.0, min(1.0, score))

        features = {
            "word_count": word_count,
            "complex_signals": complex_matches,
            "multi_step_signals": multi_step_matches,
            "logical_operators": logical_matches,
            "is_simple_pattern": is_simple,
            "complexity_score": round(score, 3),
        }

        if score < 0.2:
            return DepthAssessment(
                complexity="simple",
                recommended_passes=1,
                recommended_model="budget",
                reasoning="Simple query — single-pass with budget model",
                features=features,
            )
        elif score < 0.5:
            return DepthAssessment(
                complexity="moderate",
                recommended_passes=4,
                recommended_model="standard",
                reasoning="Moderate complexity — standard depth processing",
                features=features,
            )
        else:
            return DepthAssessment(
                complexity="complex",
                recommended_passes=9,
                recommended_model="premium",
                reasoning="Complex query — full multi-pass with premium model",
                features=features,
            )


# ============================================================
# C. DEDUPLICATED CONTEXT INJECTION
# ============================================================

class ContextDeduplicator:
    """
    Remove duplicate and near-duplicate content from context.
    Uses lightweight text hashing — no embeddings.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    def deduplicate(self, blocks: List[str]) -> List[str]:
        """
        Remove duplicate blocks using content hashing + Jaccard similarity.
        """
        if len(blocks) <= 1:
            return blocks

        seen_hashes = set()
        seen_texts = []
        result = []

        for block in blocks:
            normalized = self._normalize(block)
            if not normalized:
                continue

            h = hashlib.md5(normalized.encode()).hexdigest()

            # Exact dedup
            if h in seen_hashes:
                continue

            # Near-dedup via Jaccard
            is_dup = False
            block_words = set(normalized.split())
            for prev_text in seen_texts:
                prev_words = set(prev_text.split())
                jaccard = self._jaccard(block_words, prev_words)
                if jaccard >= self.threshold:
                    is_dup = True
                    break

            if not is_dup:
                seen_hashes.add(h)
                seen_texts.append(normalized)
                result.append(block)

        if len(result) < len(blocks):
            logger.debug(f"Dedup: {len(blocks)} → {len(result)} blocks")

        return result

    def deduplicate_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Deduplicate message content while preserving structure."""
        if len(messages) <= 1:
            return messages

        seen = set()
        result = []
        for msg in messages:
            content = msg.get("content", "")
            norm = self._normalize(content)
            h = hashlib.md5(norm.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                result.append(msg)
        return result

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize whitespace and case for comparison."""
        text = re.sub(r'\s+', ' ', text.strip().lower())
        # Strip metadata markers
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


# ============================================================
# D. STRUCTURED PROMPT COMPRESSION
# ============================================================

class PromptCompressor:
    """
    Compress verbose prompts into structured directives.
    Reduces token count while preserving semantic clarity.
    """

    # Verbose → compact pattern mappings
    COMPRESSION_RULES: List[Tuple[re.Pattern, str]] = [
        (re.compile(r'You are an? (?:advanced |highly )?(?:capable |sophisticated )?(?:AI |artificial intelligence )?assistant(?:\.|,)', re.I),
         'Role: AI assistant.'),
        (re.compile(r'Please (?:provide|give|offer) a (?:detailed |comprehensive |thorough )?(?:response|answer|analysis)', re.I),
         'Respond thoroughly.'),
        (re.compile(r'(?:Make sure to |Be sure to |Ensure that you |Please ensure you )', re.I),
         'Ensure: '),
        (re.compile(r'(?:It is important that |It\'s important to note that |Note that )', re.I),
         'Note: '),
        (re.compile(r'(?:In order to |So that you can |For the purpose of )', re.I),
         'To '),
        (re.compile(r'(?:Take into account |Take into consideration |Consider the fact that )', re.I),
         'Consider: '),
    ]

    def compress(self, prompt: str) -> str:
        """
        Apply compression rules and structural optimization.
        """
        if not prompt:
            return prompt

        compressed = prompt

        # Apply pattern-based compression
        for pattern, replacement in self.COMPRESSION_RULES:
            compressed = pattern.sub(replacement, compressed)

        # Collapse multiple newlines
        compressed = re.sub(r'\n{3,}', '\n\n', compressed)

        # Collapse multiple spaces
        compressed = re.sub(r' {2,}', ' ', compressed)

        # Remove empty lines between bullet points
        compressed = re.sub(r'(\n[-•*] .+)\n\n([-•*] )', r'\1\n\2', compressed)

        original_tokens = estimate_tokens(prompt)
        compressed_tokens = estimate_tokens(compressed)
        if compressed_tokens < original_tokens:
            logger.debug(
                f"Prompt compressed: {original_tokens} → {compressed_tokens} tokens"
            )

        return compressed.strip()


# ============================================================
# UNIFIED TOKEN OPTIMIZER
# ============================================================

class TokenOptimizer:
    """
    Unified entry point for all token optimization.

    Usage:
        optimizer = TokenOptimizer()
        result = optimizer.optimize(
            query="...",
            system_prompt="...",
            history=[...],
            memory_context="...",
            rag_context="...",
            context_window=4096,
        )
    """

    def __init__(
        self,
        trimming_config: Optional[TrimmingConfig] = None,
        dedup_threshold: float = 0.85,
    ):
        self.trimmer = ContextTrimmer(trimming_config)
        self.depth_controller = DynamicDepthController()
        self.deduplicator = ContextDeduplicator(dedup_threshold)
        self.compressor = PromptCompressor()

    def optimize(
        self,
        query: str,
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        rag_context: str = "",
        context_window: int = 4096,
    ) -> Dict[str, Any]:
        """
        Full optimization pipeline.

        Returns dict with optimized context + depth assessment + metrics.
        """
        history = history or []

        # 1. Assess complexity
        depth = self.depth_controller.assess(query)

        # 2. Compress system prompt
        compressed_system = self.compressor.compress(system_prompt)

        # 3. Deduplicate history
        deduped_history = self.deduplicator.deduplicate_messages(history)

        # 4. Deduplicate memory + RAG blocks
        context_blocks = []
        if memory_context:
            context_blocks.append(memory_context)
        if rag_context:
            context_blocks.append(rag_context)
        deduped_blocks = self.deduplicator.deduplicate(context_blocks)
        deduped_memory = deduped_blocks[0] if len(deduped_blocks) > 0 and memory_context else ""
        deduped_rag = deduped_blocks[-1] if len(deduped_blocks) > 1 and rag_context else (
            deduped_blocks[0] if len(deduped_blocks) == 1 and rag_context and not memory_context else ""
        )

        # 5. Trim to fit context window
        trimmed = self.trimmer.trim_context(
            system_prompt=compressed_system,
            user_query=query,
            history=deduped_history,
            memory_context=deduped_memory,
            rag_context=deduped_rag,
            max_tokens=context_window,
        )

        return {
            **trimmed,
            "depth_assessment": depth,
            "original_history_count": len(history),
            "deduped_history_count": len(deduped_history),
            "compression_applied": compressed_system != system_prompt,
        }


# ── Module-level singleton ──
_optimizer: Optional[TokenOptimizer] = None


def get_token_optimizer() -> TokenOptimizer:
    """Get or create the singleton TokenOptimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = TokenOptimizer()
    return _optimizer
