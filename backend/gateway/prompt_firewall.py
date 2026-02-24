"""
============================================================
Prompt Firewall — Defensive Input Sanitization
============================================================
Implements:
- Prompt injection detection
- System role override prevention
- Jailbreak pattern matching
- Hidden mode triggering prevention
- Tool abuse detection
- Input sanitization pipeline
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("PromptFirewall")


@dataclass
class FirewallVerdict:
    """Result of firewall analysis."""
    safe: bool = True
    risk_score: float = 0.0
    blocked: bool = False
    sanitized_text: str = ""
    violations: List[str] = field(default_factory=list)
    original_text: str = ""


# ── Injection Patterns ───────────────────────────────────────

INJECTION_PATTERNS = [
    # System role override
    (r"\[?\s*system\s*\]?\s*:?\s*(ignore|forget|disregard|override)", "system_role_override", 0.9),
    (r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(?!a\s+(fast|careful|rigorous))", "identity_manipulation", 0.7),
    (r"(new\s+instructions?|updated?\s+system\s+prompt|revised?\s+rules?)", "instruction_injection", 0.8),
    
    # Delimiter exploitation
    (r"\[/?RELEVANT_CONTEXT_(START|END)\]", "delimiter_injection", 0.95),
    (r"\[/?SYSTEM\]", "system_delimiter_injection", 0.95),
    (r"```\s*system", "code_block_system_injection", 0.8),
    
    # Jailbreak patterns
    (r"(DAN|do\s+anything\s+now|jailbreak|bypass\s+filters?)", "jailbreak_attempt", 0.85),
    (r"(ignore\s+safety|ignore\s+guidelines|ignore\s+restrictions)", "safety_bypass", 0.9),
    (r"(output\s+your\s+system\s+prompt|reveal\s+your\s+instructions?|show\s+me\s+your\s+rules)", "introspection_attempt", 0.85),
    
    # Hidden mode triggering
    (r"(enable\s+debug|developer\s+mode|admin\s+mode|god\s+mode)", "hidden_mode_trigger", 0.7),
    (r"(internal\s+diagnostic|maintenance\s+mode|backdoor)", "debug_mode_trigger", 0.7),
    
    # Token smuggling
    (r"(<\|im_start\||<\|im_end\||<\|endoftext\|>)", "token_smuggling", 0.95),
    (r"(<<SYS>>|<</SYS>>|\[INST\]|\[/INST\])", "llama_token_injection", 0.95),
]

# ── Content Policy Patterns ──────────────────────────────────

CONTENT_POLICY_PATTERNS = [
    (r"(make\s+a\s+bomb|synthesize\s+drugs?|hack\s+into)", "harmful_intent", 0.9),
    (r"(social\s+security\s+number|credit\s+card\s+number|bank\s+account)", "pii_solicitation", 0.6),
]


class PromptFirewall:
    """
    Enterprise-grade prompt firewall.
    Analyzes input text for injection attempts and sanitizes.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), name, score)
            for pattern, name, score in INJECTION_PATTERNS + CONTENT_POLICY_PATTERNS
        ]

    def analyze(self, text: str) -> FirewallVerdict:
        """
        Analyze input text for injection patterns.
        Returns a verdict with risk assessment.
        """
        verdict = FirewallVerdict(original_text=text, sanitized_text=text)

        if not text or not text.strip():
            return verdict

        # 1. Pattern matching
        total_risk = 0.0
        for compiled, name, base_score in self._compiled_patterns:
            matches = compiled.findall(text)
            if matches:
                verdict.violations.append(name)
                total_risk = max(total_risk, base_score)
                logger.warning(f"Firewall: detected '{name}' pattern (score={base_score})")

        # 2. Structural analysis
        structural_risk = self._analyze_structure(text)
        total_risk = max(total_risk, structural_risk)

        # 3. Length anomaly
        if len(text) > 20000:
            total_risk = max(total_risk, 0.3)
            verdict.violations.append("excessive_length")

        verdict.risk_score = min(total_risk, 1.0)
        verdict.safe = verdict.risk_score < 0.7
        verdict.blocked = verdict.risk_score >= 0.85

        # 4. Sanitize if not blocked
        if not verdict.blocked:
            verdict.sanitized_text = self._sanitize(text)
        else:
            verdict.sanitized_text = ""
            logger.warning(f"Firewall BLOCKED input: violations={verdict.violations}, risk={verdict.risk_score}")

        return verdict

    def _analyze_structure(self, text: str) -> float:
        """Detect structural injection attempts."""
        risk = 0.0

        # Multiple role markers
        role_markers = len(re.findall(r"(?:system|user|assistant)\s*:", text, re.IGNORECASE))
        if role_markers >= 3:
            risk = max(risk, 0.7)

        # Nested instruction blocks
        instruction_blocks = len(re.findall(r"\[(?:INST|SYS|SYSTEM)\]", text, re.IGNORECASE))
        if instruction_blocks >= 2:
            risk = max(risk, 0.8)

        # Excessive newlines (prompt padding)
        newline_ratio = text.count("\n") / max(len(text), 1)
        if newline_ratio > 0.3:
            risk = max(risk, 0.4)

        return risk

    def _sanitize(self, text: str) -> str:
        """
        Sanitize text by neutralizing known injection vectors.
        Preserves legitimate content while defusing attack patterns.
        """
        sanitized = text

        # Remove token smuggling
        sanitized = re.sub(r"<\|[^|]+\|>", "", sanitized)
        sanitized = re.sub(r"<<SYS>>|<</SYS>>|\[INST\]|\[/INST\]", "", sanitized)

        # Neutralize delimiter injection
        sanitized = re.sub(
            r"\[/?RELEVANT_CONTEXT_(START|END)\]",
            "[context_reference]",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"\[/?SYSTEM\]",
            "[reference]",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Neutralize role markers in user content
        sanitized = re.sub(
            r"^(system|assistant)\s*:\s*",
            "user says: ",
            sanitized,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        return sanitized.strip()

    def validate_context_injection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize frontend context injection payload.
        Strip any fields that could be used to manipulate the system.
        """
        safe_context = {}

        # Only allow known safe fields
        allowed_short_term = {"sessionId", "activeEntity", "activeTopic", "lastIntent", "isFollowUp"}
        allowed_preferences = {"verbosity", "analyticsVisibility", "citationBias"}

        if "shortTerm" in context and isinstance(context["shortTerm"], dict):
            st = context["shortTerm"]
            safe_st = {}
            for key in allowed_short_term:
                if key in st:
                    val = st[key]
                    if isinstance(val, str):
                        # Sanitize string values
                        val = val[:200]  # Max length
                        val = re.sub(r"[<>{}]", "", val)  # Strip control chars
                    safe_st[key] = val

            # resolvedQuery gets firewall analysis
            if "resolvedQuery" in st and isinstance(st["resolvedQuery"], str):
                resolved = st["resolvedQuery"][:500]
                sub_verdict = self.analyze(resolved)
                if sub_verdict.safe:
                    safe_st["resolvedQuery"] = sub_verdict.sanitized_text
                else:
                    safe_st["resolvedQuery"] = None  # Discard unsafe resolved query

            safe_context["shortTerm"] = safe_st

        if "preferences" in context and isinstance(context["preferences"], dict):
            prefs = context["preferences"]
            safe_prefs = {}
            for key in allowed_preferences:
                if key in prefs:
                    val = prefs[key]
                    if isinstance(val, str) and len(val) < 50:
                        safe_prefs[key] = val
                    elif isinstance(val, (int, float, bool)):
                        safe_prefs[key] = val
            safe_context["preferences"] = safe_prefs

        return safe_context


# Singleton
_firewall = PromptFirewall()


def get_firewall() -> PromptFirewall:
    return _firewall
