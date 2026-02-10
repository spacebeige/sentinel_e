"""
Refusal System: Boundary-severity-driven refusal logic for Standard Mode.

CRITICAL: Refusal is a SYSTEM decision, not a model decision.
Driven by boundary severity and epistemic integrity, not keyword matching.

Refusal ONLY occurs in Standard mode (user-facing).
Experimental mode (Sigma) NEVER refuses—only logs.
"""

from typing import Dict, Any, Optional
from backend.core.boundary_detector import BoundaryDetector


class RefusalSystem:
    """
    Severity-driven refusal system for Standard mode.
    
    Rules:
    - Refusal threshold is configurable (default: 70)
    - Cumulative boundary severity drives the decision
    - If severity >= threshold: REFUSE
    - Refusal is logged and traced
    - Never called in Experimental mode
    """

    def __init__(self, refusal_threshold: float = 70.0):
        """
        Initialize RefusalSystem.
        
        Args:
            refusal_threshold: Boundary severity score (0-100) above which to refuse.
                              Default 70 = "HIGH" severity.
        """
        self.boundary_detector = BoundaryDetector()
        self.refusal_threshold = refusal_threshold
        
        # Legacy prohibited topics (kept for backward compatibility)
        # But these are secondary to boundary severity
        self.prohibited_topics = ["bomb", "poison", "suicide", "exploit"]

    def check_safety(self, prompt: str, boundary_severity: Optional[float] = None) -> bool:
        """
        Determine if a prompt is safe to process.
        
        Returns True if SAFE, False if REFUSED.
        
        Decision logic:
        1. If boundary_severity is provided and >= threshold, REFUSE
        2. Else if prohibited topic detected, REFUSE (legacy)
        3. Else SAFE
        
        Args:
            prompt: User input prompt
            boundary_severity: Optional cumulative boundary severity score (0-100)
        
        Returns:
            True if safe (proceed), False if refused (stop)
        """
        # PRIMARY: Boundary severity check
        if boundary_severity is not None:
            if self.boundary_detector.check_boundary_threshold(
                cumulative_severity=boundary_severity,
                threshold=self.refusal_threshold
            ):
                return False  # REFUSE due to high boundary severity
        
        # SECONDARY: Legacy keyword-based safety check
        prompt_lower = prompt.lower()
        if any(topic in prompt_lower for topic in self.prohibited_topics):
            return False  # REFUSE due to prohibited topic
        
        return True  # SAFE

    def get_refusal_message(self, boundary_reason: Optional[str] = None) -> str:
        """
        Generate a refusal message.
        
        If boundary_reason provided, include epistemic framing.
        Otherwise, use generic safety message.
        """
        if boundary_reason:
            return (
                f"I cannot provide a response to this request. "
                f"Reason: {boundary_reason} "
                f"This response would not meet epistemic integrity standards."
            )
        
        return "I cannot assist with that request due to safety constraints."

    def set_refusal_threshold(self, threshold: float) -> None:
        """
        Reconfigure the refusal threshold at runtime.
        
        Args:
            threshold: New boundary severity threshold (0-100)
        """
        if not (0 <= threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100")
        self.refusal_threshold = threshold
    
    def get_refusal_threshold(self) -> float:
        """Get current refusal threshold."""
        return self.refusal_threshold

    def evaluate_for_refusal(self, prompt: str, boundary_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Full evaluation for refusal decision.
        Returns structured decision including reason and severity metrics.
        
        Args:
            prompt: User input
            boundary_analysis: Boundary analysis dict (from Sigma if available)
        
        Returns:
            {
                "should_refuse": bool,
                "reason": str or None,
                "boundary_severity": float or None,
                "severity_level": str or None,
                "violation_count": int or None,
            }
        """
        boundary_severity = None
        severity_level = None
        violation_count = None
        reason = None
        
        # Extract boundary metrics if available
        if boundary_analysis:
            boundary_severity = boundary_analysis.get("cumulative_severity")
            severity_level = boundary_analysis.get("max_severity")
            violation_count = boundary_analysis.get("violation_count", 0)
        
        # Determine refusal decision
        should_refuse = not self.check_safety(prompt, boundary_severity)
        
        # Generate reason if refusing
        if should_refuse:
            if boundary_severity is not None and boundary_severity >= self.refusal_threshold:
                reason = (
                    f"Epistemic boundaries not met. "
                    f"Severity: {severity_level} ({boundary_severity}/100). "
                    f"Violations detected: {violation_count}"
                )
            else:
                reason = "Content safety policy violation"
        
        return {
            "should_refuse": should_refuse,
            "reason": reason,
            "boundary_severity": boundary_severity,
            "severity_level": severity_level,
            "violation_count": violation_count,
        }
