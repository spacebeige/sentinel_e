"""
Structured report generator for Sentinel-E compressed pipeline.
Parses debate output into the standard Sentinel Analysis Report format.
"""

import re
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

from compressed.debate_engine import DebateResult

logger = logging.getLogger("compressed.report")


@dataclass
class SentinelReport:
    """Structured Sentinel Analysis Report."""
    overview: str = ""
    assessment: str = ""
    critical_issues: str = ""
    tactical_map: str = ""
    final_synthesis: str = ""
    confidence_score: int = 70
    confidence_justification: str = ""

    # Metadata
    query: str = ""
    api_calls: int = 0
    total_tokens: int = 0
    models_used: list = field(default_factory=list)
    latency_ms: float = 0.0
    search_used: bool = False

    def to_formatted_output(self) -> str:
        """Render as the canonical Sentinel report format."""
        lines = [
            "━" * 60,
            "SENTINEL ANALYSIS REPORT",
            "━" * 60,
            "",
            "## OVERVIEW",
            self.overview or "No overview available.",
            "",
            "## ASSESSMENT",
            self.assessment or "No assessment available.",
            "",
            "## CRITICAL ISSUES",
            self.critical_issues or "No critical issues identified.",
            "",
            "## TACTICAL MAP",
            self.tactical_map or "No tactical options mapped.",
            "",
            "## FINAL SYNTHESIS",
            self.final_synthesis or "No synthesis available.",
            "",
            "## CONFIDENCE",
            f"{self.confidence_score}% — {self.confidence_justification}" if self.confidence_justification else f"{self.confidence_score}%",
            "",
            "━" * 60,
            f"Pipeline: {self.api_calls} API calls | {self.total_tokens} tokens | {self.latency_ms:.0f}ms",
            f"Models: {', '.join(self.models_used) if self.models_used else 'N/A'}",
            f"Web search: {'Yes' if self.search_used else 'No'}",
            "━" * 60,
        ]
        return "\n".join(lines)

    def to_metadata(self) -> Dict:
        """Return omega_metadata-compatible dict."""
        return {
            "version": "sigma-compressed-1.0",
            "mode": "compressed",
            "engine": "CondensedDebateEngine",
            "confidence": self.confidence_score / 100.0,
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "models_used": self.models_used,
            "latency_ms": self.latency_ms,
            "search_used": self.search_used,
            "debate_steps": {
                "analysis": True,
                "critique": True,
                "synthesis": True,
            },
        }


class ReportGenerator:
    """Parses debate synthesis output into structured report sections."""

    # Section patterns (flexible matching)
    _SECTION_PATTERNS = {
        "overview": r"(?:##?\s*)?OVERVIEW\s*\n(.*?)(?=(?:##?\s*)?(?:ASSESSMENT|CRITICAL|TACTICAL|FINAL|CONFIDENCE)|$)",
        "assessment": r"(?:##?\s*)?ASSESSMENT\s*\n(.*?)(?=(?:##?\s*)?(?:CRITICAL|TACTICAL|FINAL|CONFIDENCE)|$)",
        "critical_issues": r"(?:##?\s*)?CRITICAL\s*(?:ISSUES?)?\s*\n(.*?)(?=(?:##?\s*)?(?:TACTICAL|FINAL|CONFIDENCE)|$)",
        "tactical_map": r"(?:##?\s*)?TACTICAL\s*(?:MAP)?\s*\n(.*?)(?=(?:##?\s*)?(?:FINAL|CONFIDENCE)|$)",
        "final_synthesis": r"(?:##?\s*)?FINAL\s*(?:SYNTHESIS)?\s*\n(.*?)(?=(?:##?\s*)?CONFIDENCE|$)",
        "confidence": r"(?:##?\s*)?CONFIDENCE\s*(?:SCORE?)?\s*\n(.*?)$",
    }

    def generate(self, debate_result: DebateResult, search_used: bool = False) -> SentinelReport:
        """Parse debate output into structured report."""
        report = SentinelReport(
            query=debate_result.query,
            api_calls=debate_result.api_calls,
            total_tokens=debate_result.total_tokens_in + debate_result.total_tokens_out,
            latency_ms=debate_result.total_latency_ms,
            search_used=search_used,
        )

        # Collect models used
        models = set()
        for step in [debate_result.analysis, debate_result.critique, debate_result.synthesis]:
            if step:
                models.add(step.model)
        report.models_used = sorted(models)

        # Parse synthesis output
        raw = debate_result.final_output
        if not raw:
            report.overview = "Analysis failed — no output produced."
            report.confidence_score = 0
            return report

        # Try to extract structured sections
        sections = self._extract_sections(raw)

        report.overview = sections.get("overview", "").strip()
        report.assessment = sections.get("assessment", "").strip()
        report.critical_issues = sections.get("critical_issues", "").strip()
        report.tactical_map = sections.get("tactical_map", "").strip()
        report.final_synthesis = sections.get("final_synthesis", "").strip()

        # Parse confidence
        conf_text = sections.get("confidence", "").strip()
        report.confidence_score, report.confidence_justification = self._parse_confidence(conf_text)

        # If no sections were extracted, use the raw output as assessment
        if not any([report.overview, report.assessment, report.final_synthesis]):
            report.assessment = raw
            report.overview = self._extract_first_sentence(raw)
            report.confidence_score = 65

        return report

    def _extract_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        for name, pattern in self._SECTION_PATTERNS.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[name] = match.group(1).strip()
        return sections

    def _parse_confidence(self, text: str) -> tuple:
        if not text:
            return 70, "Default confidence"

        # Look for percentage
        match = re.search(r"(\d{1,3})\s*%", text)
        score = int(match.group(1)) if match else 70

        # Clamp to valid range
        score = max(0, min(100, score))

        # Extract justification (everything after the percentage)
        justification = text
        if match:
            justification = text[match.end():].strip().lstrip("—-–:").strip()
        if not justification:
            justification = "Based on debate quality"

        return score, justification

    def _extract_first_sentence(self, text: str) -> str:
        match = re.match(r"(.+?[.!?])\s", text)
        return match.group(1) if match else text[:200]
