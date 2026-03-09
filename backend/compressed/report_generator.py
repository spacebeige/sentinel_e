"""
Structured report generator for Sentinel-E compressed pipeline.
Parses role-based reasoning output into the standard Sentinel Analysis Report format.
Includes visualization data for debate, glass, and evidence views.
"""

import re
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from compressed.role_engine import RoleResult

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
    verification_consensus: Optional[bool] = None

    # Visualization data
    debate_data: Dict = field(default_factory=dict)
    glass_data: Dict = field(default_factory=dict)
    evidence_data: Dict = field(default_factory=dict)

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
        ]
        if self.verification_consensus is not None:
            lines.append(f"Verification: {'CONSENSUS' if self.verification_consensus else 'DISAGREEMENT'}")
        lines.append("━" * 60)
        return "\n".join(lines)

    def to_metadata(self) -> Dict:
        """Return omega_metadata-compatible dict."""
        return {
            "version": "sigma-rolebased-2.0",
            "mode": "compressed",
            "engine": "RoleBasedEngine",
            "confidence": self.confidence_score / 100.0,
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "models_used": self.models_used,
            "latency_ms": self.latency_ms,
            "search_used": self.search_used,
            "verification_consensus": self.verification_consensus,
            "stages": {
                "analysis": True,
                "critique": True,
                "synthesis": True,
                "verification": True,
            },
            "debate_data": self.debate_data,
            "glass_data": self.glass_data,
            "evidence_data": self.evidence_data,
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

    def generate(self, role_result: RoleResult, search_used: bool = False, search_bundle=None) -> SentinelReport:
        """Parse role-based reasoning output into structured report."""
        report = SentinelReport(
            query=role_result.query,
            api_calls=role_result.api_calls,
            total_tokens=role_result.total_tokens_in + role_result.total_tokens_out,
            latency_ms=role_result.total_latency_ms,
            search_used=search_used,
            verification_consensus=role_result.verification_consensus,
        )

        # Collect models used
        report.models_used = role_result.models_used

        # Parse synthesis output
        raw = role_result.final_output
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

        # Build visualization data
        report.debate_data = self._build_debate_data(role_result)
        report.glass_data = self._build_glass_data(role_result)
        report.evidence_data = self._build_evidence_data(search_bundle)

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

    def _build_debate_data(self, role_result: RoleResult) -> Dict:
        """Build debate visualization data from pipeline stages."""
        stages = []
        if role_result.analysis:
            stages.append({
                "stage": "analysis",
                "model": role_result.analysis.model,
                "content": role_result.analysis.content[:500],
                "latency_ms": role_result.analysis.latency_ms,
            })
        for c in role_result.critiques:
            stages.append({
                "stage": c.role,
                "model": c.model,
                "content": c.content[:500],
                "latency_ms": c.latency_ms,
            })
        if role_result.synthesis:
            stages.append({
                "stage": "synthesis",
                "model": role_result.synthesis.model,
                "content": role_result.synthesis.content[:500],
                "latency_ms": role_result.synthesis.latency_ms,
            })
        for v in role_result.verifications:
            stages.append({
                "stage": v.role,
                "model": v.model,
                "content": v.content[:300],
                "latency_ms": v.latency_ms,
            })
        return {"stages": stages, "total_stages": len(stages)}

    def _build_glass_data(self, role_result: RoleResult) -> Dict:
        """Build glass (transparency) visualization data — token usage per model."""
        model_usage = {}
        all_stages = (
            ([role_result.analysis] if role_result.analysis else []) +
            role_result.critiques +
            ([role_result.synthesis] if role_result.synthesis else []) +
            role_result.verifications
        )
        for s in all_stages:
            if s and s.model:
                if s.model not in model_usage:
                    model_usage[s.model] = {"tokens_in": 0, "tokens_out": 0, "calls": 0, "roles": []}
                model_usage[s.model]["tokens_in"] += s.tokens_in
                model_usage[s.model]["tokens_out"] += s.tokens_out
                model_usage[s.model]["calls"] += 1
                model_usage[s.model]["roles"].append(s.role)
        return {"model_usage": model_usage}

    def _build_evidence_data(self, search_bundle) -> Dict:
        """Build evidence visualization data from search results."""
        if not search_bundle or not hasattr(search_bundle, "results"):
            return {"sources": [], "count": 0}
        sources = []
        for r in search_bundle.results:
            sources.append({
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet[:200],
                "provider": getattr(r, "source", "unknown"),
            })
        return {"sources": sources, "count": len(sources)}
