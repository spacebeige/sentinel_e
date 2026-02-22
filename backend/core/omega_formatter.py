"""
Omega Structured Output Formatter — Sentinel-E v4.5

Generates strictly formatted markdown output per operational mode:
- STANDARD: Clean ChatGPT-grade response with structured sections
- DEBATE: Per-round per-model adversarial cards — no merging
- GLASS: Research console with progress bars, collapsible sections, descriptive metrics
- KILL: Descriptive diagnostic snapshot (inside Glass only)
- EVIDENCE: Source cards, reliability bars, single clean conclusion
- CODING: Standard + architecture/modules/tests

v4.5 Rules:
- No raw JSON anywhere
- No raw float scores — always descriptive
- No provider names (Groq, Tavily) in user-facing output
- Progress bars use [█████░░░░░] notation
- All numbers have human-readable interpretations
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("Omega-Formatter")


def _progress_bar(value: float, width: int = 10) -> str:
    """Generate a text progress bar: [█████░░░░░] 50%"""
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    empty = width - filled
    pct = int(value * 100)
    return f"[{'█' * filled}{'░' * empty}] {pct}%"


def _confidence_label(value: float) -> str:
    """Convert confidence float to descriptive label."""
    if value >= 0.85:
        return "High"
    elif value >= 0.65:
        return "Moderate"
    elif value >= 0.40:
        return "Fair"
    elif value >= 0.20:
        return "Low"
    return "Very Low"


def _risk_description(level: str, score: int) -> str:
    """Generate human-readable risk description."""
    level = (level or "LOW").upper()
    if level == "HIGH" or score >= 70:
        return "Elevated risk — conclusions require independent verification"
    elif level == "MEDIUM" or score >= 40:
        return "Moderate risk — some claims may need additional context"
    return "Low risk — within safe operating parameters"


def _expertise_description(score: float) -> str:
    """Convert expertise score to descriptive text."""
    if score >= 0.8:
        return "Expert (domain-specific terminology detected)"
    elif score >= 0.6:
        return "Advanced (familiar with technical concepts)"
    elif score >= 0.4:
        return "Intermediate (general technical awareness)"
    elif score >= 0.2:
        return "Developing (exploratory questions detected)"
    return "Beginner (foundational level)"


def _domain_display(domain: str) -> str:
    """Convert domain key to display name."""
    mapping = {
        "software_engineering": "Software Engineering",
        "machine_learning": "Machine Learning & AI",
        "cybersecurity": "Cybersecurity",
        "business_strategy": "Business Strategy",
        "scientific_research": "Scientific Research",
        "systems_thinking": "Systems Thinking",
    }
    return mapping.get(domain, domain.replace("_", " ").title() if domain else "General")


class OmegaOutputFormatter:
    """
    Enforces the Omega Cognitive Kernel output contract v4.5.
    Every response is structured markdown with mandatory sections.
    No raw JSON. No raw numbers without descriptions.
    """

    # ============================================================
    # STANDARD MODE FORMAT
    # ============================================================

    def format_standard(self, data: Dict[str, Any]) -> str:
        """
        Format STANDARD mode — clean ChatGPT-grade response.
        
        Sections:
        - Chat Name (first message only)
        - Executive Summary
        - Problem Decomposition
        - Identified Assumptions
        - Logical Risk Assessment
        - Solution
        - Session Notes
        - Confidence
        """
        sections = []

        if data.get("is_first_message") and data.get("chat_name"):
            sections.append(f"# {data['chat_name']}")
            sections.append("")

        # Executive Summary
        sections.append("## Summary")
        sections.append(data.get("executive_summary", "No summary available."))
        sections.append("")

        # Problem Decomposition
        decomposition = data.get("problem_decomposition", [])
        if isinstance(decomposition, list) and decomposition:
            sections.append("## Problem Breakdown")
            for i, item in enumerate(decomposition, 1):
                sections.append(f"{i}. **{item.get('component', 'Component')}** — {item.get('description', '')}")
            sections.append("")

        # Assumptions (only show if present)
        assumptions = data.get("assumptions", {})
        explicit = assumptions.get("explicit", [])
        implicit = assumptions.get("implicit", [])
        if explicit or implicit:
            sections.append("## Assumptions")
            if explicit:
                for a in explicit:
                    sections.append(f"- {a}")
            if implicit:
                sections.append("")
                sections.append("*Implicit assumptions detected:*")
                for a in implicit:
                    sections.append(f"- {a}")
            sections.append("")

        # Logical Risks (only show if present)
        gaps = data.get("logical_gaps", {})
        gap_list = gaps.get("gaps", [])
        if gap_list:
            sections.append("## Risk Assessment")
            for gap in gap_list:
                severity = gap.get("severity", "unknown").upper()
                sections.append(f"- **{severity}**: {gap.get('description', '')}")
            sections.append("")

        # Solution
        sections.append("## Analysis")
        solution = data.get("solution", "")
        sections.append(solution if solution else "Analysis pending.")
        sections.append("")

        # Boundary (only show if elevated)
        boundary = data.get("boundary", {})
        severity_score = boundary.get("severity_score", 0)
        if severity_score > 20:
            sections.append("## Boundary Check")
            risk_level = boundary.get("risk_level", "LOW")
            sections.append(f"Risk: **{risk_level}** {_progress_bar(severity_score / 100)}")
            explanation = boundary.get("explanation", "")
            if explanation:
                sections.append(f"  {explanation}")
            sections.append("")

        # Session + Confidence (compact)
        session = data.get("session", {})
        confidence = data.get("confidence", {})
        conf_val = confidence.get("value", 0.5)
        sections.append("---")
        expertise = session.get("user_expertise_score", 0.5)
        sections.append(f"*Confidence: {_confidence_label(conf_val)} ({conf_val:.0%}) · "
                        f"Expertise: {_expertise_description(expertise)}*")

        return "\n".join(sections)

    # ============================================================
    # DEBATE SUB-MODE FORMAT (v4.5 — Per-Round Per-Model Cards)
    # ============================================================

    def format_debate(self, data: Dict[str, Any]) -> str:
        """
        Format DEBATE sub-mode — True Adversarial Multi-Round Output.
        
        Per-round headers with per-model cards.
        No merging. Each model's position preserved independently.
        """
        sections = []
        debate = data.get("debate", {})
        rounds = debate.get("rounds", [])
        analysis = debate.get("analysis", {})

        # Title
        total_rounds = debate.get("total_rounds", len(rounds))
        models_used = debate.get("models_used", [])
        sections.append(f"# Adversarial Debate — {total_rounds} Rounds")
        sections.append(f"*{len(models_used)} models debating independently*")
        sections.append("")

        # Per-Round Output
        model_colors = {"model_a": "🔵", "model_b": "🟠", "model_c": "🟣"}
        
        for round_idx, round_models in enumerate(rounds):
            round_num = round_idx + 1
            sections.append(f"## Round {round_num}")
            sections.append("")
            
            for mo in round_models:
                color_emoji = model_colors.get(mo.get("model_id", ""), "⚪")
                label = mo.get("model_label", "Model")
                conf = mo.get("confidence", 0.5)
                
                sections.append(f"### {color_emoji} {label}")
                
                # Position
                position = mo.get("position", "")
                if position and position != "[Model unavailable this round]":
                    sections.append(f"**Position:** {position}")
                
                # Rebuttals (rounds 2+)
                rebuttals = mo.get("rebuttals", "")
                if rebuttals:
                    sections.append(f"**Rebuttals:** {rebuttals}")
                
                # Argument
                argument = mo.get("argument", "")
                if argument:
                    sections.append(f"\n{argument}")
                
                # Assumptions (round 1)
                assumptions = mo.get("assumptions", [])
                if assumptions:
                    sections.append("\n**Assumptions:**")
                    for a in assumptions:
                        sections.append(f"- {a}")
                
                # Risks (round 1)
                risks = mo.get("risks", [])
                if risks:
                    sections.append("\n**Risks:**")
                    for r in risks:
                        sections.append(f"- {r}")
                
                # Weaknesses found (rounds 2+)
                weaknesses = mo.get("weaknesses_found", "")
                if weaknesses:
                    sections.append(f"\n**Weaknesses Found:** {weaknesses}")
                
                # Position shift (rounds 2+)
                shift = mo.get("position_shift", "none")
                if shift and shift != "none":
                    sections.append(f"\n**Position Shift:** {shift}")
                
                # Confidence bar
                sections.append(f"\nConfidence: {_progress_bar(conf)}")
                sections.append("")
            
            sections.append("---")
            sections.append("")

        # Debate Analysis
        if analysis:
            sections.append("## Debate Analysis")
            sections.append("")
            
            # Conflict axes
            conflict = analysis.get("conflict_axes", [])
            if conflict:
                sections.append("**Points of Contention:**")
                for c in conflict:
                    sections.append(f"- {c}")
                sections.append("")
            
            # Disagreement strength
            disagree = analysis.get("disagreement_strength", 0.0)
            sections.append(f"**Disagreement:** {_progress_bar(disagree)} — "
                           f"{'Strong' if disagree > 0.7 else 'Moderate' if disagree > 0.4 else 'Mild'} divergence")
            
            # Logical stability
            stability = analysis.get("logical_stability", 0.5)
            sections.append(f"**Logical Stability:** {_progress_bar(stability)}")
            
            # Convergence
            conv_level = analysis.get("convergence_level", "none")
            conv_detail = analysis.get("convergence_detail", "")
            sections.append(f"\n**Convergence:** {conv_level.title()}")
            if conv_detail:
                sections.append(f"  {conv_detail}")
            
            # Position shifts
            shifts = analysis.get("position_shifts", [])
            if shifts:
                sections.append("\n**Position Shifts:**")
                for s in shifts:
                    sections.append(f"- {s}")
            
            # Strongest/weakest
            strongest = analysis.get("strongest_argument", "")
            weakest = analysis.get("weakest_argument", "")
            if strongest:
                sections.append(f"\n**Strongest Argument:** {strongest}")
            if weakest:
                sections.append(f"\n**Weakest Argument:** {weakest}")
            
            sections.append("")

        # Synthesis
        synthesis = data.get("synthesis", "")
        if synthesis:
            sections.append("## Synthesis")
            sections.append(synthesis)
            sections.append("")

        # Confidence + Fragility footer
        conf_evo = data.get("confidence_evolution", {})
        fragility = data.get("fragility", {})
        final_conf = conf_evo.get("final", 0.5)
        frag_score = fragility.get("score", 0.0)
        
        sections.append("---")
        sections.append(f"*Confidence: {_confidence_label(final_conf)} ({final_conf:.0%}) · "
                        f"Fragility: {_confidence_label(1.0 - frag_score)} ({frag_score:.0%})*")

        return "\n".join(sections)

    # Keep backward compat alias
    def format_experimental(self, data: Dict[str, Any]) -> str:
        return self.format_debate(data)

    # ============================================================
    # GLASS SUB-MODE FORMAT (v4.5 — Research Console)
    # ============================================================

    def format_glass(self, data: Dict[str, Any]) -> str:
        """
        Format GLASS sub-mode — Research-grade transparency console.
        
        Progress bars for all metrics. Descriptive labels.
        No raw JSON. No raw floats without context.
        Collapsible-ready section markers.
        """
        sections = []

        sections.append("# Glass — Transparency Console")
        sections.append("")

        # Session Context
        sections.append("## Session Context")
        session = data.get("session", {})
        goal = session.get("primary_goal", "Analysis")
        domain = session.get("inferred_domain", "General")
        expertise = session.get("user_expertise_score", 0.5)
        sections.append(f"**Goal:** {goal}")
        sections.append(f"**Domain:** {_domain_display(domain)}")
        sections.append(f"**User Expertise:** {_expertise_description(expertise)} {_progress_bar(expertise)}")
        sections.append("")

        # Boundary Signals
        sections.append("## Boundary Signals")
        boundary = data.get("boundary", {})
        risk_level = boundary.get("risk_level", "LOW")
        severity = boundary.get("severity_score", 0)
        explanation = boundary.get("explanation", "")
        
        sections.append(f"**Risk Level:** {risk_level}")
        sections.append(f"**Severity:** {_progress_bar(severity / 100)}")
        sections.append(f"**Assessment:** {_risk_description(risk_level, severity)}")
        if explanation and explanation != "Within safe operating parameters.":
            sections.append(f"**Detail:** {explanation}")
        sections.append("")

        # Behavioral Analytics
        sections.append("## Behavioral Analytics")
        behavioral = data.get("behavioral_risk", {})
        overall = behavioral.get("overall_risk", 0.0)
        risk_label = behavioral.get("risk_level", "LOW")
        
        sections.append(f"**Overall Behavioral Risk:** {risk_label} {_progress_bar(overall)}")
        sections.append("")
        
        # Individual dimensions with progress bars
        dimensions = [
            ("Self-Preservation", behavioral.get("self_preservation_score", 0.0)),
            ("Manipulation Risk", behavioral.get("manipulation_probability", 0.0)),
            ("Evasion Index", behavioral.get("evasion_index", 0.0)),
            ("Confidence Inflation", behavioral.get("confidence_inflation", 0.0)),
        ]
        for name, val in dimensions:
            status = "⚠️" if val > 0.5 else "✓" if val < 0.2 else "—"
            sections.append(f"  {status} **{name}:** {_progress_bar(val)}")
        
        beh_explanation = behavioral.get("explanation", "")
        if beh_explanation:
            sections.append(f"\n*{beh_explanation}*")
        sections.append("")

        # Severity Trend
        sections.append("## Severity Trend")
        trend = data.get("severity_trend", {})
        trend_val = trend.get("trend", "insufficient_data")
        trend_icons = {
            "increasing": "📈 Increasing — risk is escalating",
            "decreasing": "📉 Decreasing — risk is subsiding",
            "stable": "➡️ Stable — no significant change",
            "insufficient_data": "📊 Insufficient data — need more interactions",
        }
        sections.append(trend_icons.get(trend_val, f"📊 {trend_val}"))
        sections.append("")

        # Confidence Pipeline
        sections.append("## Confidence Pipeline")
        conf = data.get("confidence_evolution", {})
        stages = [
            ("Initial", conf.get("initial", 0.5)),
            ("Post-Analysis", conf.get("post_analysis", 0.5)),
            ("Post-Boundary", conf.get("post_boundary", 0.5)),
            ("Final", conf.get("final", 0.5)),
        ]
        for stage_name, stage_val in stages:
            sections.append(f"  **{stage_name}:** {_progress_bar(stage_val)} — {_confidence_label(stage_val)}")
        sections.append("")

        # Fragility
        sections.append("## Fragility Index")
        fragility = data.get("fragility", {})
        frag_score = fragility.get("score", 0.0)
        frag_exp = fragility.get("explanation", "Stable.")
        sections.append(f"**Score:** {_progress_bar(frag_score)}")
        sections.append(f"**Assessment:** {frag_exp}")
        sections.append("")

        # Synthesis
        sections.append("## Analysis Output")
        sections.append(data.get("synthesis", "Synthesis pending."))

        return "\n".join(sections)

    # ============================================================
    # KILL MODE FORMAT (v4.5 — Descriptive Diagnostic)
    # ============================================================

    def format_kill(self, data: Dict[str, Any]) -> str:
        """
        Format KILL mode — descriptive diagnostic snapshot.
        No new reasoning generated. Pure state readout with human-readable labels.
        """
        sections = []

        sections.append("# Kill Switch — Diagnostic Snapshot")
        sections.append("*No new reasoning generated. This is a frozen state readout.*")
        sections.append("")

        # Session State
        sections.append("## Session State")
        state = data.get("session_state", {})
        sections.append(f"**Chat:** {state.get('chat_name', 'Unknown')}")
        sections.append(f"**Goal:** {state.get('primary_goal', 'Unknown')}")
        sections.append(f"**Domain:** {_domain_display(state.get('inferred_domain', 'Unknown'))}")
        expertise = state.get("user_expertise_score", 0.0)
        sections.append(f"**User Expertise:** {_expertise_description(expertise)} {_progress_bar(expertise)}")
        sections.append("")

        # Boundary State
        sections.append("## Boundary State")
        boundary = data.get("boundary_snapshot", {})
        latest = boundary.get("latest_severity", 0)
        trend = boundary.get("trend", "insufficient_data")
        sections.append(f"**Latest Severity:** {_progress_bar(latest / 100)}")
        trend_labels = {
            "increasing": "📈 Increasing — risk trending up",
            "decreasing": "📉 Decreasing — risk trending down",
            "stable": "➡️ Stable",
            "insufficient_data": "📊 Not enough data yet",
        }
        sections.append(f"**Trend:** {trend_labels.get(trend, trend)}")
        sections.append("")

        # Disagreement
        sections.append("## Model Disagreement")
        disagree = data.get("disagreement_score", 0.0)
        sections.append(f"**Score:** {_progress_bar(disagree)}")
        if disagree > 0.6:
            sections.append("*Models showed significant disagreement — conclusions may be contested.*")
        elif disagree > 0.3:
            sections.append("*Moderate disagreement detected between analytical perspectives.*")
        else:
            sections.append("*Models largely agreed — consensus is strong.*")
        sections.append("")

        # Fragility
        sections.append("## Session Fragility")
        frag = data.get("fragility_index", 0.0)
        sections.append(f"**Index:** {_progress_bar(frag)}")
        if frag >= 0.7:
            sections.append("*High fragility — multiple destabilizing factors active. Exercise caution.*")
        elif frag >= 0.4:
            sections.append("*Moderate fragility — some risk factors present.*")
        else:
            sections.append("*Session is stable.*")
        sections.append("")

        # Error Patterns
        sections.append("## Detected Error Patterns")
        patterns = data.get("error_patterns", [])
        if patterns:
            for p in patterns:
                ptype = p.get("type", "unknown").replace("_", " ").title()
                desc = p.get("description", "")
                count = p.get("occurrences", 1)
                sections.append(f"- **{ptype}** (×{count}): {desc}")
        else:
            sections.append("No recurring error patterns detected.")
        sections.append("")

        # Behavioral Risk
        sections.append("## Behavioral Risk Profile")
        beh = data.get("behavioral_risk_profile", {})
        if beh:
            overall = beh.get("overall_risk", 0.0)
            sections.append(f"**Overall Behavioral Risk:** {_progress_bar(overall)}")
            dimensions = [
                ("Self-Preservation", beh.get("self_preservation_score", 0.0)),
                ("Manipulation Risk", beh.get("manipulation_probability", 0.0)),
                ("Evasion Index", beh.get("evasion_index", 0.0)),
                ("Confidence Inflation", beh.get("confidence_inflation", 0.0)),
            ]
            for name, val in dimensions:
                sections.append(f"  **{name}:** {_progress_bar(val)}")
        else:
            sections.append("No behavioral data available.")
        sections.append("")

        # Session Confidence
        sections.append("## Session Confidence")
        conf = data.get("session_confidence", 0.5)
        sections.append(f"**Confidence:** {_confidence_label(conf)} {_progress_bar(conf)}")
        explanation = data.get("confidence_explanation", "")
        if explanation:
            sections.append(f"*{explanation}*")

        return "\n".join(sections)

    # ============================================================
    # EVIDENCE SUB-MODE FORMAT (v4.5 — Source Cards + Clean Conclusion)
    # ============================================================

    def format_evidence(self, data: Dict[str, Any]) -> str:
        """
        Format EVIDENCE sub-mode — source cards with reliability bars,
        contradiction detection, and a single clean conclusion.
        """
        sections = []

        sections.append("# Evidence Analysis")
        sections.append("")

        # Session Context (compact)
        session = data.get("session", {})
        domain = session.get("inferred_domain", "General")
        sections.append(f"*Domain: {_domain_display(domain)} · "
                        f"Goal: {session.get('primary_goal', 'Analysis')}*")
        sections.append("")

        # Evidence Sources
        evidence = data.get("evidence", {})
        sources = evidence.get("sources", [])
        if sources:
            sections.append("## Sources")
            for i, src in enumerate(sources, 1):
                reliability = src.get("reliability_score", 0.5)
                title = src.get("title", "Unknown Source")
                url = src.get("url", "#")
                domain_name = src.get("domain", "unknown")
                snippet = src.get("content_snippet", "")[:200]
                
                sections.append(f"### Source {i}: {title}")
                sections.append(f"*{domain_name}* · Reliability: {_progress_bar(reliability)}")
                if snippet:
                    sections.append(f"> {snippet}")
                sections.append("")
        else:
            sections.append("## Sources")
            sections.append("*No external sources available. Evidence engine may be in offline mode.*")
            sections.append("")

        # Contradictions
        contradictions = evidence.get("contradictions", [])
        if contradictions:
            sections.append("## Contradictions")
            for i, c in enumerate(contradictions, 1):
                severity = c.get("severity", 0)
                sections.append(f"**Contradiction {i}** — Severity: {_progress_bar(severity)}")
                claim_a = c.get("claim_a", "")[:150]
                claim_b = c.get("claim_b", "")[:150]
                if claim_a:
                    sections.append(f"> Source A: {claim_a}")
                if claim_b:
                    sections.append(f"> Source B: {claim_b}")
                sections.append("")

        # Evidence Metrics
        sections.append("## Evidence Metrics")
        ev_conf = evidence.get("evidence_confidence", 0.5)
        agreement = evidence.get("source_agreement", 0.0)
        source_count = evidence.get("source_count", 0)
        contradiction_count = evidence.get("contradiction_count", 0)
        
        sections.append(f"**Evidence Confidence:** {_progress_bar(ev_conf)} — {_confidence_label(ev_conf)}")
        sections.append(f"**Source Agreement:** {_progress_bar(agreement)}")
        sections.append(f"**Sources Found:** {source_count}")
        if contradiction_count > 0:
            sections.append(f"**Contradictions:** {contradiction_count} ⚠️")
        sections.append("")

        # Data Lineage
        lineage = evidence.get("lineage", [])
        if lineage:
            sections.append("## Data Lineage")
            for step in lineage:
                sections.append(f"- **{step.get('step', '?')}**: {step.get('detail', '')}")
            sections.append("")

        # Boundary (only if elevated)
        boundary = data.get("boundary", {})
        if boundary.get("severity_score", 0) > 20:
            sections.append("## Boundary Check")
            sections.append(f"Risk: **{boundary.get('risk_level', 'LOW')}** {_progress_bar(boundary.get('severity_score', 0) / 100)}")
            sections.append("")

        # Conclusion
        sections.append("## Conclusion")
        sections.append(data.get("synthesis", "Conclusion pending evidence review."))
        sections.append("")

        # Footer
        conf_evo = data.get("confidence_evolution", {})
        final = conf_evo.get("final", 0.5)
        sections.append("---")
        sections.append(f"*Evidence Confidence: {_confidence_label(final)} ({final:.0%})*")

        return "\n".join(sections)

    # ============================================================
    # CODING TASK FORMAT (STANDARD MODE EXTENSION)
    # ============================================================

    def format_coding_task(self, data: Dict[str, Any]) -> str:
        """
        Extended format for coding tasks per Coding Task Enforcement protocol.
        Wraps standard format with additional code-specific sections.
        """
        base = self.format_standard(data)

        code_sections = []
        code_data = data.get("coding", {})

        if code_data:
            code_sections.append("")
            code_sections.append("---")
            code_sections.append("")

            code_sections.append("## Architecture")
            code_sections.append(code_data.get("architecture", "See solution above."))
            code_sections.append("")

            modules = code_data.get("modules", [])
            if modules:
                code_sections.append("## Modules")
                for m in modules:
                    code_sections.append(f"- **{m.get('name', '')}**: {m.get('description', '')}")
                code_sections.append("")

            code_sections.append("## Validation")
            code_sections.append(code_data.get("validation", "Standard input validation."))
            code_sections.append("")

            code_sections.append("## Error Handling")
            code_sections.append(code_data.get("error_handling", "Try-except with structured logging."))
            code_sections.append("")

            code_sections.append("## Scalability")
            code_sections.append(code_data.get("scalability", "Horizontal scaling ready."))
            code_sections.append("")

            tests = code_data.get("tests", [])
            if tests:
                code_sections.append("## Test Outline")
                for t in tests:
                    code_sections.append(f"- {t}")

        return base + "\n".join(code_sections)
    # ============================================================
    # STRESS TEST FORMAT (RESEARCH MODE — STRESS SUB-MODE)
    # ============================================================

    def format_stress(self, data: Dict[str, Any]) -> str:
        """
        Format stress test results.

        Shows:
        - Overall stability score
        - Per-vector results (counterexample, adversarial, logical inversion, boundary amplification)
        - Contradictions found
        - Revised confidence
        - Fragility impact
        """
        sections = []
        sections.append("# Stress Test Results")
        sections.append("")

        session = data.get("session", {})
        sections.append(f"**Domain:** {_domain_display(session.get('inferred_domain', 'General'))}")
        sections.append(f"**Objective:** {session.get('primary_goal', 'Stress-test answer stability')}")
        sections.append("")

        stress = data.get("stress", {})
        if not stress:
            sections.append("*No stress test was executed. StressEngine may not be available.*")
            sections.append("")
        else:
            # Overall Stability
            stability = stress.get("stability_after_stress", 0.5)
            contradictions = stress.get("contradictions_found", 0)
            revised_conf = stress.get("revised_confidence", 0.5)
            overall_stability = stress.get("overall_stability", stability)

            sections.append("## Overall Stability")
            sections.append(f"- **Stability Score:** {_progress_bar(overall_stability)}")
            if overall_stability >= 0.8:
                sections.append("  *Answer is highly robust under adversarial pressure.*")
            elif overall_stability >= 0.5:
                sections.append("  *Answer shows moderate resilience. Some vectors exposed weaknesses.*")
            else:
                sections.append("  *Answer is fragile. Multiple stress vectors found vulnerabilities.*")
            sections.append(f"- **Contradictions Found:** {contradictions}")
            sections.append(f"- **Revised Confidence:** {_confidence_label(revised_conf)} ({revised_conf:.0%})")
            sections.append("")

            # Per-Vector Results
            vector_results = stress.get("vector_results", {})
            if vector_results:
                sections.append("## Stress Vectors")
                sections.append("")

                vector_labels = {
                    "counterexample": "Counterexample Attack",
                    "adversarial": "Adversarial Challenge",
                    "logical_inversion": "Logical Inversion",
                    "boundary_amplification": "Boundary Amplification",
                }

                for vector_key, label in vector_labels.items():
                    vr = vector_results.get(vector_key, {})
                    if vr:
                        severity = vr.get("severity", 0.0)
                        stable = vr.get("stable", True)
                        finding = vr.get("finding", "No finding.")
                        status_icon = "✅" if stable else "⚠️"
                        sections.append(f"### {status_icon} {label}")
                        sections.append(f"- **Severity:** {_progress_bar(severity)}")
                        sections.append(f"- **Status:** {'Stable' if stable else 'Vulnerability Found'}")
                        sections.append(f"- **Finding:** {finding[:500]}")
                        sections.append("")

            # Breakdown Points
            breakdown = stress.get("breakdown_points", [])
            if breakdown:
                sections.append("## Breakdown Points")
                for i, bp in enumerate(breakdown, 1):
                    sections.append(f"{i}. {bp}")
                sections.append("")

        # Confidence Evolution
        conf_evo = data.get("confidence_evolution", {})
        stages = conf_evo.get("stages", [])
        if stages:
            sections.append("## Confidence Pipeline")
            for stage in stages:
                name = stage.get("stage", "unknown")
                val = stage.get("value", 0.5)
                note = stage.get("note", "")
                sections.append(f"- **{name}:** {_progress_bar(val)} — {note}")
            sections.append("")

        # Boundary
        boundary = data.get("boundary", {})
        if boundary:
            risk = boundary.get("risk_level", "LOW")
            sev = boundary.get("severity_score", 0)
            sections.append("## Boundary Assessment")
            sections.append(f"- **Risk Level:** {risk}")
            sections.append(f"- **Severity:** {_progress_bar(sev / 100)}")
            sections.append(f"- {boundary.get('explanation', '')}")
            sections.append("")

        # Fragility
        frag = data.get("fragility", {})
        if frag:
            frag_score = frag.get("score", 0.0)
            sections.append("## Fragility Impact")
            sections.append(f"- **Score:** {_progress_bar(frag_score)}")
            sections.append(f"- {frag.get('explanation', '')}")
            sections.append("")

        # Synthesis (the original answer)
        synthesis = data.get("synthesis", "")
        if synthesis:
            sections.append("---")
            sections.append("")
            sections.append("## Original Answer (Tested)")
            sections.append("")
            sections.append(synthesis[:2000])
            sections.append("")

        # Footer
        final_conf = stress.get("revised_confidence", 0.5) if stress else 0.5
        sections.append("---")
        sections.append(f"*Post-Stress Confidence: {_confidence_label(final_conf)} ({final_conf:.0%})*")

        return "\n".join(sections)