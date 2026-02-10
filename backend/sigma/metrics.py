def calculate_hfi(graph_data: dict, rounds: int) -> float:
    """
    Hypothesis Fragility Index (HFI).
    Higher = More Fragile.
    """
    # Placeholder logic:
    # If intersection drops as rounds increase -> Fragile
    # For now, just a dummy calc based on graph density
    
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', graph_data.get('edges', []))
    
    if not nodes:
        return 1.0
        
    # Density = 2 * links / (nodes * (nodes-1))
    # High density of shared hypotheses = stable? Or fragile "single point of failure"?
    # Sentinel-Sigma logic says "Agreement depends on one dominant hypothesis -> Fragile"
    
    return len(links) / float(len(nodes)) * (1.0 + (rounds * 0.1))

def calculate_integrity_score(agreement_stable: bool, rounds_survived: int) -> float:
    if not agreement_stable:
        return 0.0
    return rounds_survived / 6.0  # Max rounds 6

def calculate_boundary_severity_impact(boundary_analysis: dict) -> float:
    """
    Calculate how much boundary violations impact epistemic integrity.
    
    Returns a severity score (0-100) that can be used for refusal decisions.
    """
    if not boundary_analysis:
        return 0.0
    
    cumulative_severity = boundary_analysis.get("cumulative_severity", 0.0)
    violation_count = boundary_analysis.get("violation_count", 0)
    
    # Severity is primarily driven by cumulative severity score
    # But amplified by violation count (multiplicative effect)
    severity_impact = cumulative_severity * (1.0 + (violation_count * 0.1))
    
    return min(100.0, severity_impact)

def extract_boundary_metrics(boundary_analysis: dict) -> dict:
    """
    Extract key boundary metrics for analysis.
    """
    if not boundary_analysis:
        return {
            "cumulative_severity": 0.0,
            "violation_count": 0,
            "max_severity": "minimal",
            "severity_impact": 0.0,
            "human_review_required": False,
        }
    
    return {
        "cumulative_severity": boundary_analysis.get("cumulative_severity", 0.0),
        "violation_count": boundary_analysis.get("violation_count", 0),
        "max_severity": boundary_analysis.get("max_severity", "minimal"),
        "severity_impact": calculate_boundary_severity_impact(boundary_analysis),
        "human_review_required": boundary_analysis.get("human_review_required", False),
    }
