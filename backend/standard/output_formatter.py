def format_output(result: dict) -> str:
    """
    Formats the final output for the user.
    Includes technical metadata about the pipeline stages (KNN, Neural, etc.)
    """
    text = result.get("text", "")
    warning = result.get("warning")
    boundary_warning = result.get("boundary_warning")
    confidence = result.get("confidence", 0.0)
    
    # Metadata
    knn_active = result.get("knn_active", False)
    neural_method = result.get("method", "centroid")
    model_count = result.get("model_count", 0)

    output_lines = []
    
    # === CLEAN OUTPUT MODE ===
    # For Conversational/User-Facing Mode, we suppress all internal telemetry.
    if boundary_warning:
        output_lines.append(f"Note: {boundary_warning}")
        output_lines.append("")

    # Content Only
    output_lines.append(text)
    
    return "\n\n".join(output_lines)
