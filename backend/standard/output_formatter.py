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
    
    # === SYSTEM HEADER ===
    # output_lines.append("--- SENTINEL-E STANDARD MODE ---")
    if warning:
        output_lines.append(f"[SYSTEM WARNING: {warning}]")
    
    if boundary_warning:
        output_lines.append(f"[{boundary_warning}]")
        
    # === PIPELINE STATUS ===
    status_line = f"[Models: {model_count}] | [KNN Context: {'Active' if knn_active else 'Idle'}] | [Strategy: {neural_method}]"
    output_lines.append(status_line)
    output_lines.append("-" * len(status_line))
    
    # === CONTENT ===
    output_lines.append(text)
    
    return "\n\n".join(output_lines)
