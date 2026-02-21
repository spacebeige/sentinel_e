import re

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "from", "with", "by", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def generate_chat_name(text: str, mode: str) -> str:
    """
    Generates a deterministic chat name based on the first meaningful 6 words.
    """
    # Clean valid characters only (alphanumeric + spaces)
    text_clean = re.sub(r'[^\w\s]', '', text).strip()
    
    # Tokenize
    words = text_clean.split()
    
    # Filter stop words and take first 6 meaningful words
    meaningful = [w for w in words if w.lower() not in STOP_WORDS]
    
    # Fallback if no meaningful words
    if not meaningful:
        meaningful = words
        
    # Take first 6
    selected = meaningful[:6]
    
    # Construct name
    name = " ".join(selected).title()
    
    # Truncate if too long (backup safety)
    if len(name) > 60:
        name = name[:57] + "..."
        
    if not name:
        name = "New Analysis"
        
    return name
    
    # Filter stop words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Take top 3 unique keywords, preserving order
    seen = set()
    top_keywords = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            top_keywords.append(w.title())
        if len(top_keywords) >= 3:
            break
            
    if not top_keywords:
        base_name = "New Chat"
    else:
        base_name = " ".join(top_keywords)
    
    # Mode suffix mapping
    mode_map = {
        "conversational": "", # Default, no suffix
        "experimental": "— Experimental",
        "forensic": "— Forensic", 
        "shadow": "— Shadow"
    }
    
    suffix = mode_map.get(mode, "")
    
    return f"{base_name} {suffix}".strip()
