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
    Generates a deterministic chat name using lightweight extraction.
    """
    if not text:
        return "New Chat"
        
    text_clean = re.sub(r'[^\w\s\-]', '', text).strip()
    
    starters = [
        "how do i ", "how to ", "can you ", "could you ", "what is ", 
        "what are ", "explain ", "tell me about ", "help me with ", 
        "i need ", "write a ", "create a ", "build a ", "show me "
    ]
    
    lower_text = text_clean.lower()
    for s in starters:
        if lower_text.startswith(s):
            text_clean = text_clean[len(s):].strip()
            break
            
    words = text_clean.split()
    if not words:
        return "New Chat"
        
    title = " ".join(words[:4]).title()
    
    if len(title) > 40:
        title = title[:37] + "..."
        
    return title
