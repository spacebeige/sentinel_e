"""
Output Sanitization Utility — Sentinel-E v5.0

Removes internal reasoning tags, debug markers, and hidden artifacts from LLM outputs.
Ensures clean, user-facing responses without internal thinking process exposure.

Sanitizes:
  - <think>...</think> blocks
  - <analysis>...</analysis> blocks
  - <reasoning>...</reasoning> blocks
  - Other internal XML-like tags
  - Debug artifacts

Priority:
  1. Remove all hidden reasoning tags
  2. Preserve semantic meaning and structure
  3. Clean up excessive whitespace
  4. Maintain original formatting where appropriate
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("OutputSanitizer")


def sanitize_output(text: Optional[str]) -> str:
    """
    Clean LLM output by removing internal reasoning tags and debug artifacts.
    
    Args:
        text: Raw LLM output (may contain <think>, <analysis>, etc.)
    
    Returns:
        Cleaned text suitable for user consumption
    """
    if not text or not isinstance(text, str):
        return ""
    
    original_len = len(text)
    
    # 1. Remove <think>...</think> blocks (most important)
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # 2. Remove <analysis>...</analysis> blocks
    text = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # 3. Remove <reasoning>...</reasoning> blocks
    text = re.sub(
        r"<reasoning>.*?</reasoning>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # 4. Remove other common debug/internal tags
    debug_tags = [
        r"<internal>.*?</internal>",
        r"<debug>.*?</debug>",
        r"<hidden>.*?</hidden>",
        r"<scratch>.*?</scratch>",
        r"<scratchpad>.*?</scratchpad>",
        r"<planning>.*?</planning>",
        r"<strategy>.*?</strategy>",
        r"<meta>.*?</meta>",
    ]
    
    for tag_pattern in debug_tags:
        text = re.sub(
            tag_pattern,
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
    
    # 5. Clean up excessive whitespace
    # Remove leading/trailing whitespace from each line
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    
    # Remove empty lines but preserve intentional spacing
    # Keep max 2 consecutive newlines
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 6. Remove any remaining orphaned opening/closing tags
    text = re.sub(r"</?[a-z]+>", "", text, flags=re.IGNORECASE)
    
    # Final trim
    text = text.strip()
    
    # Log if significant content was removed
    removed_chars = original_len - len(text)
    if removed_chars > 100:
        logger.debug(
            f"Sanitized output: removed {removed_chars} characters "
            f"({removed_chars/original_len*100:.1f}% of original)"
        )
    
    return text


def sanitize_json_response(response_dict: dict) -> dict:
    """
    Sanitize all string fields in a response dictionary.
    
    Args:
        response_dict: API response as dictionary
    
    Returns:
        Response with all string fields sanitized
    """
    if not isinstance(response_dict, dict):
        return response_dict
    
    sanitized = {}
    
    for key, value in response_dict.items():
        if isinstance(value, str):
            # Sanitize string fields
            sanitized[key] = sanitize_output(value)
        elif isinstance(value, dict):
            # Recursively sanitize nested dicts
            sanitized[key] = sanitize_json_response(value)
        elif isinstance(value, list):
            # Sanitize items in lists
            sanitized[key] = [
                sanitize_output(item) if isinstance(item, str)
                else sanitize_json_response(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            # Keep non-string values as-is
            sanitized[key] = value
    
    return sanitized


def has_internal_tags(text: Optional[str]) -> bool:
    """
    Check if text contains internal reasoning tags.
    
    Returns:
        True if any prohibited tags detected
    """
    if not text or not isinstance(text, str):
        return False
    
    prohibited_patterns = [
        r"<think>",
        r"<analysis>",
        r"<reasoning>",
        r"<internal>",
        r"<debug>",
        r"<hidden>",
        r"<scratch>",
        r"<scratchpad>",
        r"<planning>",
        r"<strategy>",
        r"<meta>",
    ]
    
    for pattern in prohibited_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False
