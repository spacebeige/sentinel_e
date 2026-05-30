"""
============================================================
API Response Utilities v2 — Safe, Deterministic Responses
============================================================

Ensures all API responses are:
  • Never null/undefined
  • Always have {success, data, error} structure
  • Type-safe (no unexpected structures)
  • Deterministic (same input → same output)
  • Never contain sensitive data

Usage:
    from utils.safe_responses import success, error, empty_structure
    
    # Success response
    return success({"chat_id": "...", "title": "...", "messages": []})
    
    # Error response
    return error("Invalid chat ID", 400)
    
    # Safe default structures
    return success(empty_structure("chat"))
"""

from typing import Any, Dict, Optional, List, Union
from fastapi.responses import JSONResponse
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# SAFE RESPONSE BUILDERS
# ─────────────────────────────────────────────────────────────

def success(
    data: Any,
    code: int = 200,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build success response.
    
    Args:
        data: Response payload (dict, list, etc)
        code: HTTP status code
        message: Optional success message
    
    Returns:
        {success: true, data: {...}, error: null}
    """
    # Never allow null data
    if data is None:
        data = {}
    
    response = {
        "success": True,
        "data": data,
        "error": None,
    }
    
    if message:
        response["message"] = message
    
    from fastapi.responses import JSONResponse
    return JSONResponse(content=response, status_code=code)


def error(
    message: str,
    code: int = 400,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build error response.
    
    Args:
        message: Human-readable error message
        code: HTTP status code
        details: Optional error details (not sensitive)
    
    Returns:
        {success: false, data: {}, error: {...}}
    """
    error_obj = {
        "message": message,
        "code": code,
    }
    
    if details:
        error_obj["details"] = details
    
    response = {
        "success": False,
        "data": {},
        "error": error_obj,
    }
    
    from fastapi.responses import JSONResponse
    return JSONResponse(content=response, status_code=code)


# ─────────────────────────────────────────────────────────────
# SAFE DEFAULT STRUCTURES (Never Null)
# ─────────────────────────────────────────────────────────────

def empty_user_structure() -> Dict[str, Any]:
    """Safe empty user structure."""
    return {
        "id": None,  # Will be set by API
        "email": "",
        "name": "",
        "provider": "clerk",
        "role": "user",
        "is_active": True,
        "created_at": None,
    }


def empty_session_structure() -> Dict[str, Any]:
    """Safe empty session structure."""
    return {
        "id": "",
        "user_id": "",
        "client": "web",
        "created_at": None,
        "last_active_at": None,
        "metadata": {},
    }


def empty_chat_structure() -> Dict[str, Any]:
    """Safe empty chat structure."""
    return {
        "id": "",
        "user_id": "",
        "title": "Untitled Chat",
        "mode": "conversational",
        "messages": [],
        "message_count": 0,
        "created_at": None,
        "updated_at": None,
        "is_archived": False,
        "metadata": {},
    }


def empty_message_structure() -> Dict[str, Any]:
    """Safe empty message structure."""
    return {
        "id": "",
        "chat_id": "",
        "user_id": "",
        "role": "user",  # user | assistant | system
        "content": "",
        "reasoning_json": None,
        "metadata": {},
        "image_url": None,
        "created_at": None,
    }


def empty_history_structure() -> Dict[str, Any]:
    """Safe empty history structure (for chat refresh)."""
    return {
        "chats": [],
        "chat_count": 0,
        "total_messages": 0,
    }


def empty_memory_structure() -> Dict[str, Any]:
    """Safe empty memory structure."""
    return {
        "entries": [],
        "entry_count": 0,
    }


def empty_structure(structure_type: str) -> Dict[str, Any]:
    """
    Get empty structure by type.
    
    Args:
        structure_type: user | session | chat | message | history | memory
    
    Returns:
        Safe empty structure with no null values
    """
    structures = {
        "user": empty_user_structure,
        "session": empty_session_structure,
        "chat": empty_chat_structure,
        "message": empty_message_structure,
        "history": empty_history_structure,
        "memory": empty_memory_structure,
    }
    
    builder = structures.get(structure_type, empty_chat_structure)
    return builder()


# ─────────────────────────────────────────────────────────────
# RESPONSE BUILDERS FROM MODELS
# ─────────────────────────────────────────────────────────────

def user_to_dict(user) -> Dict[str, Any]:
    """Convert User model to dict (safe for frontend)."""
    if user is None:
        return empty_user_structure()
    
    return {
        "id": str(user.id),
        "email": user.email or "",
        "name": user.name or "",
        "provider": user.provider or "clerk",
        "role": user.role or "user",
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def session_to_dict(session) -> Dict[str, Any]:
    """Convert Session model to dict."""
    if session is None:
        return empty_session_structure()
    
    return {
        "id": str(session.id),
        "user_id": str(session.user_id),
        "client": session.client or "web",
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "last_active_at": session.last_active_at.isoformat() if session.last_active_at else None,
        "metadata": getattr(session, "metadata_json", None) or {},
    }


def chat_to_dict(chat, messages: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Convert Chat model to dict.
    
    Args:
        chat: Chat model
        messages: Optional list of message dicts
    
    Returns:
        Safe chat structure
    """
    if chat is None:
        return empty_chat_structure()
    
    return {
        "id": str(chat.id),
        "user_id": str(chat.user_id),
        "title": chat.title or "Untitled Chat",
        "mode": chat.mode or "conversational",
        "messages": messages or [],
        "message_count": len(messages) if messages else 0,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
        "is_archived": chat.is_archived,
        "metadata": {
            "machine": chat.machine_metadata or {},
            "user": chat.user_metadata or {},
        },
    }


def message_to_dict(message) -> Dict[str, Any]:
    """Convert Message model to dict."""
    if message is None:
        return empty_message_structure()
    
    return {
        "id": str(message.id),
        "chat_id": str(message.chat_id),
        "user_id": str(message.user_id),
        "role": message.role or "user",
        "content": message.content or "",
        "reasoning_json": message.reasoning_json,
        "metadata": getattr(message, "metadata_json", None) or {},
        "image_url": message.image_url,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def memory_to_dict(memory) -> Dict[str, Any]:
    """Convert Memory model to dict."""
    if memory is None:
        return {
            "id": "",
            "user_id": "",
            "key": "",
            "value": {},
            "weight": 0.0,
            "confidence": 0,
            "updated_at": None,
        }
    
    return {
        "id": str(memory.id),
        "user_id": str(memory.user_id),
        "key": memory.key,
        "value": memory.value,
        "weight": memory.weight,
        "confidence": memory.confidence,
        "tag": memory.tag,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
    }


# ─────────────────────────────────────────────────────────────
# COMMON RESPONSE PATTERNS
# ─────────────────────────────────────────────────────────────

def chat_history_response(chats: List[Any], messages_by_chat: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Build chat history response (for /api/history endpoint).
    
    Args:
        chats: List of Chat models
        messages_by_chat: Dict mapping chat_id → list of message dicts
    
    Returns:
        {chats: [...], total_messages: int}
    """
    chat_dicts = []
    total_messages = 0
    
    for chat in chats:
        messages = messages_by_chat.get(str(chat.id), [])
        total_messages += len(messages)
        chat_dicts.append(chat_to_dict(chat, messages))
    
    return {
        "chats": chat_dicts,
        "chat_count": len(chat_dicts),
        "total_messages": total_messages,
    }


def context_window_response(messages: List[Any], memory_entries: List[Any]) -> Dict[str, Any]:
    """
    Build context window response (for /api/context endpoint).
    
    Args:
        messages: List of message dicts (chronological, limited by tokens)
        memory_entries: List of memory dicts (by weight)
    
    Returns:
        {messages: [...], memory: [...], token_estimate: int}
    """
    # Rough token estimate: 1 token ≈ 4 chars
    total_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
    total_tokens += sum(len(str(m.get("value", ""))) // 4 for m in memory_entries)
    
    return {
        "messages": messages,
        "memory": memory_entries,
        "message_count": len(messages),
        "memory_count": len(memory_entries),
        "token_estimate": total_tokens,
    }


# ─────────────────────────────────────────────────────────────
# ERROR HELPERS
# ─────────────────────────────────────────────────────────────

def unauthorized_response() -> Dict[str, Any]:
    """Unauthorized (401) response."""
    return error("Unauthorized: missing or invalid auth token", 401)


def forbidden_response(reason: str = "Access denied") -> Dict[str, Any]:
    """Forbidden (403) response."""
    return error(f"Forbidden: {reason}", 403)


def not_found_response(resource: str = "Resource") -> Dict[str, Any]:
    """Not found (404) response."""
    return error(f"{resource} not found", 404)


def bad_request_response(message: str) -> Dict[str, Any]:
    """Bad request (400) response."""
    return error(message, 400)


def server_error_response(message: str = "Internal server error") -> Dict[str, Any]:
    """Server error (500) response."""
    return error(message, 500)
