"""
Server-side runtime administrator authorization.

This module is intentionally small and independent so admin dashboards,
orchestration telemetry, and mission-control streams share one policy.
Frontend route guards are only a convenience; this is the enforcement layer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, Query, Request

from gateway.auth_v2 import (
    extract_token_from_header,
    get_current_user,
    verify_supabase_token,
)

logger = logging.getLogger("AdminAccess")

DEFAULT_RUNTIME_ADMIN_EMAIL = "oomkaragarkhed0710@gmail.com"


def _configured_admin_emails() -> set[str]:
    raw = os.getenv("SENTINEL_RUNTIME_ADMIN_EMAILS", DEFAULT_RUNTIME_ADMIN_EMAIL)
    return {
        email.strip().lower()
        for email in raw.split(",")
        if email and email.strip()
    }


def is_runtime_admin_email(email: Optional[str]) -> bool:
    return bool(email and email.strip().lower() in _configured_admin_emails())


def enrich_runtime_admin_role(user: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not user:
        return user
    enriched = dict(user)
    if is_runtime_admin_email(enriched.get("email")):
        enriched["role"] = "admin"
        enriched["runtime_admin"] = True
    else:
        enriched["runtime_admin"] = False
    return enriched


async def require_runtime_admin(
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    enriched = enrich_runtime_admin_role(user)
    if not enriched or not is_runtime_admin_email(enriched.get("email")):
        logger.warning(
            "Blocked runtime admin access for user_id=%s email=%s",
            (enriched or {}).get("user_id"),
            (enriched or {}).get("email"),
        )
        raise HTTPException(status_code=403, detail="Runtime administrator access required")
    return enriched


async def require_runtime_admin_for_stream(
    request: Request,
    authorization: Optional[str] = Header(None),
    access_token: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    EventSource cannot set Authorization headers. For SSE only, accept the
    short-lived Supabase access token as a query parameter and verify it
    server-side before exposing orchestration events.
    """
    token = access_token or extract_token_from_header(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    claims = await verify_supabase_token(token)
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    user = enrich_runtime_admin_role({
        "id": claims.get("sub"),
        "user_id": claims.get("sub"),
        "email": claims.get("email", ""),
        "name": (claims.get("user_metadata") or {}).get("full_name", ""),
        "role": claims.get("role", "authenticated"),
        "provider": "supabase",
        "authenticated": True,
        "is_guest": False,
    })
    request.state.current_user = user
    request.state.user_id = user.get("user_id")

    if not is_runtime_admin_email(user.get("email")):
        logger.warning(
            "Blocked runtime stream access for user_id=%s email=%s",
            user.get("user_id"),
            user.get("email"),
        )
        raise HTTPException(status_code=403, detail="Runtime administrator access required")
    return user
