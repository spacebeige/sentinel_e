"""
============================================================
Auth Integration v2 — Supabase JWT (Production-only)
============================================================

PRODUCTION BEHAVIOR:
  • Every request requires a valid Supabase Bearer token
  • user_id is the Supabase UUID (stable, immutable)
  • No guest fallback — authentication is mandatory
  • Firebase auth fully removed (Supabase is the sole auth provider)

Dev-only header path:
  • resolve_temp_user_from_headers reads X-Debug-User headers
  • Only available in ENVIRONMENT != 'production'
  • Used by local dev when SUPABASE_JWT_SECRET is not set
"""

import os
import re
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession

# PyJWT for Supabase token verification
try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

logger = logging.getLogger("Auth")

# ── Supabase JWT verification ──────────────────────────────────
_SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
_RUNTIME_ADMIN_EMAILS = {
    email.strip().lower()
    for email in os.getenv("SENTINEL_RUNTIME_ADMIN_EMAILS", "oomkaragarkhed0710@gmail.com").split(",")
    if email and email.strip()
}

# ── Environment guards ─────────────────────────────────────────
_ENVIRONMENT_RAW = str(os.getenv("ENVIRONMENT", "development")).strip().lower()
_IS_PRODUCTION = _ENVIRONMENT_RAW == "production"


# ─────────────────────────────────────────────────────────────
# JWT TOKEN EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_token_from_header(authorization: Optional[str]) -> Optional[str]:
    """
    Extract JWT token from Authorization header.
    Format: Authorization: Bearer <token>
    """
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


# ─────────────────────────────────────────────────────────────
# SUPABASE JWT VERIFICATION
# ─────────────────────────────────────────────────────────────

async def verify_supabase_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Supabase JWT (access_token) using the project's JWT secret.
    Returns decoded claims if valid, None otherwise.
    """
    if not pyjwt:
        logger.warning("[Auth] PyJWT not installed — Supabase token verification unavailable")
        return None
    if not _SUPABASE_JWT_SECRET:
        logger.warning("[Auth] SUPABASE_JWT_SECRET not set — cannot verify Supabase JWT")
        return None

    # Supabase JWT secret is base64 encoded. We must decode it.
    import base64
    # Ensure correct padding for base64 decoding
    b64_secret = _SUPABASE_JWT_SECRET
    b64_secret += "=" * ((4 - len(b64_secret) % 4) % 4)
    try:
        secret_bytes = base64.b64decode(b64_secret)
    except Exception as e:
        logger.error(f"[Auth] Failed to decode base64 SUPABASE_JWT_SECRET: {e}")
        secret_bytes = _SUPABASE_JWT_SECRET.encode("utf-8") # Fallback just in case it wasn't base64

    try:
        claims = pyjwt.decode(
            token,
            secret_bytes,
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase anon JWTs may not have aud
        )
        logger.debug(f"[Auth] Supabase token verified for sub={claims.get('sub')}")
        return claims
    except pyjwt.ExpiredSignatureError:
        logger.warning("[Auth] Supabase JWT expired")
        return None
    except pyjwt.InvalidTokenError as e:
        logger.warning(f"[Auth] Supabase JWT invalid: {e}")
        return None
    except Exception as e:
        logger.warning(f"[Auth] Supabase JWT verification error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# DEPENDENCY: CURRENT USER
# ─────────────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(lambda: None),  # Placeholder
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.

    Priority:
      1. Supabase JWT (Bearer token from Authorization header)
      2. X-Debug-User header (dev-only, blocked in production)

    Returns:
      Authenticated user dict with id, user_id, email, name, role, provider.

    Raises:
      HTTPException 401 — if no valid auth token is present.
    """
    token = extract_token_from_header(authorization)

    # ── 1. Try Supabase JWT (primary path, all environments) ──
    if token:
        supabase_claims = await verify_supabase_token(token)
        if supabase_claims:
            user_id = supabase_claims.get("sub")
            email = supabase_claims.get("email", "")
            if user_id:
                role = "admin" if str(email).strip().lower() in _RUNTIME_ADMIN_EMAILS else supabase_claims.get("role", "authenticated")
                logger.info(f"[Auth] Supabase JWT verified: user_id={user_id}")
                return {
                    "id": user_id,
                    "user_id": user_id,
                    "email": email,
                    "name": (supabase_claims.get("user_metadata") or {}).get("full_name", ""),
                    "role": role,
                    "provider": "supabase",
                    "authenticated": True,
                    "is_guest": False,
                }

        logger.warning("[Auth] Bearer token present but Supabase verification failed")

    logger.warning("[Auth] No valid auth token — returning 401")
    raise HTTPException(status_code=401, detail="Missing or invalid auth token")

async def get_optional_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(lambda: None),
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to optionally get current authenticated user.
    Returns None if no valid token is provided.
    """
    try:
        return await get_current_user(request, authorization, db)
    except HTTPException:
        return None


# ─────────────────────────────────────────────────────────────
# USER UPSERT HELPER
# ─────────────────────────────────────────────────────────────

async def ensure_user_exists(
    user: Dict[str, Any],
    db: AsyncSession,
) -> str:
    """
    Ensure user exists in database (idempotent upsert on first request).

    Returns:
        user_id (Supabase UUID)

    Raises:
        HTTPException: If upsert fails
    """
    from database.crud_v2 import upsert_user

    try:
        user_id = user["id"]
        email = user.get("email", f"{user_id}@supabase.local")
        name = user.get("name")
        provider = user.get("provider", "supabase")

        await upsert_user(
            db,
            user_id=user_id,
            email=email,
            name=name,
            provider=provider,
        )
        logger.info(f"User ensured in DB: {user_id}")
        return user_id
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to ensure user: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize user")


# ─────────────────────────────────────────────────────────────
# AUDIT LOGGING
# ─────────────────────────────────────────────────────────────

async def log_auth_event(
    user_id: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
):
    """Log authentication event for audit trail."""
    logger.info(f"AUTH_EVENT: user={user_id} event={event} details={details}")


# ─────────────────────────────────────────────────────────────
# STARTUP CHECK
# ─────────────────────────────────────────────────────────────

async def check_auth_setup() -> Dict[str, Any]:
    """Check if auth is properly configured."""
    return {
        "auth_provider": "supabase",
        "supabase_jwt_secret_set": bool(_SUPABASE_JWT_SECRET),
        "environment": _ENVIRONMENT_RAW,
        "is_production": _IS_PRODUCTION,
        "dev_header_fallback_enabled": not _IS_PRODUCTION,
        "admin_emails_configured": len(_RUNTIME_ADMIN_EMAILS),
    }
