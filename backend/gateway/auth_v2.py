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
    logger.info("JWT received: true")

    if not pyjwt:
        logger.error("verification failed: PyJWT not installed")
        raise HTTPException(status_code=500, detail="PyJWT not installed")

    # Log unverified token payload
    try:
        unverified_claims = pyjwt.decode(token, options={"verify_signature": False})
        logger.info(f"iss={unverified_claims.get('iss', 'None')}")
        logger.info(f"aud={unverified_claims.get('aud', 'None')}")
        logger.info(f"sub={unverified_claims.get('sub', 'None')}")
        logger.info(f"exp={unverified_claims.get('exp', 'None')}")
        logger.info(f"role={unverified_claims.get('role', 'None')}")
    except Exception as e:
        logger.error(f"verification failed: DecodeError during unverified parse: {e}")
        raise HTTPException(status_code=401, detail=f"DecodeError (Unverified): {e}")

    # Log secret status
    secret_exists = bool(_SUPABASE_JWT_SECRET)
    logger.info(f"JWT secret loaded: {secret_exists}")
    if secret_exists:
        logger.info(f"JWT secret length: {len(_SUPABASE_JWT_SECRET)}")
    if not _SUPABASE_JWT_SECRET:
        logger.error("verification failed: SUPABASE_JWT_SECRET not set in environment")
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET not set in environment")

    import base64
    # Use the plain string encoded as UTF-8, exactly as Supabase expects.
    # Do NOT try to base64 decode it, even if it looks like a base64 string.
    secret_bytes = _SUPABASE_JWT_SECRET.encode("utf-8")

    try:
        claims = pyjwt.decode(
            token,
            secret_bytes,
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase anon JWTs may not have aud
        )
        return claims
    except pyjwt.ExpiredSignatureError as e:
        logger.error(f"verification failed: ExpiredSignatureError - {e}")
        raise HTTPException(status_code=401, detail=f"ExpiredSignatureError: {e}")
    except pyjwt.InvalidSignatureError as e:
        logger.error(f"verification failed: InvalidSignatureError - {e}")
        raise HTTPException(status_code=401, detail=f"InvalidSignatureError: {e}")
    except pyjwt.InvalidIssuerError as e:
        logger.error(f"verification failed: InvalidIssuerError - {e}")
        raise HTTPException(status_code=401, detail=f"InvalidIssuerError: {e}")
    except pyjwt.InvalidAudienceError as e:
        logger.error(f"verification failed: InvalidAudienceError - {e}")
        raise HTTPException(status_code=401, detail=f"InvalidAudienceError: {e}")
    except pyjwt.DecodeError as e:
        logger.error(f"verification failed: DecodeError - {e}")
        raise HTTPException(status_code=401, detail=f"DecodeError: {e}")
    except Exception as e:
        logger.error(f"verification failed: Exception - {e}")
        raise HTTPException(status_code=401, detail=f"Exception: {e}")


# ─────────────────────────────────────────────────────────────
# DEV-ONLY HEADER FALLBACK
# ─────────────────────────────────────────────────────────────

_DEBUG_USER_SANITIZER = re.compile(r"[^a-zA-Z0-9._:@-]")


def _sanitize_debug_user_id(raw_user_id: Optional[str]) -> Optional[str]:
    if not raw_user_id:
        return None
    cleaned = _DEBUG_USER_SANITIZER.sub("", str(raw_user_id).strip())
    return cleaned[:200] if cleaned else None


def resolve_temp_user_from_headers(headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """
    Build a deterministic user from X-Debug-User headers.

    ONLY available in non-production environments.
    In production, this path is never reached — a 401 is returned instead.

    The frontend api.js sends X-Debug-User/X-Debug-Email from the Supabase
    session claims as a secondary identification path when SUPABASE_JWT_SECRET
    is not configured on the backend (local dev without secret).
    """
    safe_headers = headers or {}
    debug_user = _sanitize_debug_user_id(
        safe_headers.get("x-debug-user") or safe_headers.get("x-user-id")
    )
    if not debug_user:
        return None

    email = (
        safe_headers.get("x-debug-email")
        or safe_headers.get("x-user-email")
        or f"{debug_user}@sentinel.local"
    )
    name = (
        safe_headers.get("x-debug-name")
        or safe_headers.get("x-user-name")
        or email.split("@")[0]
        or "User"
    )
    provider = safe_headers.get("x-auth-provider") or "supabase"
    role = "admin" if str(email).strip().lower() in _RUNTIME_ADMIN_EMAILS else "authenticated"

    return {
        "id": debug_user,
        "user_id": debug_user,
        "email": email,
        "name": name,
        "role": role,
        "provider": provider,
        "authenticated": True,
        "is_guest": False,
    }


def resolve_temp_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    header_map = {}
    try:
        header_map = {k.lower(): v for k, v in request.headers.items()}
    except Exception:
        header_map = {}
    return resolve_temp_user_from_headers(header_map)


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

    # ── 2. Dev-only header fallback (non-production only) ─────
    if not _IS_PRODUCTION:
        temp_user = resolve_temp_user_from_request(request)
        if temp_user:
            logger.info(f"[Auth][DEV] Header fallback used for user_id={temp_user.get('user_id')}")
            return temp_user

    logger.warning("[Auth] No valid auth token — returning 401")
    raise HTTPException(status_code=401, detail="Missing or invalid auth token")


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
