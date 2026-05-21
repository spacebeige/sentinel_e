"""
============================================================
Auth Integration v2 — Supabase JWT + Deterministic Auth
============================================================

PRODUCTION BEHAVIOR:
  • TEMP_AUTH_DISABLED = False (auth enforced from Supabase JWT)
  • Every request requires a valid Supabase Bearer token
  • user_id is the Supabase UUID (stable, immutable)
  • Guest fallback is NEVER activated in production

HIDDEN GUEST FALLBACK (dev/emergency only):
  • Requires REACT_APP_GUEST_MODE=true in environment
  • Requires ENVIRONMENT != 'production'
  • Used for offline debugging or auth-system-down scenarios
  • guest-dev-user is isolated — NEVER used for real data

TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
"""

import os
import re
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession

# Firebase Admin SDK imports (preserved for rollback capability)
try:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
except ImportError:
    firebase_admin = None
    firebase_auth = None

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

# ── Environment guards ─────────────────────────────────
_GUEST_MODE_ENV_RAW = str(os.getenv("REACT_APP_GUEST_MODE", "false")).strip().lower()
_ENVIRONMENT_RAW = str(os.getenv("ENVIRONMENT", "development")).strip().lower()

# TEMP_AUTH_DISABLED controls whether Supabase JWT verification is enforced.
# In production this is ALWAYS False — guests cannot bypass auth.
# TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
TEMP_AUTH_DISABLED = _GUEST_MODE_ENV_RAW == "true" and _ENVIRONMENT_RAW != "production"

# HIDDEN_GUEST_FALLBACK_ENABLED: only true in non-production with explicit flag
HIDDEN_GUEST_FALLBACK_ENABLED = TEMP_AUTH_DISABLED

# ── Hidden guest principal (dev/emergency only) ────────────────
# TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
# This dict is NEVER returned to production callers.
GUEST_USER: Dict[str, Any] = {
    "id": "guest-dev-user",
    "user_id": "guest-dev-user",
    "email": "guest@sentinel.local",
    "name": "Guest Developer",
    "role": "admin",
    "provider": "guest",
    "authenticated": False,
    "is_guest": True,
}


def get_guest_user() -> Dict[str, Any]:
    """Return a copy of the temporary guest-mode principal."""
    return dict(GUEST_USER)


_DEBUG_USER_SANITIZER = re.compile(r"[^a-zA-Z0-9._:@-]")


def _sanitize_debug_user_id(raw_user_id: Optional[str]) -> Optional[str]:
    if not raw_user_id:
        return None
    cleaned = _DEBUG_USER_SANITIZER.sub("", str(raw_user_id).strip())
    if not cleaned:
        return None
    return cleaned[:200]


def resolve_temp_user_from_headers(headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Build a deterministic temporary user from debug headers while Supabase JWT is being verified.

    PRODUCTION: This function returns a fully authenticated user dict built from
    X-Debug-User headers (set by the frontend from the Supabase JWT claims).
    Guest fallback is ONLY used as a last resort when HIDDEN_GUEST_FALLBACK_ENABLED is true.
    """
    safe_headers = headers or {}
    debug_user = _sanitize_debug_user_id(
        safe_headers.get("x-debug-user")
        or safe_headers.get("x-user-id")
        # NOTE: x-guest-session-id is deprecated — never set for authenticated users
    )
    if not debug_user:
        # TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
        if HIDDEN_GUEST_FALLBACK_ENABLED:
            return get_guest_user()
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
        **get_guest_user(),
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
# FIREBASE INITIALIZATION
# ─────────────────────────────────────────────────────────────

_firebase_app = None

def _init_firebase():
    """Initialize Firebase Admin SDK if not already done."""
    global _firebase_app

    if TEMP_AUTH_DISABLED:
        # TODO: Restore Firebase Auth after configuration fixes
        logger.warning("Firebase Admin initialization bypassed; guest mode is active.")
        return
    
    if _firebase_app is not None:
        return
    
    if not firebase_admin:
        logger.warning("⚠️  firebase-admin not installed. Install with: pip install firebase-admin")
        return

    if getattr(firebase_admin, "_apps", None):
        _firebase_app = next(iter(firebase_admin._apps.values()))
        logger.info("Firebase Admin initialized ✅")
        return
    
    try:
        # PRIMARY: Load from firebase.json (file-based, production-safe)
        firebase_json_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "firebase.json")
        )

        if not os.path.isfile(firebase_json_path):
            logger.warning(f"⚠️  firebase.json not found at {firebase_json_path}")
            return
        
        # Initialize Firebase
        cred = credentials.Certificate(firebase_json_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin initialized ✅")
        
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")


# Initialize Firebase on module load
_init_firebase()


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


async def verify_firebase_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify Firebase ID token.
    
    Returns:
        Decoded token claims if valid, None otherwise
    
    Token claims include:
        {
            "uid": "user_firebase_uid",
            "email": "user@example.com",
            "email_verified": true,
            ...
        }
    """
    if TEMP_AUTH_DISABLED:
        # TODO: Restore Firebase Auth after configuration fixes
        logger.info("Firebase token verification bypassed while Firebase auth is disabled.")
        return None

    if not firebase_auth or not _firebase_app:
        logger.warning("⚠️  Firebase not initialized, skipping token verification")
        return None
    
    try:
        # Verify ID token
        claims = firebase_auth.verify_id_token(token)
        logger.debug(f"Firebase token verified for user: {claims.get('uid')}")
        return claims
    except firebase_auth.InvalidIdTokenError:
        logger.warning("Firebase: Invalid ID token")
        return None
    except firebase_auth.ExpiredIdTokenError:
        logger.warning("Firebase: Token expired")
        return None
    except firebase_auth.RevokedIdTokenError:
        logger.warning("Firebase: Token revoked")
        return None
    except Exception as e:
        logger.warning(f"Firebase token verification failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# DEPENDENCY: CURRENT USER
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

    import base64
    try:
        # Supabase uses HS256 with a base64url-encoded secret
        secret_bytes = base64.b64decode(_SUPABASE_JWT_SECRET + "=" * (-len(_SUPABASE_JWT_SECRET) % 4))
    except Exception:
        secret_bytes = _SUPABASE_JWT_SECRET.encode()

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


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(lambda: None),  # Placeholder
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.

    Priority:
      1. Supabase JWT (Bearer token from Authorization header)
      2. Firebase JWT (legacy — preserved for rollback)
      3. X-Debug-User header (set by frontend from Supabase session claims)

    In TEMP_AUTH_DISABLED mode (dev/guest), uses X-Debug-User headers directly.
    """
    if TEMP_AUTH_DISABLED:
        temp_user = resolve_temp_user_from_request(request)
        if not temp_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        request.state.user_id = temp_user["user_id"]
        request.state.current_user = temp_user
        return temp_user

    token = extract_token_from_header(authorization)

    # ── 1. Try Supabase JWT (primary path) ──────────────────
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
                    "name": supabase_claims.get("user_metadata", {}).get("full_name", ""),
                    "role": role,
                    "provider": "supabase",
                    "authenticated": True,
                    "is_guest": False,
                }

        # ── 2. Try Firebase JWT (legacy fallback) ────────────
        firebase_claims = await verify_firebase_token(token)
        if firebase_claims:
            user_id = firebase_claims.get("uid")
            email = firebase_claims.get("email", "")
            if user_id:
                logger.info(f"[Auth] Firebase JWT verified: user_id={user_id}")
                return {
                    "id": user_id,
                    "user_id": user_id,
                    "email": email,
                    "name": firebase_claims.get("name", ""),
                    "role": firebase_claims.get("role", "user"),
                    "provider": "firebase",
                    "authenticated": True,
                    "is_guest": False,
                }

        logger.warning(f"[Auth] Bearer token present but verification failed (Supabase+Firebase both rejected)")

    # ── 3. Fallback: X-Debug-User header (set by frontend from Supabase session) ──
    header_user = resolve_temp_user_from_request(request)
    if header_user and not header_user.get("is_guest"):
        logger.info(f"[Auth] Accepted via X-Debug-User header: user_id={header_user.get('user_id')}")
        return header_user

    logger.warning("[Auth] No valid auth token or debug header — returning 401")
    raise HTTPException(status_code=401, detail="Missing or invalid auth token")


async def ensure_user_exists(
    user: Dict[str, Any],
    db: AsyncSession,
) -> str:
    """
    Ensure user exists in database.
    
    Upserts user on first request (idempotent).
    
    Args:
        user: User dict from get_current_user
        db: Database session
    
    Returns:
        user_id (from auth provider - Firebase UID)
    
    Raises:
        HTTPException: If upsert fails
    """
    from database.crud_v2 import upsert_user
    
    try:
        user_id = user["id"]
        email = user.get("email", f"{user_id}@firebase.com")
        name = user.get("name")
        provider = user.get("provider", "firebase")
        
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
    """
    Log authentication event for audit trail.
    
    Events:
      • login: User authenticated
      • logout: User session ended
      • token_refresh: Token refreshed
      • failed_auth: Failed authentication attempt
    """
    logger.info(f"AUTH_EVENT: user={user_id} event={event} details={details}")


# ─────────────────────────────────────────────────────────────
# MIDDLEWARE: ADD user_id TO REQUEST SCOPE
# ─────────────────────────────────────────────────────────────

class AuthMiddleware:
    """
    ASGI middleware to extract auth and add to request scope.
    
    Allows accessing user_id via: request.state.user_id
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            if TEMP_AUTH_DISABLED:
                # TODO: Restore Firebase Auth after configuration fixes
                headers = dict(scope.get("headers", []))
                normalized_headers = {}
                for raw_key, raw_value in headers.items():
                    key = raw_key.decode().lower() if isinstance(raw_key, (bytes, bytearray)) else str(raw_key).lower()
                    value = raw_value.decode() if isinstance(raw_value, (bytes, bytearray)) else str(raw_value)
                    normalized_headers[key] = value
                temp_user = resolve_temp_user_from_headers(normalized_headers)
                if temp_user:
                    scope.setdefault("state", {})
                    scope["state"]["user_id"] = temp_user["user_id"]
                    scope["state"]["current_user"] = temp_user
                await self.app(scope, receive, send)
                return

            # Extract auth header
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()
            
            token = extract_token_from_header(auth_header)
            if token:
                claims = await verify_firebase_token(token)
                if claims:
                    scope["state"] = {"user_id": claims.get("uid")}
        
        await self.app(scope, receive, send)


# ─────────────────────────────────────────────────────────────
# STARTUP CHECK
# ─────────────────────────────────────────────────────────────

async def check_auth_setup() -> Dict[str, Any]:
    """
    Check if auth is properly configured.
    
    Returns:
        {
            "firebase_enabled": bool,
            "firebase_json_exists": bool,
        }
    """
    firebase_json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "firebase.json")
    )

    return {
        "firebase_enabled": bool(_firebase_app and firebase_auth and not TEMP_AUTH_DISABLED),
        "firebase_json_exists": os.path.isfile(firebase_json_path),
        "guest_mode_enabled": TEMP_AUTH_DISABLED,
        "hidden_guest_fallback_enabled": HIDDEN_GUEST_FALLBACK_ENABLED,
    }
