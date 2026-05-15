"""
============================================================
Auth Integration v2 — Firebase + Deterministic Auth
============================================================

Features:
  • Extract user_id from Firebase ID token
  • Automatic user upsert on first request
  • Session tracking per request
  • Role-based access control (RBAC)
  • Audit logging

Principles:
  • Every request requires valid auth
  • user_id is immutable Firebase UID
  • email is unique per user
  • No duplicate users

Configuration:
    • firebase.json file in backend/ (service account JSON)
"""

import os
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession

# Firebase Admin SDK imports
try:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
except ImportError:
    firebase_admin = None
    firebase_auth = None

logger = logging.getLogger("Auth")

TEMP_AUTH_DISABLED = True

# TODO: Restore Firebase Auth after configuration fixes
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
        logger.info("Guest mode active; Firebase token verification bypassed.")
        return get_guest_user()

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

async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(lambda: None),  # Placeholder
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.
    
    Extracts user_id from Firebase ID token.
    Ensures user exists in database.
    
    Usage:
        @app.get("/api/endpoint")
        async def endpoint(user: Dict = Depends(get_current_user)):
            user_id = user["id"]
            ...
    
    Returns:
        {
            "id": "firebase_uid",
            "email": "user@example.com",
            "name": "User Name",
            "provider": "firebase",
        }
    
    Raises:
        HTTPException(401): If auth token invalid or missing
    """
    if TEMP_AUTH_DISABLED:
        # TODO: Restore Firebase Auth after configuration fixes
        request.state.user_id = GUEST_USER["user_id"]
        request.state.current_user = get_guest_user()
        return get_guest_user()

    # Extract token
    token = extract_token_from_header(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    
    # Verify Firebase token
    decoded = await verify_firebase_token(token)
    if not decoded:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    
    # Extract user_id and email from claims
    try:
        user_id = decoded["uid"]
    except KeyError:
        raise HTTPException(status_code=401, detail="Token missing user ID")

    email = decoded.get("email")
    name = decoded.get("name", "")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user ID")
    
    # Strict user_id validation
    if not isinstance(user_id, str):
        raise HTTPException(status_code=401, detail="Invalid user")

    # Log for runtime diagnostics
    try:
        logger.info(f"FIREBASE USER_ID: {user_id}")
        print("FIREBASE USER_ID:", user_id)
    except Exception:
        pass

    # Return user info
    return {
        "id": user_id,
        "user_id": user_id,
        "email": email or "",
        "name": name,
        "role": decoded.get("role", "user"),
        "provider": "firebase",
    }


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
                scope.setdefault("state", {})
                scope["state"]["user_id"] = GUEST_USER["user_id"]
                scope["state"]["current_user"] = get_guest_user()
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
    }
