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
  • FIREBASE_SERVICE_ACCOUNT_JSON: From Firebase Console (JSON string or path)
"""

import os
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
import json

# Firebase Admin SDK imports
try:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
except ImportError:
    firebase_admin = None
    firebase_auth = None

logger = logging.getLogger("Auth")

# ─────────────────────────────────────────────────────────────
# FIREBASE INITIALIZATION
# ─────────────────────────────────────────────────────────────

_firebase_app = None


def _build_service_account_from_env() -> Optional[Dict[str, Any]]:
    """Build a Firebase service-account dict from individual env vars."""
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    private_key_id = os.getenv("FIREBASE_PRIVATE_KEY_ID")
    private_key = os.getenv("FIREBASE_PRIVATE_KEY")
    client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
    client_id = os.getenv("FIREBASE_CLIENT_ID")

    if not all([project_id, private_key_id, private_key, client_email, client_id]):
        return None

    return {
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": private_key_id,
        "private_key": private_key,
        "client_email": client_email,
        "client_id": client_id,
        "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.getenv(
            "FIREBASE_AUTH_PROVIDER_X509_CERT_URL",
            "https://www.googleapis.com/oauth2/v1/certs",
        ),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL", ""),
    }

def _init_firebase():
    """Initialize Firebase Admin SDK if not already done."""
    global _firebase_app
    
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
        # Try to get service account from environment
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        
        service_account = None
        if service_account_json:
            # Parse JSON (could be a JSON string or path)
            try:
                service_account = json.loads(service_account_json)
            except json.JSONDecodeError:
                # Try treating it as a file path
                if os.path.isfile(service_account_json):
                    with open(service_account_json) as f:
                        service_account = json.load(f)
                else:
                    logger.warning("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON or file path; falling back to FIREBASE_* env vars")

        if service_account is None:
            service_account = _build_service_account_from_env()

        if not service_account:
            logger.warning("⚠️  FIREBASE_SERVICE_ACCOUNT_JSON not set and FIREBASE_* env vars incomplete")
            return
        
        # Initialize Firebase
        cred = credentials.Certificate(service_account)
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
            "clerk_enabled": bool,
            "secret_key_set": bool,
            "publishable_key_set": bool,
        }
    """
    return {
        "firebase_enabled": bool(_firebase_app and firebase_auth),
        "service_account_set": bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or _build_service_account_from_env()),
    }
