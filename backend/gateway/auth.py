"""
============================================================
JWT Authentication & Session Management
============================================================
Implements:
- JWT access/refresh token issuance
- Token verification middleware
- Anonymous session bootstrapping (auto-issued on first visit)
- User identity isolation for chats
"""

import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.gateway.config import get_settings

logger = logging.getLogger("Auth")
security = HTTPBearer(auto_error=False)


def create_access_token(
    user_id: str,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Issue a signed JWT access token."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        "jti": str(uuid.uuid4()),
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Issue a signed JWT refresh token."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
        "jti": str(uuid.uuid4()),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT token. Raises on failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    Extract user identity from JWT.
    
    In development mode with no token, auto-bootstrap an anonymous session.
    In production, require valid JWT.
    """
    settings = get_settings()

    if credentials and credentials.credentials:
        payload = decode_token(credentials.credentials)
        return {
            "user_id": payload["sub"],
            "token_type": payload.get("type", "access"),
            "authenticated": True,
        }

    # Anonymous bootstrap (development/staging only)
    if not settings.is_production:
        anon_id = f"anon-{uuid.uuid4().hex[:12]}"
        logger.debug(f"Anonymous session bootstrapped: {anon_id}")
        return {
            "user_id": anon_id,
            "token_type": "anonymous",
            "authenticated": False,
        }

    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Like get_current_user but returns None instead of raising."""
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
