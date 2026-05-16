# """
# ============================================================
# Authentication & Session Management
# ============================================================
# Primary auth path:
# - Clerk JWT verification (Bearer token)
# - Neon-backed user profile upsert into existing users table

# Legacy compatibility:
# - JWT helpers remain for older tests / fallback code paths
# """

# import functools
# import logging
# import uuid
# from datetime import datetime, timedelta, timezone
# from typing import Any, Dict, List, Optional

# import jwt
# from fastapi import Depends, HTTPException, Request
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# from database.connection import AsyncSessionLocal
# from database.crud import get_user_by_user_id, upsert_authenticated_user
# from gateway.config import get_settings

# logger = logging.getLogger("Auth")
# security = HTTPBearer(auto_error=False)


# @functools.lru_cache(maxsize=4)
# def _jwks_client_for_url(jwks_url: str) -> jwt.PyJWKClient:
#     return jwt.PyJWKClient(jwks_url)


# def _clerk_requested() -> bool:
#     settings = get_settings()
#     return bool(settings.clerk_jwks_url or settings.CLERK_JWT_ISSUER)


# def _pick_identity_value(*values: Optional[str]) -> Optional[str]:
#     for value in values:
#         if isinstance(value, str) and value.strip():
#             return value.strip()
#     return None


# def _extract_email_from_payload(payload: Dict[str, Any]) -> Optional[str]:
#     direct_email = _pick_identity_value(
#         payload.get("email"),
#         payload.get("email_address"),
#         payload.get("primary_email_address"),
#         payload.get("https://clerk.dev/email"),
#     )
#     if direct_email:
#         return direct_email

#     email_addresses = payload.get("email_addresses")
#     if isinstance(email_addresses, list):
#         for item in email_addresses:
#             if isinstance(item, dict):
#                 candidate = _pick_identity_value(item.get("email_address"), item.get("email"))
#                 if candidate:
#                     return candidate
#             elif isinstance(item, str) and item.strip():
#                 return item.strip()

#     return None


# def _extract_name_from_payload(payload: Dict[str, Any], email: Optional[str]) -> Optional[str]:
#     first_name = _pick_identity_value(payload.get("first_name"), payload.get("given_name"))
#     last_name = _pick_identity_value(payload.get("last_name"), payload.get("family_name"))

#     composed_name = None
#     if first_name or last_name:
#         composed_name = " ".join(part for part in [first_name, last_name] if part)

#     return _pick_identity_value(
#         payload.get("name"),
#         payload.get("full_name"),
#         composed_name,
#         email.split("@")[0] if email else None,
#     )


# def _decode_clerk_token(token: str) -> Dict[str, Any]:
#     settings = get_settings()

#     jwks_url = settings.clerk_jwks_url
#     if not jwks_url:
#         raise HTTPException(status_code=503, detail="Clerk JWKS URL is not configured.")

#     try:
#         signing_key = _jwks_client_for_url(jwks_url).get_signing_key_from_jwt(token)
#     except Exception as exc:
#         raise HTTPException(status_code=401, detail="Invalid authentication token") from exc

#     decode_kwargs: Dict[str, Any] = {
#         "key": signing_key.key,
#         "algorithms": ["RS256"],
#         "options": {
#             "verify_aud": bool(settings.CLERK_JWT_AUDIENCE),
#             "verify_iss": bool(settings.CLERK_JWT_ISSUER),
#         },
#     }

#     if settings.CLERK_JWT_AUDIENCE:
#         decode_kwargs["audience"] = settings.CLERK_JWT_AUDIENCE

#     if settings.CLERK_JWT_ISSUER:
#         decode_kwargs["issuer"] = settings.CLERK_JWT_ISSUER

#     try:
#         payload = jwt.decode(token, **decode_kwargs)
#     except jwt.ExpiredSignatureError as exc:
#         raise HTTPException(status_code=401, detail="Authentication token expired") from exc
#     except jwt.InvalidTokenError as exc:
#         raise HTTPException(status_code=401, detail="Invalid authentication token") from exc

#     if not payload.get("sub"):
#         raise HTTPException(status_code=401, detail="Authentication token missing subject")

#     return payload


# def create_access_token(
#     user_id: str,
#     extra_claims: Optional[Dict[str, Any]] = None,
# ) -> str:
#     """Legacy JWT helper retained for backward compatibility."""
#     settings = get_settings()
#     now = datetime.now(timezone.utc)
#     payload = {
#         "sub": user_id,
#         "iat": now,
#         "exp": now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
#         "jti": str(uuid.uuid4()),
#         "type": "access",
#     }
#     if extra_claims:
#         payload.update(extra_claims)
#     return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


# def create_refresh_token(user_id: str) -> str:
#     """Legacy JWT helper retained for backward compatibility."""
#     settings = get_settings()
#     now = datetime.now(timezone.utc)
#     payload = {
#         "sub": user_id,
#         "iat": now,
#         "exp": now + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
#         "jti": str(uuid.uuid4()),
#         "type": "refresh",
#     }
#     return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


# def decode_token(token: str) -> Dict[str, Any]:
#     """Decode and verify a legacy JWT token. Raises on failure."""
#     settings = get_settings()
#     try:
#         return jwt.decode(
#             token,
#             settings.JWT_SECRET_KEY,
#             algorithms=[settings.JWT_ALGORITHM],
#         )
#     except jwt.ExpiredSignatureError as exc:
#         raise HTTPException(status_code=401, detail="Token expired") from exc
#     except jwt.InvalidTokenError as exc:
#         raise HTTPException(status_code=401, detail="Invalid token") from exc


# def get_auth_cors_headers() -> List[str]:
#     """Additional CORS headers required by auth layer."""
#     return ["Authorization"]


# def get_auth_middleware():
#     """No framework middleware required for Clerk JWT auth."""
#     return None


# # Backward-compatible aliases used by legacy imports.
# def get_supertokens_cors_headers() -> List[str]:
#     return get_auth_cors_headers()


# def get_supertokens_middleware():
#     return get_auth_middleware()


# async def _load_role_for_user(user_id: str, fallback_role: str = "user") -> str:
#     try:
#         async with AsyncSessionLocal() as db:
#             user = await get_user_by_user_id(db, user_id)
#             return user.role if user else fallback_role
#     except Exception as exc:  # pragma: no cover - defensive
#         logger.debug("Role lookup failed for %s: %s", user_id, exc)
#         return fallback_role


# async def _get_current_user_from_clerk_token(
#     credentials: Optional[HTTPAuthorizationCredentials],
#     *,
#     session_required: bool,
# ) -> Optional[Dict[str, Any]]:
#     if not credentials or not credentials.credentials:
#         if session_required:
#             raise HTTPException(
#                 status_code=401,
#                 detail="Authentication required",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )
#         return None

#     payload = _decode_clerk_token(credentials.credentials)

#     clerk_id = payload["sub"]
#     user_id = clerk_id
#     role = payload.get("role", "user")
    
#     try:
#         from sqlalchemy import select
#         from database.models import User
#         async with AsyncSessionLocal() as db:
#             result = await db.execute(select(User).where(User.clerk_user_id == clerk_id))
#             user = result.scalars().first()
#             if not user:
#                 result = await db.execute(select(User).where(User.user_id == clerk_id))
#                 user = result.scalars().first()
            
#             if user:
#                 user_id = user.user_id
#                 role = user.role
#     except Exception as exc:
#         logger.debug("User lookup failed for %s: %s", clerk_id, exc)

#     email = _extract_email_from_payload(payload)
#     name = _extract_name_from_payload(payload, email)

#     return {
#         "user_id": user_id,
#         "role": role,
#         "token_type": "clerk_jwt",
#         "authenticated": True,
#         "session": None,
#         "session_payload": payload,
#         "email": email,
#         "name": name,
#         "provider": "clerk",
#     }


# async def _get_current_user_from_legacy_token(
#     credentials: Optional[HTTPAuthorizationCredentials],
# ) -> Dict[str, Any]:
#     settings = get_settings()

#     if credentials and credentials.credentials:
#         payload = decode_token(credentials.credentials)
#         user_id = payload["sub"]
#         role = await _load_role_for_user(user_id, payload.get("role", "user"))
#         return {
#             "user_id": user_id,
#             "role": role,
#             "token_type": payload.get("type", "access"),
#             "authenticated": True,
#             "session": None,
#             "session_payload": payload,
#             "email": payload.get("email"),
#             "name": payload.get("name"),
#             "provider": payload.get("provider"),
#         }

#     if not settings.is_production:
#         anon_id = f"anon-{uuid.uuid4().hex[:12]}"
#         logger.debug("Anonymous legacy session bootstrapped: %s", anon_id)
#         return {
#             "user_id": anon_id,
#             "role": "user",
#             "token_type": "anonymous",
#             "authenticated": False,
#             "session": None,
#             "session_payload": {},
#             "email": None,
#             "name": None,
#             "provider": "anonymous",
#         }

#     raise HTTPException(
#         status_code=401,
#         detail="Authentication required",
#         headers={"WWW-Authenticate": "Bearer"},
#     )


# async def get_current_user(
#     request: Request,
#     credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
# ) -> Dict[str, Any]:
#     """
#     Resolve the current user from Clerk JWTs when configured.
#     Falls back to the legacy JWT path for backward compatibility.
#     """
#     if _clerk_requested():
#         user = await _get_current_user_from_clerk_token(credentials, session_required=True)
#         if user is None:  # pragma: no cover - defensive
#             raise HTTPException(status_code=401, detail="Authentication required")
#         return user

#     return await _get_current_user_from_legacy_token(credentials)


# async def get_optional_user(
#     request: Request,
#     credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
# ) -> Optional[Dict[str, Any]]:
#     """Like get_current_user but returns None instead of raising."""
#     try:
#         if _clerk_requested():
#             return await _get_current_user_from_clerk_token(
#                 credentials, session_required=False
#             )
#         return await _get_current_user_from_legacy_token(credentials)
#     except HTTPException:
#         return None


# async def sync_authenticated_user(
#     db,
#     current_user: Dict[str, Any],
#     *,
#     email: Optional[str] = None,
#     name: Optional[str] = None,
#     provider: Optional[str] = None,
# ):
#     payload = current_user.get("session_payload") or {}
#     resolved_email = _pick_identity_value(
#         email,
#         current_user.get("email"),
#         payload.get("email"),
#         payload.get("email_address"),
#     )
#     resolved_name = _pick_identity_value(
#         name,
#         current_user.get("name"),
#         payload.get("name"),
#         payload.get("full_name"),
#         resolved_email.split("@")[0] if resolved_email else None,
#     )
#     resolved_provider = _pick_identity_value(
#         provider,
#         current_user.get("provider"),
#         payload.get("provider"),
#         "clerk" if _clerk_requested() else None,
#     )

#     user_record = await upsert_authenticated_user(
#         db,
#         user_id=current_user["user_id"],
#         email=resolved_email,
#         name=resolved_name,
#         provider=resolved_provider,
#     )

#     current_user.update(
#         {
#             "role": user_record.role,
#             "email": user_record.email,
#             "name": user_record.name,
#             "provider": user_record.provider,
#         }
#     )

#     return user_record


# def serialize_current_user(
#     current_user: Dict[str, Any],
#     user_record: Optional[Any] = None,
# ) -> Dict[str, Any]:
#     source = user_record or current_user
#     user_id = getattr(source, "user_id", None) or current_user.get("user_id")
#     provider = getattr(source, "provider", None) or current_user.get("provider")
#     email = getattr(source, "email", None) or current_user.get("email")
#     name = getattr(source, "name", None) or current_user.get("name")
#     role = getattr(source, "role", None) or current_user.get("role", "user")
#     record_id = getattr(source, "id", None)

#     return {
#         "id": str(record_id) if record_id is not None else user_id,
#         "user_id": user_id,
#         "email": email,
#         "name": name or (email.split("@")[0] if email else "User"),
#         "provider": provider,
#         "role": role,
#         "is_authenticated": current_user.get("authenticated", False),
#     }


# def require_admin():
#     """Decorator to require admin role for a route."""

#     def decorator(func):
#         @functools.wraps(func)
#         async def wrapper(*args, **kwargs):
#             current_user = kwargs.get("current_user")
#             if current_user and current_user.get("role") != "admin":
#                 settings = get_settings()
#                 if settings.is_production or current_user.get("authenticated", False):
#                     raise HTTPException(
#                         status_code=403, detail="Admin privileges required"
#                     )
#             return await func(*args, **kwargs)

#         return wrapper

#     return decorator

# """
# Unified Auth Layer — Clerk + Optional Anonymous Support
# """

# import jwt
# from typing import Optional, Dict, Any
# from fastapi import Depends, HTTPException, Request
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from jose import jwt as jose_jwt
# from jose import jwk
# from jose.utils import base64url_decode
# import requests

# from gateway.config import get_settings

# security = HTTPBearer(auto_error=False)


# def verify_clerk_token(token: str) -> Dict[str, Any]:
#     settings = get_settings()

#     jwks_url = settings.clerk_jwks_url
#     if not jwks_url:
#         raise HTTPException(status_code=500, detail="Clerk not configured")

#     jwks = requests.get(jwks_url).json()

#     headers = jwt.get_unverified_header(token)
#     kid = headers.get("kid")

#     key = next((k for k in jwks["keys"] if k["kid"] == kid), None)
#     if not key:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     public_key = jwk.construct(key)

#     message, encoded_signature = token.rsplit(".", 1)
#     decoded_signature = base64url_decode(encoded_signature.encode())

#     if not public_key.verify(message.encode(), decoded_signature):
#         raise HTTPException(status_code=401, detail="Invalid signature")

#     payload = jose_jwt.get_unverified_claims(token)

#     return payload


# async def get_current_user(
#     request: Request,
#     credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
# ) -> Dict[str, Any]:

#     if not credentials or not credentials.credentials:
#         raise HTTPException(status_code=401, detail="Authentication required")

#     payload = verify_clerk_token(credentials.credentials)

#     return {
#         "user_id": payload.get("sub"),
#         "email": payload.get("email"),
#         "role": payload.get("role", "user"),
#         "authenticated": True,
#     }


# async def get_optional_user(
#     request: Request,
#     credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
# ) -> Optional[Dict[str, Any]]:

#     try:
#         if not credentials:
#             return None
#         payload = verify_clerk_token(credentials.credentials)
#         return {
#             "user_id": payload.get("sub"),
#             "email": payload.get("email"),
#             "role": payload.get("role", "user"),
#             "authenticated": True,
#         }
#     except Exception:
#         return None

# # ── Admin Guard (REQUIRED) ─────────────────────────────

# async def require_admin(
#     user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """
#     Ensures user has admin role.
#     Works with Clerk-based auth.
#     """

#     if not user:
#         raise HTTPException(status_code=401, detail="Not authenticated")

#     if user.get("role") != "admin":
#         raise HTTPException(status_code=403, detail="Admin access required")

#     return user



"""
Firebase-backed authentication compatibility layer.

This module preserves the older import path used throughout the backend
while delegating all live auth behavior to the Firebase implementation.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request

from gateway.auth_v2 import (
    extract_token_from_header,
    get_current_user as firebase_get_current_user,
    get_guest_user,
    resolve_temp_user_from_request,
    TEMP_AUTH_DISABLED,
    verify_firebase_token,
)


async def get_current_user(
    request: Request,
) -> Dict[str, Any]:
    if TEMP_AUTH_DISABLED:
        # TODO: Restore Firebase Auth after configuration fixes
        temp_user = resolve_temp_user_from_request(request)
        if not temp_user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        request.state.user_id = temp_user["user_id"]
        request.state.current_user = temp_user
        return temp_user
    return await firebase_get_current_user(request=request)


async def get_optional_user(
    request: Request,
) -> Optional[Dict[str, Any]]:
    if TEMP_AUTH_DISABLED:
        return resolve_temp_user_from_request(request)
    try:
        return await firebase_get_current_user(request=request)
    except HTTPException:
        return None


async def require_admin(
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if TEMP_AUTH_DISABLED:
        # TODO: Restore Firebase Auth after configuration fixes
        return user or get_guest_user()

    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return user


async def get_user_id(request: Request) -> Optional[str]:
    """Returns the Firebase UID from the Authorization header if present."""
    try:
        if TEMP_AUTH_DISABLED:
            user = resolve_temp_user_from_request(request)
            return user.get("user_id") if user else None

        auth_header = request.headers.get("Authorization")
        token = extract_token_from_header(auth_header)
        if not token:
            return None

        payload = await verify_firebase_token(token)
        return payload.get("uid") if payload else None
    except Exception:
        return None
