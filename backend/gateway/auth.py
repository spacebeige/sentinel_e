"""
============================================================
Authentication & Session Management
============================================================
Primary auth path:
- SuperTokens ThirdParty (Google + GitHub)
- SuperTokens Session (httpOnly cookies)
- Neon-backed user profile upsert into existing users table

Legacy compatibility:
- JWT helpers remain for older tests / fallback code paths
"""

import functools
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from database.connection import AsyncSessionLocal
from database.crud import get_user_by_user_id, upsert_authenticated_user
from gateway.config import get_settings

logger = logging.getLogger("Auth")
security = HTTPBearer(auto_error=False)

try:
    from supertokens_python import (
        InputAppInfo,
        SupertokensConfig,
        get_all_cors_headers as supertokens_get_all_cors_headers,
        init as supertokens_init,
    )
    from supertokens_python.framework.fastapi import get_middleware as get_supertokens_middleware_impl
    from supertokens_python.recipe.session import SessionContainer
    from supertokens_python.recipe.session.framework.fastapi import verify_session as supertokens_verify_session
    import supertokens_python.recipe.session as SessionRecipe
    import supertokens_python.recipe.thirdparty as ThirdPartyRecipe
    from supertokens_python.recipe.thirdparty import SignInAndUpFeature
    from supertokens_python.recipe.thirdparty.provider import (
        ProviderClientConfig,
        ProviderConfig,
        ProviderInput,
    )
    from supertokens_python.recipe.thirdparty.providers.github import Github
    from supertokens_python.recipe.thirdparty.providers.google import Google

    SUPERTOKENS_SDK_AVAILABLE = True
    SUPERTOKENS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment-dependent
    SessionContainer = Any  # type: ignore
    SUPERTOKENS_SDK_AVAILABLE = False
    SUPERTOKENS_IMPORT_ERROR = exc


_SUPERTOKENS_INITIALIZED = False


def create_access_token(
    user_id: str,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Legacy JWT helper retained for backward compatibility."""
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
    """Legacy JWT helper retained for backward compatibility."""
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
    """Decode and verify a legacy JWT token. Raises on failure."""
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def _supertokens_requested() -> bool:
    return bool(get_settings().SUPERTOKENS_CONNECTION_URI)


def _supertokens_enabled() -> bool:
    return _supertokens_requested() and SUPERTOKENS_SDK_AVAILABLE


def _build_provider_input(
    third_party_id: str,
    client_id: str,
    client_secret: str,
    scope: Optional[List[str]] = None,
) -> ProviderInput:
    return ProviderInput(
        config=ProviderConfig(
            third_party_id=third_party_id,
            clients=[
                ProviderClientConfig(
                    client_id=client_id,
                    client_secret=client_secret,
                    scope=scope,
                )
            ],
        )
    )


def _build_supertokens_providers() -> List[Any]:
    settings = get_settings()
    providers: List[Any] = []

    if settings.GOOGLE_OAUTH_CLIENT_ID and settings.GOOGLE_OAUTH_CLIENT_SECRET:
        providers.append(
            Google(
                _build_provider_input(
                    "google",
                    settings.GOOGLE_OAUTH_CLIENT_ID,
                    settings.GOOGLE_OAUTH_CLIENT_SECRET,
                    scope=["openid", "email", "profile"],
                )
            )
        )

    if settings.GITHUB_OAUTH_CLIENT_ID and settings.GITHUB_OAUTH_CLIENT_SECRET:
        providers.append(
            Github(
                _build_provider_input(
                    "github",
                    settings.GITHUB_OAUTH_CLIENT_ID,
                    settings.GITHUB_OAUTH_CLIENT_SECRET,
                    scope=["read:user", "user:email"],
                )
            )
        )

    return providers


def init_supertokens() -> bool:
    global _SUPERTOKENS_INITIALIZED

    if _SUPERTOKENS_INITIALIZED:
        return True

    if not _supertokens_requested():
        return False

    if not SUPERTOKENS_SDK_AVAILABLE:
        logger.error("SuperTokens SDK import failed: %s", SUPERTOKENS_IMPORT_ERROR)
        return False

    settings = get_settings()
    providers = _build_supertokens_providers()

    try:
        supertokens_init(
            app_info=InputAppInfo(
                app_name=settings.APP_NAME,
                api_domain=settings.API_DOMAIN,
                website_domain=settings.WEBSITE_DOMAIN,
                api_base_path=settings.SUPERTOKENS_API_BASE_PATH,
                website_base_path=settings.SUPERTOKENS_WEBSITE_BASE_PATH,
            ),
            framework="fastapi",
            supertokens_config=SupertokensConfig(
                connection_uri=settings.SUPERTOKENS_CONNECTION_URI,
                api_key=settings.SUPERTOKENS_API_KEY or None,
            ),
            recipe_list=[
                ThirdPartyRecipe.init(
                    sign_in_and_up_feature=SignInAndUpFeature(providers=providers)
                ),
                SessionRecipe.init(
                    cookie_secure=settings.supertokens_cookie_secure,
                    cookie_same_site=settings.supertokens_cookie_same_site,
                    cookie_domain=settings.SUPERTOKENS_COOKIE_DOMAIN,
                    anti_csrf="VIA_TOKEN",
                    expose_access_token_to_frontend_in_cookie_based_auth=False,
                ),
            ],
            mode="asgi",
            telemetry=False,
            debug=settings.DEBUG,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.error("SuperTokens initialization failed: %s", exc)
        return False

    _SUPERTOKENS_INITIALIZED = True
    logger.info(
        "SuperTokens initialized with %d third-party provider(s).",
        len(providers),
    )
    return True


def get_supertokens_cors_headers() -> List[str]:
    if not init_supertokens():
        return []
    try:
        return supertokens_get_all_cors_headers()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to fetch SuperTokens CORS headers: %s", exc)
        return []


def get_supertokens_middleware():
    if not init_supertokens():
        return None
    return get_supertokens_middleware_impl()


async def _load_role_for_user(user_id: str, fallback_role: str = "user") -> str:
    try:
        async with AsyncSessionLocal() as db:
            user = await get_user_by_user_id(db, user_id)
            return user.role if user else fallback_role
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Role lookup failed for %s: %s", user_id, exc)
        return fallback_role


async def _get_current_user_from_supertokens(
    request: Request,
    *,
    session_required: bool,
) -> Optional[Dict[str, Any]]:
    if not init_supertokens():
        if _supertokens_requested():
            raise HTTPException(status_code=503, detail="Authentication is not available.")
        return None

    verifier = supertokens_verify_session(session_required=session_required)
    session = await verifier(request)
    if session is None:
        return None

    payload = session.get_access_token_payload() or {}
    user_id = session.get_user_id()
    role = await _load_role_for_user(user_id, payload.get("role", "user"))

    return {
        "user_id": user_id,
        "role": role,
        "token_type": "session",
        "authenticated": True,
        "session": session,
        "session_payload": payload,
        "email": payload.get("email"),
        "name": payload.get("name"),
        "provider": payload.get("provider"),
    }


async def _get_current_user_from_legacy_token(
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Dict[str, Any]:
    settings = get_settings()

    if credentials and credentials.credentials:
        payload = decode_token(credentials.credentials)
        user_id = payload["sub"]
        role = await _load_role_for_user(user_id, payload.get("role", "user"))
        return {
            "user_id": user_id,
            "role": role,
            "token_type": payload.get("type", "access"),
            "authenticated": True,
            "session": None,
            "session_payload": payload,
            "email": payload.get("email"),
            "name": payload.get("name"),
            "provider": payload.get("provider"),
        }

    if not settings.is_production:
        anon_id = f"anon-{uuid.uuid4().hex[:12]}"
        logger.debug("Anonymous legacy session bootstrapped: %s", anon_id)
        return {
            "user_id": anon_id,
            "role": "user",
            "token_type": "anonymous",
            "authenticated": False,
            "session": None,
            "session_payload": {},
            "email": None,
            "name": None,
            "provider": "anonymous",
        }

    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    Resolve the current user from SuperTokens sessions when configured.
    Falls back to the legacy JWT path for backward compatibility.
    """
    if _supertokens_requested():
        user = await _get_current_user_from_supertokens(request, session_required=True)
        if user is None:  # pragma: no cover - defensive
            raise HTTPException(status_code=401, detail="Authentication required")
        return user

    return await _get_current_user_from_legacy_token(credentials)


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Like get_current_user but returns None instead of raising."""
    try:
        if _supertokens_requested():
            return await _get_current_user_from_supertokens(
                request, session_required=False
            )
        return await _get_current_user_from_legacy_token(credentials)
    except HTTPException:
        return None


def _pick_identity_value(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def sync_authenticated_user(
    db,
    current_user: Dict[str, Any],
    *,
    email: Optional[str] = None,
    name: Optional[str] = None,
    provider: Optional[str] = None,
):
    payload = current_user.get("session_payload") or {}
    resolved_email = _pick_identity_value(
        email,
        current_user.get("email"),
        payload.get("email"),
    )
    resolved_name = _pick_identity_value(
        name,
        current_user.get("name"),
        payload.get("name"),
        resolved_email.split("@")[0] if resolved_email else None,
    )
    resolved_provider = _pick_identity_value(
        provider,
        current_user.get("provider"),
        payload.get("provider"),
    )

    user_record = await upsert_authenticated_user(
        db,
        user_id=current_user["user_id"],
        email=resolved_email,
        name=resolved_name,
        provider=resolved_provider,
    )

    current_user.update(
        {
            "role": user_record.role,
            "email": user_record.email,
            "name": user_record.name,
            "provider": user_record.provider,
        }
    )

    session: Optional[SessionContainer] = current_user.get("session")
    if session is not None:
        desired_claims = {"role": user_record.role}
        if user_record.email:
            desired_claims["email"] = user_record.email
        if user_record.name:
            desired_claims["name"] = user_record.name
        if user_record.provider:
            desired_claims["provider"] = user_record.provider

        current_payload = session.get_access_token_payload() or {}
        if any(current_payload.get(key) != value for key, value in desired_claims.items()):
            await session.merge_into_access_token_payload(desired_claims)

    return user_record


def serialize_current_user(
    current_user: Dict[str, Any],
    user_record: Optional[Any] = None,
) -> Dict[str, Any]:
    source = user_record or current_user
    user_id = getattr(source, "user_id", None) or current_user.get("user_id")
    provider = getattr(source, "provider", None) or current_user.get("provider")
    email = getattr(source, "email", None) or current_user.get("email")
    name = getattr(source, "name", None) or current_user.get("name")
    role = getattr(source, "role", None) or current_user.get("role", "user")
    record_id = getattr(source, "id", None)

    return {
        "id": str(record_id) if record_id is not None else user_id,
        "user_id": user_id,
        "email": email,
        "name": name or (email.split("@")[0] if email else "User"),
        "provider": provider,
        "role": role,
        "is_authenticated": current_user.get("authenticated", False),
    }


def require_admin():
    """Decorator to require admin role for a route."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if current_user and current_user.get("role") != "admin":
                settings = get_settings()
                if settings.is_production or current_user.get("authenticated", False):
                    raise HTTPException(
                        status_code=403, detail="Admin privileges required"
                    )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


init_supertokens()
