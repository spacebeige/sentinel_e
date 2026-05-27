"""
Configuration validation fixes
Ensures all required environment variables are set before startup.
Prevents hard-to-debug silent failures in production.
"""

from typing import Optional
from pydantic import validator
from pydantic_settings import BaseSettings
import sys
import logging

logger = logging.getLogger(__name__)


def validate_production_config():
    """
    Validate that settings are safe for production.
    Call this during app startup.
    Fixes: Hardcoded localhost URLs, weak JWT secrets
    """
    from gateway.config import get_settings
    settings = get_settings()
    
    errors = []
    
    # Check critical environment variables
    if getattr(settings, "is_production", False):
        database_url = getattr(settings, "DATABASE_URL", "")
        postgres_host = getattr(settings, "POSTGRES_HOST", "")
        redis_host = getattr(settings, "REDIS_HOST", "")
        allowed_origins = getattr(settings, "ALLOWED_ORIGINS", "")
        jwt_secret_key = getattr(settings, "JWT_SECRET_KEY", "")
        clerk_jwt_issuer = getattr(settings, "CLERK_JWT_ISSUER", "")
        clerk_jwks_url = getattr(settings, "CLERK_JWKS_URL", "")
        api_domain = getattr(settings, "API_DOMAIN", "")
        website_domain = getattr(settings, "WEBSITE_DOMAIN", "")

        # Database
        if not database_url and (not postgres_host or postgres_host == "localhost"):
            errors.append("❌ POSTGRES_HOST must be configured for production (not localhost)")
        
        # Cache
        if not redis_host or redis_host == "localhost":
            errors.append("❌ REDIS_HOST must be configured for production (not localhost)")
        
        # CORS
        if allowed_origins and "*" in allowed_origins:
            errors.append("❌ ALLOWED_ORIGINS cannot be '*' in production (CORS vulnerability)")
        
        # JWT Secret
        if jwt_secret_key:
            if "CHANGE-ME" in jwt_secret_key:
                errors.append("❌ JWT_SECRET_KEY still uses default 'CHANGE-ME' value in production!")
            
            if len(jwt_secret_key) < 32:
                errors.append("❌ JWT_SECRET_KEY too short (must be 32+ characters)")
        
        # Database URL
        if database_url and ("password_placeholder" in database_url):
            errors.append("❌ DATABASE_URL missing credentials or using placeholder")

        # Clerk
        if not clerk_jwt_issuer and not clerk_jwks_url:
            errors.append("❌ CLERK_JWT_ISSUER or CLERK_JWKS_URL must be configured for production")
        if api_domain and not api_domain.startswith("https://"):
            errors.append("❌ API_DOMAIN must use https in production")
        if website_domain and not website_domain.startswith("https://"):
            errors.append("❌ WEBSITE_DOMAIN must use https in production")
    
    if errors:
        logger.error("⚠️  CONFIGURATION VALIDATION FAILED:")
        for error in errors:
            logger.error(error)
        
        if settings.is_production:
            logger.critical("🛑 Refusing to start in production with invalid config")
            sys.exit(1)
        else:
            logger.warning("⚠️  Running in development mode; these issues will cause production failures")
    
    logger.info("✅ Configuration validation passed")


def validate_required_env_var(varname: str, description: str, default: Optional[str] = None) -> str:
    """
    Validate a required environment variable exists.
    Fixes: Missing env vars causing silent failures
    
    Args:
        varname: Environment variable name
        description: Human-readable description of what this var does
        default: Optional default value (if needed)
    
    Returns:
        The environment variable value or default
    
    Raises:
        ValueError if required var is missing
    """
    import os
    
    value = os.getenv(varname, default)
    
    if not value:
        raise ValueError(
            f"❌ Required environment variable '{varname}' is not set.\n"
            f"   Purpose: {description}\n"
            f"   Set it with: export {varname}=<value>"
        )
    
    logger.debug(f"✓ {varname} configured")
    return value
