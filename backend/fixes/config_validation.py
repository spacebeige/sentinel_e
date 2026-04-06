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
    if settings.is_production:
        # Database
        if not settings.POSTGRES_HOST or settings.POSTGRES_HOST == "localhost":
            errors.append("❌ POSTGRES_HOST must be configured for production (not localhost)")
        
        # Cache
        if not settings.REDIS_HOST or settings.REDIS_HOST == "localhost":
            errors.append("❌ REDIS_HOST must be configured for production (not localhost)")
        
        # CORS
        if "*" in settings.ALLOWED_ORIGINS:
            errors.append("❌ ALLOWED_ORIGINS cannot be '*' in production (CORS vulnerability)")
        
        # JWT Secret
        if "CHANGE-ME" in settings.JWT_SECRET_KEY:
            errors.append("❌ JWT_SECRET_KEY still uses default 'CHANGE-ME' value in production!")
        
        if len(settings.JWT_SECRET_KEY) < 32:
            errors.append("❌ JWT_SECRET_KEY too short (must be 32+ characters)")
        
        # Database URL
        if "password" not in settings.DATABASE_URL or "password_placeholder" in settings.DATABASE_URL:
            errors.append("❌ DATABASE_URL missing credentials or using placeholder")
    
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
