"""
============================================================
Sentinel-E Production Configuration
============================================================
Centralized configuration with environment-based overrides.
No hardcoded secrets. No debug printing of credentials.
"""

import os
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

# Determine base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Settings(BaseSettings):
    """Production settings — all secrets from environment."""

    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "Sentinel-E"
    APP_VERSION: str = "5.0.0"
    ENVIRONMENT: str = Field(default="development", description="development | staging | production")
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Security ─────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(default="CHANGE-ME-IN-PRODUCTION-USE-openssl-rand-hex-64", description="JWT signing key")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    ALLOWED_ORIGINS: str = Field(default="http://localhost:3000", description="Comma-separated CORS origins")
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    MAX_INPUT_LENGTH: int = 50000  # characters
    MAX_ROUNDS: int = 10

    # ── LLM Providers ────────────────────────────────────────
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # ── MCO Per-Model Keys (provider isolation) ───────────────
    QWEN3_CODER_API_KEY: str = ""
    QWEN3_VL_API_KEY: str = ""
    NEMOTRON_API_KEY: str = ""
    KIMI_API_KEY: str = ""

    # ── Database ─────────────────────────────────────────────
    POSTGRES_URL: str = ""
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "sentinel_sigma"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    DATABASE_URL: str = ""

    # ── Redis ────────────────────────────────────────────────
    REDIS_URL: str = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_DB: str = "0"
    REDIS_SESSION_TTL: int = 7200  # 2 hours

    # ── Memory System ────────────────────────────────────────
    SHORT_TERM_MEMORY_SIZE: int = 12  # messages
    ROLLING_SUMMARY_INTERVAL: int = 8  # every N exchanges
    ROLLING_SUMMARY_MAX_TOKENS: int = 500

    # ── RAG ──────────────────────────────────────────────────
    RAG_CONFIDENCE_THRESHOLD: float = 0.6
    RAG_MAX_SOURCES: int = 5
    RAG_TAVILY_MAX_RESULTS: int = 5

    # ── Provider Defaults ────────────────────────────────────
    DEFAULT_PROVIDER: str = "groq"
    TOKEN_BUDGET_PER_REQUEST: int = 4096
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BASE_DELAY: float = 1.0

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def effective_database_url(self) -> str:
        """Resolve database URL with asyncpg compatibility."""
        url = self.DATABASE_URL or self.POSTGRES_URL
        if url:
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            # Strip parameters incompatible with asyncpg
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            params.pop("sslmode", None)
            params.pop("channel_binding", None)
            new_query = urlencode(params, doseq=True)
            return urlunparse(parsed._replace(query=new_query))
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def effective_ssl_required(self) -> bool:
        url = self.DATABASE_URL or self.POSTGRES_URL
        if url and "sslmode=require" in url:
            return True
        return False

    class Config:
        env_file = os.path.join(BASE_DIR, ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()
