"""Central, typed configuration for EvalPipeline Pro.

Every value is read from environment variables (or a local `.env` file) via
pydantic-settings, so the same image runs unchanged in docker-compose, GitHub
Actions and any hosting target. Field names map 1:1 (case-insensitive) to the
keys documented in `.env.example`.

Python 3.9 compatible: `Optional[...]` instead of `X | None`.
"""

from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # unknown keys in .env (e.g. compose-only vars) are not an error
    )

    # ------------------------------------------------------------------ application
    app_name: str = "EvalPipeline Pro"
    app_version: str = "0.1.0"
    environment: Literal["local", "ci", "test", "prod"] = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ------------------------------------------------------------------ database
    # TODO(database): placeholders — real values come from .env / the hosting platform's secrets.
    postgres_user: str = "evalpipeline"
    postgres_password: SecretStr = SecretStr("CHANGE_ME")
    postgres_db: str = "evalpipeline"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    # Full async DSN; when set it wins over the POSTGRES_* pieces above.
    database_url: Optional[SecretStr] = None
    db_pool_size: int = Field(5, ge=1)
    db_max_overflow: int = Field(10, ge=0)
    db_echo_sql: bool = False

    # ------------------------------------------------------------------ LLM (generation + judge)
    # Any OpenAI-compatible endpoint. Groq by default; swap base URL + key for OpenAI.
    llm_provider: Literal["groq", "openai"] = "groq"
    llm_api_key: SecretStr = SecretStr("")  # TODO(apikey): set LLM_API_KEY in .env
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"
    judge_model: str = "llama-3.3-70b-versatile"
    llm_timeout_seconds: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(3, ge=0)  # network/429 retries (OpenAI client) AND schema re-asks
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)

    # ------------------------------------------------------------------ embeddings
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(384, gt=0)  # must match the model AND the pgvector column width
    openai_embedding_api_key: SecretStr = SecretStr("")  # TODO(apikey): only for provider=openai

    # ------------------------------------------------------------------ RAG
    rag_top_k: int = Field(5, ge=1, le=50)
    prompt_version: str = "v1"

    # ------------------------------------------------------------------ evaluation engine
    eval_quality_threshold: float = Field(0.85, ge=0.0, le=1.0)
    eval_max_concurrency: int = Field(5, ge=1, le=64)

    # ------------------------------------------------------------------ dashboard
    api_base_url: str = "http://localhost:8000"  # TODO(hosting): deployed API URL

    # ------------------------------------------------------------------ validators / helpers
    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"LOG_LEVEL must be a standard logging level, got {value!r}")
        return level

    @property
    def sqlalchemy_database_url(self) -> str:
        """Async SQLAlchemy DSN (postgresql+asyncpg://...)."""
        if self.database_url is not None:
            return self.database_url.get_secret_value()
        return (
            f"postgresql+asyncpg://{quote_plus(self.postgres_user)}:"
            f"{quote_plus(self.postgres_password.get_secret_value())}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def llm_configured(self) -> bool:
        """True when a generation/judge key is present; endpoints that need it fail fast otherwise."""
        return bool(self.llm_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton; `get_settings.cache_clear()` in tests to re-read env."""
    return Settings()
