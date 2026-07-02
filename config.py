"""
Configuration management for MarketInsight AI.

Uses pydantic-settings to load and validate environment variables
from a ``.env`` file at the project root.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

PLACEHOLDER_SUBSTRINGS = ("your_", "changeme", "placeholder")
"""Substrings that indicate an API key hasn't been properly configured."""

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

VALID_SEARCH_DEPTHS = {"basic", "advanced"}

REPO_ROOT = Path(__file__).resolve().parent
"""Absolute path to the project root directory."""


# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Secrets are stored as ``pydantic.SecretStr`` to prevent
    accidental exposure in logs, tracebacks, or ``repr()`` calls.

    Reading priority (highest first):
    1. Environment variables
    2. ``.env`` file at the project root
    3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required API Keys ────────────────────────────────────────

    groq_api_key: SecretStr = SecretStr("")
    """API key for the Groq LLM service."""

    tavily_api_key: SecretStr = SecretStr("")
    """API key for the Tavily search API."""

    # ── LLM Configuration ────────────────────────────────────────

    groq_model: str = "llama-3.3-70b-versatile"
    """Groq model identifier to use for LLM calls."""

    # ── Search Configuration ─────────────────────────────────────

    tavily_search_depth: str = "basic"
    """Search depth: ``"basic"`` or ``"advanced"``."""

    tavily_max_results: int = 5
    """Maximum number of search results to return per query."""

    # ── Output Configuration ─────────────────────────────────────

    report_output_dir: str = "reports"
    """Directory (relative to project root) for generated PDF reports."""

    # ── Observability ────────────────────────────────────────────

    log_level: str = "INFO"
    """Root logger level. One of ``DEBUG | INFO | WARNING | ERROR | CRITICAL``."""

    # ── Validators ───────────────────────────────────────────────

    @field_validator("groq_api_key", "tavily_api_key", mode="before")
    @classmethod
    def _validate_api_key_not_empty(cls, v: str | SecretStr) -> str:
        """Raise ``ValueError`` if a required API key is missing or a placeholder."""
        if isinstance(v, SecretStr):
            raw = v.get_secret_value().strip()
        elif isinstance(v, str):
            raw = v.strip()
        else:
            raw = ""
        if not raw:
            raise ValueError(
                "A required API key is missing. "
                "Set the corresponding environment variable or add it to your .env file."
            )
        if any(sub in raw.lower() for sub in PLACEHOLDER_SUBSTRINGS):
            raise ValueError(
                f"API key looks like a placeholder ('{raw}'). "
                "Replace it with a valid key in your .env file."
            )
        return raw

    @field_validator("tavily_search_depth", mode="before")
    @classmethod
    def _validate_search_depth(cls, v: str) -> str:
        """Normalise and validate the search depth option."""
        raw = v.strip().lower() if isinstance(v, str) else ""
        if raw not in VALID_SEARCH_DEPTHS:
            raise ValueError(
                f"Invalid search depth '{v}'. Must be one of {VALID_SEARCH_DEPTHS}."
            )
        return raw

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Normalise and validate the logging level."""
        raw = v.strip().upper() if isinstance(v, str) else ""
        if raw not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of {VALID_LOG_LEVELS}."
            )
        return raw

    @field_validator("report_output_dir", mode="before")
    @classmethod
    def _resolve_report_dir(cls, v: str) -> str:
        """Convert a relative path to an absolute path anchored at the repo root."""
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return str(p.resolve())

    # ── Derived Properties ──────────────────────────────────────

    @property
    def resolved_report_dir(self) -> Path:
        """Report output directory as a ``Path`` object."""
        return Path(self.report_output_dir)


# ────────────────────────────────────────────────────────────────
# Singleton
# ────────────────────────────────────────────────────────────────

settings = Settings()
"""Application-wide singleton settings object.

Usage::

    from config import settings

    client = GroqLLM(api_key=settings.groq_api_key.get_secret_value())
"""


# ────────────────────────────────────────────────────────────────
# Bootstrap
# ────────────────────────────────────────────────────────────────

def configure_logging() -> None:
    """Configure the root logger based on the loaded settings."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


configure_logging()
