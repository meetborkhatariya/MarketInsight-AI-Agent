"""
Reusable Groq LLM initialiser.

Exposes ``get_llm()`` which returns a singleton ``ChatGroq`` instance
configured from the application settings.  Callers never handle API
keys or model names directly.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_groq import ChatGroq

from config import get_settings

logger = logging.getLogger(__name__)

# Module-level cache for the singleton instance.
_llm_instance: Optional[ChatGroq] = None
_llm_cache_key: tuple[str, str, float] | None = None


def get_llm(
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatGroq:
    """Return a configured ``ChatGroq`` singleton.

    The first call creates the instance using values from
    ``config.settings``; subsequent calls return the same cached
    instance regardless of ``model`` / ``temperature`` overrides.

    Parameters
    ----------
    model:
        Override the model name from ``settings.groq_model``.
        ``None`` (default) uses the value from configuration.
    temperature:
        Sampling temperature (0.0 = deterministic, 1.0 = creative).
        Defaults to 0.0.  Only honoured on the **first** call.

    Returns
    -------
    ChatGroq
        A ready-to-use LangChain Groq LLM instance.

    Raises
    ------
    RuntimeError
        If initialisation fails (e.g., invalid API key or network
        unreachable).
    """
    global _llm_instance, _llm_cache_key

    current_settings = get_settings()
    resolved_model = model or current_settings.groq_model
    api_key = current_settings.groq_api_key.get_secret_value()
    cache_key = (resolved_model, api_key, temperature)

    if _llm_instance is not None and _llm_cache_key == cache_key:
        return _llm_instance

    logger.info(
        "Initialising ChatGroq: model=%s, temperature=%s",
        resolved_model,
        temperature,
    )

    try:
        _llm_instance = ChatGroq(
            model=resolved_model,
            temperature=temperature,
            api_key=api_key,
        )
        _llm_cache_key = cache_key
    except Exception as exc:
        logger.exception("Failed to initialise ChatGroq")
        raise RuntimeError(
            f"Groq LLM initialisation failed: {exc}"
        ) from exc

    logger.debug("ChatGroq instance created successfully")
    return _llm_instance
