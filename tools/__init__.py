"""Convenience imports for optional tool modules.

The PDF generator is safe to expose at package import time. The LLM and
search helpers depend on optional third-party packages, so they are
loaded lazily when available instead of breaking unrelated imports like
the Streamlit UI.
"""

from __future__ import annotations

from .pdf_generator import PDFGenerator

try:  # pragma: no cover - optional dependency handling
    from .groq_llm import get_llm
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    get_llm = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency handling
    from .tavily_search import search_market_data
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    search_market_data = None  # type: ignore[assignment]

__all__ = ["PDFGenerator", "get_llm", "search_market_data"]