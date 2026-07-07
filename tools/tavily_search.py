"""
Tavily web search wrapper for market research.

Exposes ``search_market_data()`` which returns cleaned, deduplicated
search results using the ``langchain-tavily`` integration.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from tavily import TavilyClient
from config import get_settings

logger = logging.getLogger(__name__)

# Module-level cache for the singleton search tool.
_search_tool: Optional[TavilyClient] = None
_search_api_key: Optional[str] = None
# Fields extracted from each search result.
_KEEP_KEYS = frozenset({"title", "url", "content", "score"})

def _get_search_tool() -> TavilyClient:
    global _search_tool, _search_api_key

    current_settings = get_settings()
    api_key = current_settings.tavily_api_key.get_secret_value()

    if _search_tool is None or _search_api_key != api_key:
        _search_tool = TavilyClient(
            api_key=api_key
        )
        _search_api_key = api_key

    return _search_tool


def search_market_data(query: str) -> list[dict[str, Any]]:
    """Search the web for market research data on the given *query*.

    Parameters
    ----------
    query:
        Natural-language market research question
        (e.g. ``"EV battery market size Southeast Asia 2025"``).

    Returns
    -------
    list[dict[str, Any]]
        Cleaned, deduplicated results.  Each dict contains:

        - **title** -- Page title.
        - **url** -- Page URL (used for deduplication).
        - **content** -- Relevant content snippet.
        - **score** -- Relevance score (if available).

    Raises
    ------
    RuntimeError
        If the Tavily API call fails after retries or times out.
    """
    current_settings = get_settings()
    tool = _get_search_tool()
    logger.info("Searching Tavily: query='%s'", query)

    try:
        raw: dict[str, Any] = tool.search(
        query=query,
        search_depth=current_settings.tavily_search_depth,  # type: ignore[arg-type]
        max_results=current_settings.tavily_max_results,
        )
    except Exception as exc:
        logger.exception("Tavily search failed for query='%s'", query)
        raise RuntimeError(f"Tavily search failed: {exc}") from exc

    results_raw: list[dict[str, Any]] = raw.get("results", [])

    if not results_raw:
        logger.warning("Tavily returned zero results for query='%s'", query)
        return []

    # ── Clean & deduplicate ─────────────────────────────────────
    seen_urls: set[str] = set()
    cleaned: list[dict[str, Any]] = []

    for item in results_raw:
        url: str = (item.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        cleaned.append(
            {key: item.get(key) for key in _KEEP_KEYS}
        )

    logger.info(
        "Tavily search complete: %d unique results from %d raw",
        len(cleaned),
        len(results_raw),
    )

    return cleaned
