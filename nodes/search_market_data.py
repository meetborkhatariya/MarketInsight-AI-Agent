"""
LangGraph node: execute multiple targeted web searches and merge results.

Reads the ``parsed_query`` produced by ``understand_query``, builds
several search queries covering market size, competitors, trends,
policies, and news, then collects, deduplicates, and sorts the
combined results into ``market_data``.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.tavily_search import search_market_data as run_search
from state import MarketState

logger = logging.getLogger(__name__)

# Maximum number of merged results returned to the graph.
MAX_MERGED_RESULTS = 25

# Search query templates keyed by research angle.  Each tuple is
# (label, template) where {industry}, {country}, {goal} are
# interpolated from ``parsed_query``.
_SEARCH_QUERIES: list[tuple[str, str]] = [
    ("market_size", "{industry} market size {country} {goal} revenue"),
    ("growth", "{industry} CAGR growth rate {country} forecast"),
    ("competitors", "top {industry} companies competitors {country} market share"),
    ("policies", "{industry} government regulations policies {country}"),
    ("trends", "{industry} industry trends {country} {goal} 2025"),
    ("news", "{industry} {country} latest news investments partnerships"),
]


def search_market_data(state: MarketState) -> dict[str, Any]:
    """Run targeted market research searches and populate ``market_data``.

    Parameters
    ----------
    state:
        The current graph state.  Expects ``parsed_query`` with keys
        *industry*, *country*, *goal*, *report_type*, and *keywords*.

    Returns
    -------
    dict[str, Any]
        A partial state update with ``market_data`` and optionally
        ``errors`` (if any search failed).
    """
    parsed: dict[str, Any] = state.get("parsed_query", {}) or {}
    errors: list[str] = list(state.get("errors", []))

    industry: str = (parsed.get("industry") or "").strip()
    country: str = (parsed.get("country") or "").strip()

    if not industry:
        errors.append("search_market_data: parsed_query.industry is empty — skipping searches")  # noqa: E501
        logger.warning("Empty industry in parsed_query — no searches performed")
        return {"market_data": [], "errors": errors}

    logger.info(
        "Starting market search: industry='%s', country='%s'",
        industry,
        country,
    )

    all_results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    failed_queries: list[str] = []

    for label, template in _SEARCH_QUERIES:
        query = template.format(
            industry=industry,
            country=country or "global",
            goal=parsed.get("goal", "market research"),
        )

        logger.debug("Search [%s]: %s", label, query)

        try:
            results = run_search(query)
        except Exception as exc:
            logger.warning("Search [%s] failed: %s", label, exc)
            failed_queries.append(label)
            continue

        for item in results:
            url: str = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            item["_search_label"] = label
            all_results.append(item)

    if failed_queries:
        errors.append(
            f"search_market_data: {len(failed_queries)} search(s) failed — "
            f"{', '.join(failed_queries)}"
        )
        logger.warning("Failed search labels: %s", failed_queries)

    # Sort by relevance score (descending), moving None scores to end.
    all_results.sort(
        key=lambda x: x.get("score") if isinstance(x.get("score"), (int, float)) else -1,  # type: ignore[return-value]  # noqa: E501
        reverse=True,
    )

    limited = all_results[:MAX_MERGED_RESULTS]

    logger.info(
        "Market search complete: %d unique results from %d queries%s",
        len(limited),
        len(_SEARCH_QUERIES),
        f" ({len(failed_queries)} failed)" if failed_queries else "",
    )

    return {"market_data": limited, "errors": errors}
