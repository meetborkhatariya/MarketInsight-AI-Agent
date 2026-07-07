"""
LangGraph node: produce a structured market analysis from collected
search results.

Reads ``market_data``, feeds the combined content to the Groq LLM,
and writes a professional Markdown analysis into ``market_analysis``.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.groq_llm import get_llm
from state import MarketState

logger = logging.getLogger(__name__)

# Keep the analysis prompt comfortably under the Groq TPM limit.
_MAX_SOURCES = 8
_MAX_CHARS_PER_SOURCE = 1_200
_MAX_CONTEXT_CHARS = 10_000

_SYSTEM_PROMPT = """\
You are a senior market research analyst.  Your task is to produce a
thorough, evidence-based market analysis from the web search results
provided below.

Analyse the collection of articles **holistically** — synthesise
findings across sources rather than summarising each article
individually.

Cover every section listed below.  If evidence for a particular claim
is weak, contradictory, or absent, **explicitly state that** instead of
fabricating data.  Use phrases like "limited evidence suggests …",
"sources disagree on …", or "no data was found for …".

Output your analysis in **professional Markdown** using the following
structure:

# Executive Summary
(3-5 sentences capturing the most important findings)

## Market Overview
(Broad landscape description — key players, segments, maturity)

## Current Market Size
(Estimated revenue / volume with year, cite the source)

## Growth Rate (CAGR)
(Reported or consensus growth rate + forecast period)

## Emerging Trends
(Bullet list of 3-5 trends with brief evidence)

## Opportunities
(Untapped segments, adjacencies, tailwinds)

## Risks
(Regulatory, technological, macroeconomic threats)

## Challenges
(Entry barriers, supply chain, fragmentation, etc.)

## Government Policies
(Relevant regulations, subsidies, tariffs — note if none found)

## Competitive Landscape
(Major players, market concentration, differentiation)

## Future Outlook
(5-10 year forward view based on cited data)

Be specific: include numbers, company names, and source URLs where
available.  Always prefer the most recent data.
"""  # noqa: E501

_HEADER = "# Market Research Report\n\n## Source Material\n\n"


def analyze_market(state: MarketState) -> dict[str, Any]:
    """Analyse ``market_data`` and write a structured market analysis.

    Parameters
    ----------
    state:
        The current graph state.  Expects ``market_data`` to contain a
        list of search-result dicts with ``content``, ``url``,
        ``title``, and ``score`` keys.

    Returns
    -------
    dict[str, Any]
        A partial state update with ``market_analysis`` and optionally
        ``errors``.
    """
    market_data: list[dict[str, Any]] = state.get("market_data", [])
    errors: list[str] = list(state.get("errors", []))

    if not market_data:
        errors.append("analyze_market: market_data is empty — no analysis possible")
        logger.warning("Empty market_data — skipping LLM analysis")
        return {
            "market_analysis": "# Market Research Report\n\nNo search results were available to analyse.",
            "errors": errors,
        }

    logger.info("Analysing %d market data records", len(market_data))

    context = _build_context(market_data)
    prompt = f"{_HEADER}\n\n{context}\n\n## Analysis"

    llm = get_llm()

    try:
        response = llm.invoke(_SYSTEM_PROMPT + "\n\n" + prompt)
        analysis: str = response.content  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("LLM analysis call failed")
        errors.append(f"analyze_market: LLM invocation error — {exc}")
        return {
            "market_analysis": "# Market Research Report\n\nAnalysis could not be completed due to an error.",
            "errors": errors,
        }

    if not analysis or not analysis.strip():
        errors.append("analyze_market: LLM returned empty analysis")
        logger.warning("LLM returned empty analysis")
        analysis = "# Market Research Report\n\nAnalysis could not be generated."

    logger.info(
        "Market analysis produced (%d characters)",
        len(analysis),
    )

    return {"market_analysis": analysis, "errors": errors}


# ── Helpers ──────────────────────────────────────────────────────


def _build_context(records: list[dict[str, Any]]) -> str:
    """Format search results into a consolidated text context.

    Results are sorted by score (descending) so the most relevant
    content appears first.  The combined text is truncated to a small
    fixed budget so the Groq request stays below the TPM limit.
    """
    sorted_records = sorted(
        records,
        key=lambda r: r.get("score") if isinstance(r.get("score"), (int, float)) else 0,
        reverse=True,
    )

    parts: list[str] = []
    total = 0

    for i, rec in enumerate(sorted_records[:_MAX_SOURCES], start=1):
        title = (rec.get("title") or "Untitled").strip()
        url = (rec.get("url") or "").strip()
        content = " ".join((rec.get("content") or "").split())
        if len(content) > _MAX_CHARS_PER_SOURCE:
            content = content[:_MAX_CHARS_PER_SOURCE].rstrip() + " ..."

        block = (
            f"### Source {i}: {title}\n"
            f"**URL:** {url}\n\n"
            f"{content}\n\n"
        )

        total += len(block)
        if total > _MAX_CONTEXT_CHARS:
            allowance = _MAX_CONTEXT_CHARS - (total - len(block))
            if allowance > 200:
                parts.append(block[:allowance] + "\n\n[Content truncated …]")
            break

        parts.append(block)

    return "".join(parts)
