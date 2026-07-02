"""
LangGraph node: assemble all upstream outputs into a single
professional Markdown report.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from state import MarketState

logger = logging.getLogger(__name__)


def create_report(state: MarketState) -> dict[str, Any]:
    """Combine all research outputs into a final Markdown report.

    This node is **stateless with respect to the LLM** — it
    assembles content programmatically from the existing state
    fields.

    Parameters
    ----------
    state:
        The current graph state.  Reads ``parsed_query``,
        ``market_analysis``, ``insights``, ``recommendations``, and
        ``market_data``.

    Returns
    -------
    dict[str, Any]
        A partial state update with ``report`` and optionally
        ``errors``.
    """
    errors: list[str] = list(state.get("errors", []))

    parsed: dict[str, Any] = state.get("parsed_query", {}) or {}
    analysis: str = (state.get("market_analysis") or "").strip()
    insights: str = (state.get("insights") or "").strip()
    recommendations: str = (state.get("recommendations") or "").strip()
    market_data: list[dict[str, Any]] = state.get("market_data", [])

    industry: str = (parsed.get("industry") or "").strip() or "N/A"
    country: str = (parsed.get("country") or "").strip() or "Global"
    timestamp: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    logger.info("Assembling report: industry='%s', country='%s'", industry, country)

    sections: list[str] = []

    # ── Title ────────────────────────────────────────────────────
    sections.append(f"# Market Research Report: {industry}\n")
    sections.append(f"**Industry:** {industry}  \n")
    sections.append(f"**Country / Region:** {country}  \n")
    sections.append(f"**Generated:** {timestamp}  \n")

    # ── Executive Summary ────────────────────────────────────────
    sections.append("\n---\n\n## Executive Summary\n")
    first_paragraph = _first_heading_content(analysis, "Executive Summary")
    sections.append(first_paragraph if first_paragraph else "No executive summary available.\n")

    # ── Market Overview ──────────────────────────────────────────
    sections.append("\n## Market Overview\n")
    overview = _section_content(analysis, "Market Overview")
    sections.append(overview if overview else "Market overview not available.\n")

    # ── Market Analysis ──────────────────────────────────────────
    sections.append("\n## Market Analysis\n")
    sections.append(analysis if analysis else "Market analysis not available.\n")

    # ── Strategic Insights ───────────────────────────────────────
    sections.append("\n---\n\n## Strategic Insights\n")
    sections.append(insights if insights else "Strategic insights not available.\n")

    # ── Business Recommendations ─────────────────────────────────
    sections.append("\n---\n\n## Business Recommendations\n")
    sections.append(
        recommendations if recommendations
        else "Business recommendations not available.\n"
    )

    # ── Conclusion ───────────────────────────────────────────────
    sections.append("\n---\n\n## Conclusion\n")
    conclusion = _build_conclusion(analysis, insights, recommendations)
    sections.append(conclusion)

    # ── References ───────────────────────────────────────────────
    sections.append("\n---\n\n## References\n")
    refs = _build_references(market_data)
    sections.append(refs)

    report = "".join(sections)

    if not report.strip():
        errors.append("create_report: assembled report is empty")
        logger.warning("Assembled report is empty")

    logger.info("Report assembled (%d characters)", len(report))

    return {"report": report, "errors": errors}


# ── Helpers ──────────────────────────────────────────────────────


def _first_heading_content(text: str, heading: str) -> str:
    """Return the paragraph immediately after *heading* in *text*."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().lstrip("#").strip().startswith(heading.lower()):
            body: list[str] = []
            for candidate in lines[i + 1 :]:
                if candidate.strip().startswith("#"):
                    break
                if candidate.strip():
                    body.append(candidate)
            return " ".join(body).strip() if body else ""
    return ""


def _section_content(text: str, heading: str) -> str:
    """Return the content block under *heading* in *text*."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        normalized = stripped.lstrip("#").strip().lower()
        if normalized.startswith(heading.lower()):
            block: list[str] = []
            for candidate in lines[i + 1 :]:
                if candidate.strip().startswith("#") and candidate.strip().lstrip("#").strip():
                    break
                block.append(candidate)
            return "\n".join(block).strip()
    return ""


def _build_conclusion(
    analysis: str,
    insights: str,
    recommendations: str,
) -> str:
    """Generate a brief conclusion summarising coverage.

    Uses the presence of content rather than any LLM call.
    """
    parts: list[str] = [
        "This report provides a comprehensive analysis of the target market, "
        "including key data points, strategic insights, and actionable recommendations "
        "to guide business decision-making.\n"
    ]

    has_data = bool(analysis.strip())
    has_insights = bool(insights.strip())
    has_recs = bool(recommendations.strip())

    if has_data and has_insights and has_recs:
        parts.append(
            "The findings are based on web-sourced market data analysed through "
            "a multi-stage AI pipeline, offering a balanced view of opportunities, "
            "risks, and competitive dynamics."
        )
    elif not has_data:
        parts.append("Market data was unavailable for this report.")
    elif not has_insights:
        parts.append("Strategic insights could not be extracted from the available data.")
    elif not has_recs:
        parts.append("Recommendations were not generated from the available insights.")

    return " ".join(parts) + "\n"


def _build_references(market_data: list[dict[str, Any]]) -> str:
    """List every unique source URL from *market_data*."""
    seen: set[str] = set()
    refs: list[str] = []

    for item in market_data:
        url: str = (item.get("url") or "").strip()
        title: str = (item.get("title") or "Untitled").strip()
        if url and url not in seen:
            seen.add(url)
            refs.append(f"- [{title}]({url})")

    if not refs:
        return "No external sources were referenced.\n"

    return "\n".join(refs) + "\n"
