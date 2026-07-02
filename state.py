"""
State definition for the MarketInsight LangGraph workflow.

Uses ``typing.TypedDict`` for lightweight, LangGraph-compatible state
serialisation without external model dependencies.
"""

from __future__ import annotations

from typing import Any, TypedDict


class MarketState(TypedDict, total=False):
    """Graph state propagated through the market research pipeline.

    All fields except ``query`` are optional at construction time and
    are populated incrementally as the graph executes.
    """

    # ── Input ────────────────────────────────────────────────────

    query: str
    """Raw natural-language query from the user (required, set at init)."""

    # ── Processing ───────────────────────────────────────────────

    parsed_query: dict[str, Any]
    """Structured representation of the user's intent (target market,
    geography, timeframe, etc.)."""

    market_data: list[dict[str, Any]]
    """Raw search results collected from Tavily."""

    market_analysis: str
    """LLM-generated analysis of the gathered market data."""

    insights: str
    """Key insights and patterns extracted from the analysis."""

    recommendations: str
    """Actionable strategic recommendations derived from insights."""

    # ── Output ───────────────────────────────────────────────────

    report: str
    """Full text content of the final market research report."""

    report_path: str | None
    """Filesystem path to the exported PDF report (set after PDF generation)."""

    pdf_path: str | None
    """Alias for ``report_path``; kept for backward compatibility."""

    # ── Control Flow ─────────────────────────────────────────────

    feedback: str
    """User or simulated feedback used in the iterative improvement loop."""

    errors: list[str]
    """Accumulated error messages from any node that failed during execution."""
