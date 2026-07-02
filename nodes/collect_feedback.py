"""
LangGraph node: capture user feedback and classify it for the
improvement loop.
"""

from __future__ import annotations

import logging
from typing import Any

from state import MarketState

logger = logging.getLogger(__name__)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "competitor_analysis": (
        "competitor", "competition", "competitive", "market share",
        "rival", "player", "company", "vendor", "landscape",
    ),
    "market_statistics": (
        "statistic", "number", "data", "market size", "cagr",
        "growth rate", "revenue", "forecast", "figure", "percentage",
        "quantitative", "metric", "stat",
    ),
    "government_policies": (
        "government", "policy", "regulation", "regulatory", "tariff",
        "subsidy", "compliance", "law", "legal", "policy",
        "political", "trade", "import", "export",
    ),
    "recommendations": (
        "recommendation", "action", "strategy", "strategic",
        "suggestion", "advice", "next step", "roadmap",
    ),
    "pricing": (
        "price", "pricing", "cost", "revenue model", "margin",
        "profit", "monetization", "pricing strategy",
    ),
    "risks": (
        "risk", "threat", "challenge", "barrier", "downside",
        "uncertainty", "concern", "warning",
    ),
    "opportunities": (
        "opportunity", "opportunities", "potential", "growth area",
        "adjacent", "white space", "gap", "unmet",
    ),
    "references": (
        "reference", "source", "citation", "url", "link",
        "bibliography", "footnote", "cite", "attribution",
    ),
    "executive_summary": (
        "executive summary", "summary", "overview", "tl;dr",
        "high-level", "recap", "brief",
    ),
    "conclusion": (
        "conclusion", "closing", "final", "takeaway", "wrap-up",
        "summary section",
    ),
}

_CONFIDENCE_EXACT = 0.95
_CONFIDENCE_PARTIAL = 0.75
_CONFIDENCE_FALLBACK = 0.60


def collect_feedback(state: MarketState) -> dict[str, Any]:
    """Analyse user feedback and determine whether to improve the report.

    Classification is rule-based (keyword matching).  No LLM call is
    made.

    Parameters
    ----------
    state:
        The current graph state.  Expects an optional ``feedback``
        string.

    Returns
    -------
    dict[str, Any]
        A partial state update with ``feedback_analysis`` and
        optionally ``errors``.
    """
    feedback: str = (state.get("feedback") or "").strip()
    errors: list[str] = list(state.get("errors", []))

    if not feedback:
        logger.info("No feedback provided — improvement skipped")
        return {
            "feedback_analysis": {
                "needs_improvement": False,
                "category": None,
                "user_feedback": "",
                "confidence": 1.0,
            },
            "errors": errors,
        }

    logger.info("Analysing feedback (%d chars)", len(feedback))

    category, confidence = _classify(feedback)

    analysis: dict[str, Any] = {
        "needs_improvement": True,
        "category": category,
        "user_feedback": feedback,
        "confidence": confidence,
    }

    logger.info(
        "Feedback classified: category=%s, confidence=%.2f",
        category,
        confidence,
    )

    return {"feedback_analysis": analysis, "errors": errors}


# ── Classification ──────────────────────────────────────────────


def _classify(text: str) -> tuple[str, float]:
    """Return ``(category, confidence)`` for the given *text*.

    Matches against keyword lists with decreasing confidence levels.
    """
    lower = text.lower()

    best_category: str = "other"
    best_score: int = 0

    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_category = category

    if best_score == 0:
        return "other", _CONFIDENCE_FALLBACK

    confidence = (
        _CONFIDENCE_EXACT
        if best_score >= 3
        else _CONFIDENCE_PARTIAL
    )

    return best_category, confidence
