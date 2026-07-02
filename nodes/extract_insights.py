"""
LangGraph node: distil the market analysis into concise executive
insights including SWOT, opportunities, risks, and future outlook.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.groq_llm import get_llm
from state import MarketState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a chief strategy officer distilling a market analysis into
sharp, actionable insights for executive decision-makers.

Your task is to read the market analysis provided below and produce a
**concise, insight-driven summary** — do *not* repeat the analysis
verbatim.

Cover every section listed below.  If the analysis lacks evidence for a
section, state that explicitly rather than fabricating.

Output in **professional Markdown** using this structure:

## Key Findings
(5-8 bullet points synthesising the most important takeaways)

## SWOT Analysis

### Strengths
(Internal factors that give an advantage)

### Weaknesses
(Internal limitations or gaps)

### Opportunities
(External factors to capitalise on)

### Threats
(External risks that could harm position)

## Top 5 Business Opportunities
(Ranked by potential impact; 1-2 sentences each)

## Top 5 Market Risks
(Ranked by severity; 1-2 sentences each)

## Investment Attractiveness
**Rating:** High / Medium / Low
(1-2 sentences justification)

## Recommended Target Audience
(Specific segments, industries, or personas to focus on)

## Key Market Drivers
(External forces pushing the market forward)

## Future Predictions
(3-5 forward-looking statements grounded in the analysis)

Be specific — reference numbers, segments, and companies mentioned in
the source.  If the source data is inconclusive, say so.
"""  # noqa: E501


def extract_insights(state: MarketState) -> dict[str, Any]:
    """Extract structured business insights from the market analysis.

    Parameters
    ----------
    state:
        The current graph state.  Expects ``market_analysis`` to
        contain the full Markdown analysis produced by the
        ``analyze_market`` node.

    Returns
    -------
    dict[str, Any]
        A partial state update with ``insights`` and optionally
        ``errors``.
    """
    analysis: str = (state.get("market_analysis") or "").strip()
    errors: list[str] = list(state.get("errors", []))

    if not analysis:
        errors.append("extract_insights: market_analysis is empty — no insights to extract")
        logger.warning("Empty market_analysis — skipping LLM call")
        return {
            "insights": "## Key Findings\n\nNo market analysis was available to extract insights from.",
            "errors": errors,
        }

    logger.info("Extracting insights from market analysis (%d characters)", len(analysis))

    llm = get_llm()

    try:
        response = llm.invoke(_SYSTEM_PROMPT + "\n\n" + analysis)
        insights: str = response.content  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("LLM insight extraction call failed")
        errors.append(f"extract_insights: LLM invocation error — {exc}")
        return {
            "insights": "## Key Findings\n\nInsights could not be generated due to an error.",
            "errors": errors,
        }

    if not insights or not insights.strip():
        errors.append("extract_insights: LLM returned empty insights")
        logger.warning("LLM returned empty insights")
        insights = "## Key Findings\n\nInsights could not be generated."

    logger.info("Insights extracted successfully (%d characters)", len(insights))

    return {"insights": insights, "errors": errors}
