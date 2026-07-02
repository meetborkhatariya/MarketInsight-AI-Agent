"""
LangGraph node: produce actionable business recommendations grounded in
the extracted market insights.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.groq_llm import get_llm
from state import MarketState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior business strategist converting market insights into
concrete, actionable recommendations for the executive team.

Read the insights provided below and produce a **practical strategic
plan**.  Every recommendation must be directly supported by evidence
from the insights — do not invent facts.

Each recommendation must follow this exact format:

### Recommendation <N>
- **Recommendation:** <one-sentence action>
- **Reason:** <why this works, referencing the insight>
- **Expected Impact:** <quantified or qualitative outcome>
- **Priority:** High / Medium / Low

Cover every section below.  If the insights are insufficient for a
section, state "Limited evidence available" rather than fabricating.

---

## 1. Executive Recommendations
(3-5 high-level strategic priorities for leadership)

## 2. Market Entry Strategy
(Geographic or segment entry approach, mode of entry)

## 3. Product Strategy
(Features, positioning, differentiation, roadmap)

## 4. Pricing Strategy
(Pricing model, tiering, discounting approach)

## 5. Marketing Strategy
(Go-to-market, channels, messaging, brand)

## 6. Investment Strategy
(CAPEX, R&D, M&A, partnerships, funding needs)

## 7. Risk Mitigation Plan
(Key risks and specific mitigation actions)

## 8. Short-term Actions (0–6 months)
(Immediate wins and quick implementation)

## 9. Mid-term Actions (6–18 months)
(Building competitive advantage)

## 10. Long-term Strategy (18+ months)
(Market leadership and moat building)

Output in **professional Markdown**.
"""  # noqa: E501


def generate_recommendations(state: MarketState) -> dict[str, Any]:
    """Generate structured business recommendations from market insights.

    Parameters
    ----------
    state:
        The current graph state.  Expects ``insights`` to contain the
        distilled insights produced by the ``extract_insights`` node.

    Returns
    -------
    dict[str, Any]
        A partial state update with ``recommendations`` and optionally
        ``errors``.
    """
    insights: str = (state.get("insights") or "").strip()
    errors: list[str] = list(state.get("errors", []))

    if not insights:
        errors.append(
            "generate_recommendations: insights is empty — "
            "no recommendations can be generated"
        )
        logger.warning("Empty insights — skipping LLM call")
        return {
            "recommendations": (
                "## Executive Recommendations\n\n"
                "No insights were available to generate recommendations."
            ),
            "errors": errors,
        }

    logger.info(
        "Generating recommendations from insights (%d characters)",
        len(insights),
    )

    llm = get_llm()

    try:
        response = llm.invoke(_SYSTEM_PROMPT + "\n\n" + insights)
        recommendations: str = response.content  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("LLM recommendations call failed")
        errors.append(
            f"generate_recommendations: LLM invocation error — {exc}"
        )
        return {
            "recommendations": (
                "## Executive Recommendations\n\n"
                "Recommendations could not be generated due to an error."
            ),
            "errors": errors,
        }

    if not recommendations or not recommendations.strip():
        errors.append("generate_recommendations: LLM returned empty recommendations")
        logger.warning("LLM returned empty recommendations")
        recommendations = (
            "## Executive Recommendations\n\n"
            "Recommendations could not be generated."
        )

    logger.info(
        "Recommendations generated successfully (%d characters)",
        len(recommendations),
    )

    return {"recommendations": recommendations, "errors": errors}
