"""
LangGraph node: refine a specific report section based on user
feedback without regenerating the entire document.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tools.groq_llm import get_llm
from state import MarketState

logger = logging.getLogger(__name__)

# Map feedback categories to the Markdown heading they target.
_SECTION_MAP: dict[str, str] = {
    "executive_summary": "Executive Summary",
    "competitor_analysis": "Competitive Landscape",
    "market_statistics": "Market Statistics",
    "government_policies": "Government Policies",
    "recommendations": "Business Recommendations",
    "pricing": "Pricing Strategy",
    "risks": "Risks",
    "opportunities": "Opportunities",
    "references": "References",
    "conclusion": "Conclusion",
}

# Headings whose content can be sourced from the raw analysis / insights
# fields rather than the assembled report.
_SOURCE_SECTIONS: dict[str, str] = {
    "Executive Summary": "market_analysis",
    "Competitive Landscape": "market_analysis",
    "Market Statistics": "market_analysis",
    "Government Policies": "market_analysis",
    "Risks": "market_analysis",
    "Opportunities": "market_analysis",
    "Business Recommendations": "recommendations",
}


def improve_report(state: MarketState) -> dict[str, Any]:
    """Rewrite the report section indicated by ``feedback_analysis``.

    Parameters
    ----------
    state:
        The current graph state.  Expects ``report``, ``feedback``,
        ``feedback_analysis``, and optionally ``market_analysis``,
        ``insights``, and ``recommendations``.

    Returns
    -------
    dict[str, Any]
        A partial state update with the improved ``report`` and
        optionally ``errors``.
    """
    report: str = (state.get("report") or "").strip()
    feedback_analysis: dict[str, Any] = state.get("feedback_analysis", {}) or {}
    category: str = (feedback_analysis.get("category") or "").strip()
    user_feedback: str = (feedback_analysis.get("user_feedback") or "").strip()
    errors: list[str] = list(state.get("errors", []))

    if not report:
        errors.append("improve_report: report is empty — nothing to improve")
        logger.warning("Empty report — skipping improvement")
        return {"report": report, "errors": errors}

    if not category or category == "other":
        return _handle_other(report, user_feedback, errors)

    logger.info(
        "Improving section '%s' based on feedback: %s",
        category,
        user_feedback[:120],
    )

    section_heading = _SECTION_MAP.get(category)
    if not section_heading:
        logger.warning("Unknown category '%s' — appending as other", category)
        return _handle_other(report, user_feedback, errors)

    current_section = extract_section(report, section_heading)
    if current_section is None:
        errors.append(
            f"improve_report: section '{section_heading}' not found in report"
        )
        logger.warning("Section '%s' not found — appending instead", section_heading)
        return _handle_other(report, user_feedback, errors)

    context = _build_context(state, category, section_heading)
    prompt = _build_prompt(section_heading, current_section, user_feedback, context)

    llm = get_llm()

    try:
        response = llm.invoke(prompt)
        new_section: str = response.content  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("LLM call failed during report improvement")
        errors.append(f"improve_report: LLM invocation error — {exc}")
        return {"report": report, "errors": errors}

    if not new_section or not new_section.strip():
        errors.append("improve_report: LLM returned empty section")
        logger.warning("LLM returned empty section — report unchanged")
        return {"report": report, "errors": errors}

    improved = replace_section(report, section_heading, new_section.strip())

    logger.info(
        "Section '%s' improved (%d chars)",
        section_heading,
        len(improved),
    )

    return {"report": improved, "errors": errors}


# ── Section helpers ─────────────────────────────────────────────


def extract_section(markdown: str, heading: str) -> str | None:
    """Return the content under *heading* in *markdown*, or ``None``.

    The content includes everything from the line after the heading up
    to (but not including) the next top-level or second-level heading.
    """
    lines = markdown.splitlines()
    target_pattern = heading.strip().lower()
    start_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip().lower()
        if stripped == target_pattern:
            start_idx = i
            break

    if start_idx is None:
        return None

    content_lines: list[str] = []
    for line in lines[start_idx + 1 :]:
        if line.strip().startswith("#") and not line.strip().startswith("###"):
            break
        if line.strip():
            content_lines.append(line)

    return "\n".join(content_lines).strip() if content_lines else ""


def replace_section(markdown: str, heading: str, new_content: str) -> str:
    """Replace the content under *heading* with *new_content*.

    The heading line itself is preserved; only the body below it is
    replaced.
    """
    lines = markdown.splitlines()
    target_pattern = heading.strip().lower()
    heading_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip().lower()
        if stripped == target_pattern:
            heading_idx = i
            break

    if heading_idx is None:
        return markdown

    # Find where this section ends (next ## heading or end).
    end_idx = len(lines)
    for i in range(heading_idx + 1, len(lines)):
        if lines[i].strip().startswith("#") and not lines[i].strip().startswith("###"):
            end_idx = i
            break

    # Rebuild: heading line + new content + everything after the section.
    before = lines[: heading_idx + 1]
    after = lines[end_idx:]
    result = "\n".join(before) + "\n\n" + new_content.strip() + "\n"
    if after:
        result += "\n" + "\n".join(after)
    return result.strip()


# ── Other category ──────────────────────────────────────────────


def _handle_other(
    report: str,
    feedback: str,
    errors: list[str],
) -> dict[str, Any]:
    """Append an *Additional Notes* section for unclassifiable feedback."""
    if not feedback:
        logger.info("No feedback for 'other' category — report unchanged")
        return {"report": report, "errors": errors}

    notes = (
        "\n\n---\n\n"
        "## Additional Notes\n\n"
        f"{feedback}\n"
    )
    logger.info("Appending Additional Notes section based on 'other' feedback")
    return {"report": report + notes, "errors": errors}


# ── Context & Prompt ────────────────────────────────────────────


def _build_context(
    state: MarketState,
    category: str,
    heading: str,
) -> str:
    """Gather supporting information from upstream fields."""
    parts: list[str] = []

    source_field = _SOURCE_SECTIONS.get(heading)
    if source_field:
        source_text = (state.get(source_field) or "").strip()
        if source_text:
            section = extract_section(source_text, heading)
            if section:
                parts.append(f"### Relevant data from {source_field}\n{section}")
            # If the heading isn't found as a subsection, include the
            # relevant source field text for broader context.

    # Always include the full insights for cross-reference.
    insights = (state.get("insights") or "").strip()
    if insights:
        parts.append(f"### Supporting insights\n{insights[:2000]}")

    return "\n\n".join(parts) if parts else "No additional context available."


def _build_prompt(
    heading: str,
    current_section: str,
    feedback: str,
    context: str,
) -> str:
    """Build the LLM prompt for rewriting a single section."""
    return (
        "You are a market report editor.  Revise the section below "
        "to address the user's feedback.\n\n"
        f"## Section to improve\n{heading}\n\n"
        f"## Current content\n{current_section}\n\n"
        f"## User feedback\n{feedback}\n\n"
        f"## Supporting context\n{context}\n\n"
        "## Instructions\n"
        f"- Rewrite **only** the content for the '{heading}' section.\n"
        "- Output the new section content only — no heading, no preamble.\n"
        "- Stay factual; do not fabricate data the context does not support.\n"
        "- If the feedback requests information not in the context, state that "
        "the available data is limited.\n"
        "- Preserve the same heading level and tone as the original."
    )
