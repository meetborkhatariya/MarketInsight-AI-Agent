"""
LangGraph node: parse the user's natural-language query into structured fields.

Receives ``MarketState``, invokes the Groq LLM, and populates
``parsed_query`` with a JSON object containing *industry*, *country*,
*goal*, *report_type*, and *keywords*.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tools.groq_llm import get_llm
from state import MarketState

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = frozenset({"industry", "country", "goal", "report_type", "keywords"})

_SYSTEM_PROMPT = """\
You are a precise market research query parser.

Your task is to analyse the user's natural-language request and extract
structured information.  Return **only** a valid JSON object — no
markdown fences, no preamble, no explanation.

The JSON must use exactly these keys:

{
  "industry": "<target industry, e.g. \"electric vehicles\">",
  "country": "<target country or region, e.g. \"Southeast Asia\">",
  "goal": "<research objective, e.g. \"market size estimation\">",
  "report_type": "<type of report needed, e.g. \"competitive analysis\">",
  "keywords": ["<keyword 1>", "<keyword 2>", ...]
}

Rules:
- Infer missing fields from context rather than leaving them empty.
- ``keywords`` should contain 3-5 search-friendly terms.
- ``report_type`` must be one of:
  "market size", "competitive analysis", "trend analysis",
  "customer insights", "investment research", or "general overview".
- If no country is mentioned use "global".
- Output **only** the JSON object.  No surrounding text."""  # noqa: E501


def understand_query(state: MarketState) -> dict[str, Any]:
    """Convert the user's raw *query* into a structured ``parsed_query``.

    Parameters
    ----------
    state:
        The current graph state containing at minimum a ``query`` key.

    Returns
    -------
    dict[str, Any]
        A partial state update with ``parsed_query`` and optionally
        ``errors`` (if parsing failed).
    """
    query: str = state.get("query", "").strip()
    errors: list[str] = list(state.get("errors", []))

    if not query:
        errors.append("understand_query: received empty query")
        logger.warning("Empty query — skipping LLM call")
        return {"parsed_query": _empty_parsed(), "errors": errors}

    logger.info("Parsing query: '%s'", query)

    llm = get_llm()

    try:
        raw_response: str = llm.invoke(_SYSTEM_PROMPT + "\n\nUser query: " + query).content  # type: ignore[union-attr]  # noqa: E501
    except Exception as exc:
        logger.exception("LLM call failed for query='%s'", query)
        errors.append(f"understand_query: LLM invocation error — {exc}")
        return {"parsed_query": _empty_parsed(), "errors": errors}

    parsed = _try_parse_json(raw_response, errors)

    if not errors or errors == list(state.get("errors", [])):
        logger.info(
            "Query parsed successfully: industry='%s', country='%s'",
            parsed.get("industry"),
            parsed.get("country"),
        )
    else:
        logger.warning("Query parsing completed with %d error(s)", len(errors) - len(state.get("errors", [])))  # noqa: E501

    return {"parsed_query": parsed, "errors": errors}


# ── Helpers ──────────────────────────────────────────────────────


def _empty_parsed() -> dict[str, Any]:
    """Return a blank parsed query structure."""
    return {
        "industry": "",
        "country": "",
        "goal": "",
        "report_type": "",
        "keywords": [],
    }


def _try_parse_json(
    raw: str,
    errors: list[str],
) -> dict[str, Any]:
    """Attempt to parse *raw* as JSON, falling back to extraction.

    Appends error messages to *errors* on failure.  Always returns a
    dict (defaults to ``_empty_parsed()`` on total failure).
    """
    # Attempt 1 — direct parse.
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt 2 — extract from markdown code block.
        match = re.search(
            r"```(?:json)?\s*([\s\S]+?)```",
            text,
        )
        if match:
            try:
                data = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if not isinstance(data, dict):
        errors.append(
            "understand_query: LLM response is not valid JSON — "
            f"got type {type(data).__name__}"
        )
        logger.error("Failed to parse LLM response as JSON:\n%s", text[:500])
        return _empty_parsed()

    # Validate required keys.
    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        errors.append(
            f"understand_query: missing required key(s): {', '.join(sorted(missing))}"
        )
        logger.warning("Parsed JSON missing keys: %s", missing)

    # Ensure every key exists (fill missing with defaults).
    for key in _REQUIRED_KEYS:
        if key not in data:
            if key == "keywords":
                data[key] = []
            else:
                data[key] = ""

    # Sanity-check keywords is a list.
    if not isinstance(data.get("keywords"), list):
        data["keywords"] = []

    return data
