"""
LangGraph workflow definition for MarketInsight AI.

Assembles the market research pipeline with conditional feedback
routing and checkpointing via ``MemorySaver``.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes.analyze_market import analyze_market
from nodes.collect_feedback import collect_feedback
from nodes.create_report import create_report
from nodes.extract_insights import extract_insights
from nodes.generate_recommendations import generate_recommendations
from nodes.improve_report import improve_report
from nodes.search_market_data import search_market_data
from nodes.understand_query import understand_query
from state import MarketState

logger = logging.getLogger(__name__)

# ── Graph topology ───────────────────────────────────────────────
#
#   START
#     ↓
#   understand_query
#     ↓
#   search_market_data
#     ↓
#   analyze_market
#     ↓
#   extract_insights
#     ↓
#   generate_recommendations
#     ↓
#   create_report
#     ↓
#   collect_feedback
#     ↓
#   ┌─ should_improve_report? ──┐
#   │                           │
#  YES                          NO
#   │                           │
#   ↓                           ↓
#   improve_report             END
#     ↓
#   END

_NODE_MAP: dict[str, Any] = {
    "understand_query": understand_query,
    "search_market_data": search_market_data,
    "analyze_market": analyze_market,
    "extract_insights": extract_insights,
    "generate_recommendations": generate_recommendations,
    "create_report": create_report,
    "collect_feedback": collect_feedback,
    "improve_report": improve_report,
}

_NODE_ORDER = tuple(_NODE_MAP.keys())


# ── Router ───────────────────────────────────────────────────────


def should_improve_report(state: MarketState) -> str:
    """Decide whether the report needs improvement.

    Reads ``feedback_analysis.needs_improvement`` from the state.

    Parameters
    ----------
    state:
        The current graph state after ``collect_feedback``.

    Returns
    -------
    str
        ``"improve_report"`` if improvement is needed, ``END``
        otherwise.
    """
    feedback_analysis: dict[str, Any] = state.get("feedback_analysis", {}) or {}
    needs_it: bool = feedback_analysis.get("needs_improvement", False)

    if needs_it:
        logger.info("Router → improve_report")
        return "improve_report"

    logger.info("Router → END (no improvement needed)")
    return END


# ── Graph builder ───────────────────────────────────────────────


def _build_graph() -> StateGraph:
    """Construct and return an uncompiled ``StateGraph``."""
    logger.info("Building MarketInsight workflow graph")

    graph = StateGraph(MarketState)

    for name, func in _NODE_MAP.items():
        graph.add_node(name, func)
        logger.debug("Registered node: %s", name)

    graph.set_entry_point("understand_query")

    # Wire the linear pipeline up to collect_feedback.
    linear_nodes = list(_NODE_MAP.keys())  # all 8 nodes
    for src, dst in zip(linear_nodes, linear_nodes[1:]):
        graph.add_edge(src, dst)

    # Replace the edge from collect_feedback → improve_report
    # with a conditional route.
    graph.add_conditional_edges(
        "collect_feedback",
        should_improve_report,
        {"improve_report": "improve_report", END: END},
    )

    # After improvement, the workflow ends.
    graph.add_edge("improve_report", END)

    logger.info(
        "Graph topology: %s → collect_feedback → should_improve_report? → (END | improve_report → END)",
        " → ".join(linear_nodes[:-2]),
    )

    return graph


# ── Compiled singleton ───────────────────────────────────────────

memory = MemorySaver()

market_research_graph = _build_graph().compile(checkpointer=memory)
"""Compiled LangGraph executable with ``MemorySaver`` checkpointing.

Usage::

    from graph import market_research_graph
    from state import MarketState

    config = {"configurable": {"thread_id": "user-session-1"}}

    initial: MarketState = {"query": "EV battery market in India"}
    result = market_research_graph.invoke(initial, config)
"""
