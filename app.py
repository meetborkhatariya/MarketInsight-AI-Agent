"""
FastAPI application for MarketInsight AI.

Exposes the LangGraph market research workflow as a REST API with
request validation, structured responses, and OpenAPI documentation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from graph import market_research_graph
from state import MarketState

import asyncio
import uuid

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Application
# ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MarketInsight AI",
    description=(
        "Agentic AI for automated market research using LangGraph. "
        "Submit a natural-language query and receive a structured "
        "market analysis report with insights and recommendations."
    ),
    version="1.0.0",
    contact={"name": "MarketInsight AI Team"},
)

# ── Middleware ───────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> JSONResponse:
    """Log every request with method, path, duration, and status."""
    start = time.perf_counter()
    response: Any = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info(
        "%s %s → %s (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ────────────────────────────────────────────────────────────────
# Schemas
# ────────────────────────────────────────────────────────────────


class GenerateReportRequest(BaseModel):
    """Request body for the ``POST /generate-report`` endpoint."""

    query: str


class GenerateReportResponse(BaseModel):
    """Response body returned after a successful report generation."""

    success: bool
    report: str
    analysis: str
    insights: str
    recommendations: str
    errors: list[str]


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint."""

    status: str
    service: str
    version: str


# ────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service status and version information."""
    return HealthResponse(
        status="healthy",
        service="MarketInsight AI",
        version="1.0.0",
    )


@app.post(
    "/generate-report",
    response_model=GenerateReportResponse,
    responses={
        200: {"description": "Report generated successfully"},
        422: {"description": "Validation error (e.g. empty query)"},
        500: {"description": "Internal server error"},
    },
)
async def generate_report(body: GenerateReportRequest) -> GenerateReportResponse:
    """Run the full market research workflow and return the results.

    Accepts a natural-language market research query, runs it through
    the LangGraph pipeline, and returns the produced report, analysis,
    insights, recommendations, and any errors encountered.
    """
    query = body.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="query must be a non-empty string",
        )

    logger.info("Generating report for query='%s'", query)

    initial_state: MarketState = {"query": query}

    try:
        final_state: MarketState = await _run_graph(initial_state)
    except RuntimeError as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during graph execution")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        ) from exc

    report: str = (final_state.get("report") or "").strip()
    analysis: str = (final_state.get("market_analysis") or "").strip()
    insights: str = (final_state.get("insights") or "").strip()
    recommendations: str = (final_state.get("recommendations") or "").strip()
    errors: list[str] = final_state.get("errors", [])

    success = bool(report) and not any(
        "LLM invocation error" in e or "Graph execution" in e
        for e in errors
    )

    logger.info(
        "Report generation complete: success=%s, errors=%d",
        success,
        len(errors),
    )

    return GenerateReportResponse(
        success=success,
        report=report,
        analysis=analysis,
        insights=insights,
        recommendations=recommendations,
        errors=errors,
    )


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

async def _run_graph(state: MarketState) -> MarketState:
    """Execute the LangGraph workflow in a thread pool.

    Since the graph uses MemorySaver, every invocation must provide
    a unique thread_id.
    """
    import asyncio
    import uuid

    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(
        None,
        lambda: market_research_graph.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": str(uuid.uuid4())
                }
            },
        ),
    )

    return cast(MarketState, result)

# ────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
