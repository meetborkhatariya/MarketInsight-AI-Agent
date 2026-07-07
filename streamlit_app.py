"""
Streamlit frontend for MarketInsight AI.

Provides a chat-like interface for submitting market research queries
and viewing the generated report, analysis, insights, and
recommendations through the FastAPI backend.
"""

from __future__ import annotations

import logging
from datetime import datetime
import tempfile
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import streamlit as st

from tools.pdf_generator import PDFGenerator

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"
GENERATE_ENDPOINT = f"{API_BASE_URL}/generate-report"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

EXAMPLE_QUERIES = [
    "Analyze the Indian Electric Vehicle market for investment opportunities",
    "Market size and growth of cloud computing in Southeast Asia",
    "Competitive landscape of plant-based meat alternatives in North America",
    "AI in healthcare market trends and key players 2025",
    "Saudi Arabia tourism industry post-2030 vision",
]

TECH_STACK = {
    "Orchestration": "LangGraph",
    "LLM": "Groq (Llama 3.3 70B)",
    "Web Search": "Tavily API",
    "Backend": "FastAPI",
    "Frontend": "Streamlit",
    "PDF": "FPDF2",
}


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _download_button(filename: str, content: str) -> None:
    """Render a download button for the given Markdown content."""
    if content and content.strip():
        st.download_button(
            label=f"⬇ Download {filename}",
            data=content.encode("utf-8"),
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )


def _pdf_download_button(filename: str, content: str) -> None:
    """Render a download button for the given content as a PDF file."""
    if not content or not content.strip():
        return

    pdf_generator = PDFGenerator(output_dir="reports")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        pdf_path = pdf_generator.generate(content, str(temp_path))
        pdf_bytes = Path(pdf_path).read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)

    st.download_button(
        label=f"⬇ Download {filename}",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
    )


def _current_indian_time() -> str:
    """Return the current time formatted in Indian Standard Time."""
    return datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")


# ────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MarketInsight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## MarketInsight AI")
    st.markdown("*AI-Powered Market Research & Business Intelligence*")
    st.divider()

    st.markdown("### About")
    st.markdown(
        "An agentic AI system that performs end-to-end market research "
        "using a multi-stage LangGraph workflow. Submit a natural-language "
        "query and receive a structured analysis with insights and "
        "actionable recommendations."
    )
    st.divider()

    st.markdown("### Tech Stack")
    for label, value in TECH_STACK.items():
        st.markdown(f"- **{label}:** {value}")
    st.divider()

    st.markdown("### Example Queries")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
            st.session_state["query"] = q

    st.divider()
    st.caption("MarketInsight AI v1.0.0")
    st.caption(f"Current time: {_current_indian_time()}")


# ────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────

st.title("📊 MarketInsight AI")
st.markdown(
    "##### AI-Powered Market Research & Business Intelligence\n"
    "---"
)


# ────────────────────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────────────────────

query = st.text_area(
    "Enter your market research query",
    value=st.session_state.get("query", ""),
    placeholder="e.g., Analyze the Indian Electric Vehicle market for investment opportunities",
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    generate = st.button(
        "🚀 Generate Report",
        type="primary",
        use_container_width=True,
    )


# ────────────────────────────────────────────────────────────────
# API call
# ────────────────────────────────────────────────────────────────


def _call_api(query_text: str) -> dict[str, Any]:
    """Call the FastAPI backend and return the JSON response."""
    with st.spinner("Running market research workflow..."):
        try:
            resp = httpx.post(
                GENERATE_ENDPOINT,
                json={"query": query_text},
                timeout=300.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            st.error(
                f"Could not connect to the backend at `{API_BASE_URL}`. "
                "Ensure the FastAPI server is running."
            )
            logger.exception("Connection refused to %s", API_BASE_URL)
            return {}
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", str(exc))
            st.error(f"API error ({exc.response.status_code}): {detail}")
            logger.exception("API returned %s", exc.response.status_code)
            return {}
        except httpx.TimeoutException:
            st.error("Request timed out after 5 minutes. Try a more specific query.")
            logger.exception("Request timed out")
            return {}
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            logger.exception("Unexpected API error")
            return {}


# ────────────────────────────────────────────────────────────────
# Results
# ────────────────────────────────────────────────────────────────

query_text = (query or "").strip()

if generate and query_text:
    st.session_state["query"] = query_text
    data = _call_api(query_text)

    if data and data.get("success"):
        st.success("Report generated successfully!")

        report = data.get("report", "")
        analysis = data.get("analysis", "")
        insights = data.get("insights", "")
        recommendations = data.get("recommendations", "")
        errors = data.get("errors", [])

        if errors:
            with st.expander("⚠️ Warnings / Errors"):
                for err in errors:
                    st.warning(err)

        tabs = st.tabs(["📄 Report", "📊 Market Analysis", "💡 Insights", "🎯 Recommendations"])

        with tabs[0]:
            st.markdown(report or "*No report generated.*")
            _pdf_download_button("report.pdf", report)

        with tabs[1]:
            st.markdown(analysis or "*No analysis available.*")
            _pdf_download_button("market_analysis.pdf", analysis)

        with tabs[2]:
            st.markdown(insights or "*No insights available.*")
            _pdf_download_button("insights.pdf", insights)

        with tabs[3]:
            st.markdown(recommendations or "*No recommendations available.*")
            _pdf_download_button("recommendations.pdf", recommendations)

    elif data and not data.get("success"):
        st.error("Report generation completed with errors.")
        for err in data.get("errors", []):
            st.warning(err)
