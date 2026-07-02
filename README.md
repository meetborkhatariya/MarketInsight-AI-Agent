# MarketInsight AI

Agentic AI-powered market research automation using LangGraph.

## Overview

MarketInsight AI is a production-grade multi-agent system that performs end-to-end market research. Given a natural language query, it searches the web, analyzes data, extracts insights, generates strategic recommendations, and produces a professional PDF report — all orchestrated through a LangGraph state machine.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ understand_query│  Refines and structures the raw query
└────────┬────────┘
         ▼
┌─────────────────┐
│search_market_data│ Searches Tavily for relevant market data
└────────┬────────┘
         ▼
┌─────────────────┐
│ analyze_market   │  Analyzes collected data using Groq LLM
└────────┬────────┘
         ▼
┌─────────────────┐
│ extract_insights │  Extracts key insights and patterns
└────────┬────────┘
         ▼
┌──────────────────────┐
│generate_recommendations│ Generates actionable recommendations
└──────────┬───────────┘
           ▼
┌─────────────────┐
│ create_report    │  Assembles and exports a PDF report
└────────┬────────┘
         ▼
┌─────────────────┐
│ collect_feedback │  Collects feedback for improvement
└────────┬────────┘
         ▼
   ┌─────────┐
   │ improve  │◄── feedback loop (max 3 iterations)
   │ _report  │
   └────┬────┘
        ▼
      END
```

### Tech Stack

| Component       | Technology    |
|-----------------|---------------|
| Orchestration   | LangGraph     |
| LLM             | Groq API      |
| Web Search      | Tavily API    |
| API Layer       | FastAPI       |
| Frontend        | Streamlit     |
| PDF Generation  | FPDF2         |
| Configuration   | Pydantic      |

## Project Structure

```
marketinsight-ai/
├── app.py                  # FastAPI entry point
├── streamlit_app.py        # Streamlit UI
├── graph.py                # LangGraph workflow
├── state.py                # Graph state schema
├── config.py               # API keys & settings
├── nodes/                  # LangGraph node functions
│   ├── understand_query.py
│   ├── search_market_data.py
│   ├── analyze_market.py
│   ├── extract_insights.py
│   ├── generate_recommendations.py
│   ├── create_report.py
│   ├── collect_feedback.py
│   └── improve_report.py
├── tools/                  # Tool wrappers
│   ├── groq_llm.py
│   ├── tavily_search.py
│   └── pdf_generator.py
├── prompts/                # LLM prompt templates
│   ├── analysis_prompt.py
│   ├── recommendation_prompt.py
│   └── report_prompt.py
├── reports/                # Generated PDF output
├── tests/                  # Test suite
├── .env.example
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.11+
- Groq API key
- Tavily API key

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd marketinsight-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### FastAPI Server

```bash
python app.py
# Server runs at http://localhost:8000
# Health check: http://localhost:8000/health
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
# UI opens at http://localhost:8501
```

## Running Tests

```bash
pytest tests/
```

## License

MIT
