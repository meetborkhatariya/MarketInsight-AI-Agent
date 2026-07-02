# 📊 MarketInsight AI

> **AI-Powered Agentic Market Research & Business Intelligence
> Platform**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-green)
![LangChain](https://img.shields.io/badge/LangChain-Framework-success)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Overview

MarketInsight AI is an **Agentic AI application** that automates
end-to-end market research using **LangGraph**, **LangChain**, **Groq
LLM**, and **Tavily Search**.

Instead of relying on a single LLM call, the application executes a
stateful workflow that: - Understands the user's query - Searches
real-time market data - Analyzes trends - Extracts strategic insights -
Generates business recommendations - Produces a professional report -
Supports human-in-the-loop feedback refinement

------------------------------------------------------------------------

## ✨ Features

-   Multi-step Agentic AI workflow
-   Real-time web search with Tavily
-   Market trend analysis using Groq LLM
-   SWOT & strategic insights
-   Business recommendations
-   Markdown & PDF report generation
-   Feedback-driven report improvement
-   FastAPI REST API
-   Streamlit UI
-   Modular architecture

------------------------------------------------------------------------

## 🏗️ LangGraph Workflow

``` text
START
  │
  ▼
Understand Query
  │
  ▼
Search Market Data
  │
  ▼
Analyze Market
  │
  ▼
Extract Insights
  │
  ▼
Generate Recommendations
  │
  ▼
Create Report
  │
  ▼
Collect Feedback
  │
  ▼
Improve Report
  │
  ▼
END
```

------------------------------------------------------------------------

## 🧠 Workflow Nodes

  Node                       Responsibility
  -------------------------- -----------------------------------------------
  Understand Query           Extracts industry, country, goal and keywords
  Search Market Data         Retrieves live information using Tavily
  Analyze Market             Synthesizes trends and opportunities
  Extract Insights           Produces SWOT and executive insights
  Generate Recommendations   Creates business strategies
  Create Report              Builds Markdown/PDF report
  Collect Feedback           Captures user feedback
  Improve Report             Updates only requested sections

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Component         Technology
  ----------------- -------------
  Agent Framework   LangGraph
  LLM Framework     LangChain
  LLM               Groq
  Search            Tavily API
  Backend           FastAPI
  Frontend          Streamlit
  PDF               FPDF2
  Validation        Pydantic
  Language          Python 3.11

------------------------------------------------------------------------

## 📂 Project Structure

``` text
marketinsight-ai/
├── app.py
├── streamlit_app.py
├── graph.py
├── state.py
├── config.py
├── nodes/
├── tools/
├── reports/
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone https://github.com/meetborkhatariya/MarketInsight-AI-Agent.git
cd MarketInsight-AI-Agent

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

``` env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Run the backend:

``` bash
python -m uvicorn app:app --reload
```

Run the frontend:

``` bash
streamlit run streamlit_app.py
```

------------------------------------------------------------------------

## 📋 Example Query

``` text
Analyze the Indian Electric Vehicle market for investment opportunities.
```

------------------------------------------------------------------------

## 📄 Sample Report

The generated report includes:

-   Executive Summary
-   Market Overview
-   Market Analysis
-   SWOT Analysis
-   Opportunities
-   Risks
-   Business Recommendations
-   References

------------------------------------------------------------------------

## 📸 Screenshots

Add screenshots after uploading them:

``` text
screenshots/
├── home.png
├── report.png
├── api_docs.png
```

------------------------------------------------------------------------

## 🌐 API Endpoints

  Method   Endpoint           Description
  -------- ------------------ ---------------------------------
  GET      /health            Service health
  POST     /generate-report   Generate market research report

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Authentication
-   Report history
-   Interactive charts
-   Multi-agent collaboration
-   Cloud deployment
-   Database integration

------------------------------------------------------------------------

## 👨‍💻 Author

**Meet Borkhatariya**

-   GitHub: https://github.com/meetborkhatariya
-   LinkedIn: https://www.linkedin.com/in/meetborkhatariya/

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.
