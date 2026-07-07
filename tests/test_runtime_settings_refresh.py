from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_get_settings_reads_updated_environment(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("TAVILY_API_KEY", "key-two")

    from config import get_settings

    first = get_settings()

    monkeypatch.setenv("GROQ_API_KEY", "key-three")
    monkeypatch.setenv("TAVILY_API_KEY", "key-four")

    second = get_settings()

    assert first.groq_api_key.get_secret_value() == "key-one"
    assert first.tavily_api_key.get_secret_value() == "key-two"
    assert second.groq_api_key.get_secret_value() == "key-three"
    assert second.tavily_api_key.get_secret_value() == "key-four"


def test_get_llm_recreates_client_when_api_key_changes(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class DummyChatGroq:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setenv("GROQ_API_KEY", "alpha-real-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-real-key")

    langchain_groq = types.ModuleType("langchain_groq")
    langchain_groq.ChatGroq = DummyChatGroq
    monkeypatch.setitem(sys.modules, "langchain_groq", langchain_groq)

    groq_llm = _load_module("tests.groq_llm_under_test", "tools/groq_llm.py")

    monkeypatch.setattr(groq_llm, "ChatGroq", DummyChatGroq)
    monkeypatch.setattr(groq_llm, "_llm_instance", None)
    monkeypatch.setattr(groq_llm, "_llm_cache_key", None)

    first = groq_llm.get_llm()
    second = groq_llm.get_llm()

    monkeypatch.setenv("GROQ_API_KEY", "beta-real-key")
    third = groq_llm.get_llm()

    assert first is second
    assert third is not first
    assert len(created) == 2
    assert created[0]["api_key"] == "alpha-real-key"
    assert created[1]["api_key"] == "beta-real-key"


def test_search_client_recreates_when_api_key_changes(monkeypatch) -> None:
    created: list[str] = []

    class DummyTavilyClient:
        def __init__(self, *, api_key: str) -> None:
            created.append(api_key)
            self.api_key = api_key

        def search(self, **kwargs: object) -> dict[str, object]:
            return {"results": []}

    monkeypatch.setenv("GROQ_API_KEY", "groq-real-key")
    monkeypatch.setenv("TAVILY_API_KEY", "alpha-real-key")

    tavily_module = types.ModuleType("tavily")
    tavily_module.TavilyClient = DummyTavilyClient
    monkeypatch.setitem(sys.modules, "tavily", tavily_module)

    tavily_search = _load_module("tests.tavily_search_under_test", "tools/tavily_search.py")

    monkeypatch.setattr(tavily_search, "TavilyClient", DummyTavilyClient)
    monkeypatch.setattr(tavily_search, "_search_tool", None)
    monkeypatch.setattr(tavily_search, "_search_api_key", None)

    first = tavily_search._get_search_tool()
    second = tavily_search._get_search_tool()

    monkeypatch.setenv("TAVILY_API_KEY", "beta-real-key")
    third = tavily_search._get_search_tool()

    assert first is second
    assert third is not first
    assert created == ["alpha-real-key", "beta-real-key"]