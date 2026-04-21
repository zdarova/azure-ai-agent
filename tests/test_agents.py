"""Tests for the LangGraph multi-agent system."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.runnables import RunnableLambda
from agents import AgentState
from agents.router import VALID_ROUTES


def _mock_env():
    return patch.dict("os.environ", {
        "AZURE_AI_CHAT_DEPLOYMENT": "test",
        "AZURE_AI_KEY": "test",
        "AZURE_AI_ENDPOINT": "https://test",
        "AZURE_OPENAI_ENDPOINT": "https://test",
        "AZURE_OPENAI_KEY": "test",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "test",
        "PG_CONNECTION_STRING": "host=localhost port=5432 dbname=test user=test password=test sslmode=require",
    })


def _base_state(**overrides) -> AgentState:
    return {"question": "test", "context": "", "routes": ["rag"], "route": "rag", "reasoning": "", "response": "", "quality": None, "session_id": "test-session", **overrides}


def _fake_llm(content: str):
    return RunnableLambda(lambda x: MagicMock(content=content))


# --- State ---

def test_state_keys():
    assert set(_base_state().keys()) == {"question", "context", "routes", "route", "reasoning", "response", "quality", "session_id"}


def test_valid_routes():
    assert VALID_ROUTES == {"rag", "summarize", "interview", "architecture", "compare", "diagram", "lineage", "web_search", "fallback"}


# --- Router ---

@pytest.mark.parametrize("category", list(VALID_ROUTES))
def test_router_accepts_all_valid_routes(category):
    fake = _fake_llm('{"routes": ["' + category + '"], "reasoning": "test reason"}')
    with patch("agents.router._get_llm", return_value=fake):
        from agents.router import route
        result = route(_base_state())
        assert result["routes"] == [category]
        assert result["route"] == category
        assert result["reasoning"] == "test reason"


def test_router_defaults_invalid_to_rag():
    fake = _fake_llm('{"routes": ["garbage"], "reasoning": "x"}')
    with patch("agents.router._get_llm", return_value=fake):
        from agents.router import route
        result = route(_base_state())
        assert result["routes"] == ["rag"]
        assert result["route"] == "rag"


def test_router_multi_route():
    fake = _fake_llm('{"routes": ["architecture", "diagram"], "reasoning": "needs both"}')
    with patch("agents.router._get_llm", return_value=fake):
        from agents.router import route
        result = route(_base_state())
        assert result["routes"] == ["architecture", "diagram"]
        assert result["route"] == "architecture"
        assert result["reasoning"] == "needs both"


def test_router_multi_route_max_3():
    fake = _fake_llm('{"routes": ["rag", "architecture", "diagram", "compare"], "reasoning": "too many"}')
    with patch("agents.router._get_llm", return_value=fake):
        from agents.router import route
        result = route(_base_state())
        assert len(result["routes"]) == 3


def test_router_handles_old_format():
    fake = _fake_llm('{"routes": "rag", "reasoning": "single string"}')
    with patch("agents.router._get_llm", return_value=fake):
        from agents.router import route
        result = route(_base_state())
        assert result["routes"] == ["rag"]
        assert result["route"] == "rag"


def test_router_handles_malformed_json():
    fake = _fake_llm('not json at all')
    with patch("agents.router._get_llm", return_value=fake):
        from agents.router import route
        result = route(_base_state())
        assert result["routes"] == ["rag"]


# --- Graph ---

def test_graph_builds():
    with _mock_env():
        from graph import build_graph
        assert build_graph() is not None


def test_graph_has_all_nodes():
    with _mock_env():
        from graph import build_graph
        nodes = set(build_graph().get_graph().nodes.keys())
        for n in ("router", "retrieve", "rag", "summarize", "fallback", "interview", "architecture", "compare", "diagram", "lineage", "web_search"):
            assert n in nodes, f"Missing node: {n}"


# --- Agent nodes ---

def _test_agent(module_name, func_name, expected, state_overrides=None):
    with patch(f"{module_name}._get_llm", return_value=_fake_llm(expected)):
        import importlib
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name)
        result = fn(_base_state(**(state_overrides or {})))
        assert result["response"] == expected


def test_rag_agent():
    _test_agent("agents.rag_agent", "rag_generate", "Ricoh ha sede a Vimodrone", {"context": "Vimodrone"})


def test_fallback_agent():
    _test_agent("agents.fallback", "fallback", "Ciao! Sono l'assistente Ricoh.", {"route": "fallback"})


def test_interview_coach():
    _test_agent("agents.interview_coach", "interview_coach", "Usa il metodo STAR", {"route": "interview", "context": "RAG"})


def test_architect():
    _test_agent("agents.architect", "architecture_advisor", "Container Apps", {"route": "architecture", "context": "Azure"})


def test_comparator():
    _test_agent("agents.comparator", "compare", "Feature A vs B", {"route": "compare", "context": "info"})


def test_summarizer():
    _test_agent("agents.summarizer", "summarize", "Ricoh è leader", {"route": "summarize", "context": "Ricoh info"})


def test_quality_checker():
    fake = _fake_llm('{"relevance": 5, "accuracy": 4, "completeness": 4, "clarity": 5, "overall": 4, "note": "Risposta completa"}')
    with patch("agents.quality_checker._get_llm", return_value=fake):
        from agents.quality_checker import quality_check
        result = quality_check(_base_state(response="test response", context="ctx"))
        assert result["quality"]["overall"] == 4


def test_diagram_agent():
    _test_agent("agents.diagram", "diagram", "```mermaid\ngraph TD\nA-->B\n```", {"route": "diagram", "context": "Ricoh"})


def test_web_search_agent():
    fake_results = [{"title": "Ricoh", "body": "Ricoh info", "href": "https://ricoh.it"}]
    with patch("agents.web_search._search_ddg", return_value=fake_results):
        with patch("agents.web_search._get_llm", return_value=_fake_llm("Ricoh è un'azienda leader")):
            from agents.web_search import web_search
            result = web_search(_base_state(route="web_search", context="kb context"))
            assert "Ricoh" in result["response"]
            assert "Fonti web" in result["response"]


def test_web_search_no_results():
    with patch("agents.web_search._search_ddg", return_value=[]):
        with patch("agents.web_search._get_llm", return_value=_fake_llm("Nessun risultato trovato")):
            from agents.web_search import web_search
            result = web_search(_base_state(route="web_search"))
            assert result["response"] == "Nessun risultato trovato"


# --- Multi-route integration ---

def test_multi_route_state_isolation():
    """Each specialist should not mutate the shared state."""
    state = _base_state(context="shared context", routes=["architecture", "diagram"])
    responses = []
    for agent_route in state["routes"]:
        state["route"] = agent_route
        fake_state = {**state, "response": f"response from {agent_route}"}
        responses.append(fake_state["response"])
    assert state["response"] == ""  # original state untouched
    assert len(responses) == 2
    assert "architecture" in responses[0]
    assert "diagram" in responses[1]
