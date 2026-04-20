"""Tests for the LangGraph multi-agent system."""

import pytest
from unittest.mock import patch, MagicMock
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
    return {"question": "test", "context": "", "route": "rag", "response": "", **overrides}


# --- State ---

def test_state_keys():
    state = _base_state()
    assert set(state.keys()) == {"question", "context", "route", "response"}


def test_valid_routes():
    assert VALID_ROUTES == {"rag", "summarize", "interview", "architecture", "compare", "fallback"}


# --- Router ---

def test_router_defaults_invalid_to_rag():
    with patch("agents.router.ROUTER_PROMPT.__or__", return_value=MagicMock(
        invoke=lambda x: MagicMock(content="garbage")
    )), patch("agents.router._get_llm"):
        from agents.router import route
        assert route(_base_state())["route"] == "rag"


@pytest.mark.parametrize("category", ["rag", "summarize", "interview", "architecture", "compare", "fallback"])
def test_router_accepts_all_valid_routes(category):
    with patch("agents.router.ROUTER_PROMPT.__or__", return_value=MagicMock(
        invoke=lambda x: MagicMock(content=category)
    )), patch("agents.router._get_llm"):
        from agents.router import route
        assert route(_base_state())["route"] == category


# --- Graph ---

def test_graph_builds():
    with _mock_env():
        from graph import build_graph
        assert build_graph() is not None


def test_graph_has_all_nodes():
    with _mock_env():
        from graph import build_graph
        nodes = set(build_graph().get_graph().nodes.keys())
        for n in ("router", "retrieve", "rag", "summarize", "fallback", "interview", "architecture", "compare"):
            assert n in nodes


# --- Agent nodes ---

def _test_agent_node(module_path, prompt_attr, func_name, state_overrides, expected_in_response):
    with patch(f"{module_path}._get_llm"), \
         patch(f"{module_path}.{prompt_attr}.__or__", return_value=MagicMock(
             invoke=lambda x: MagicMock(content=expected_in_response)
         )):
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)
        result = fn(_base_state(**state_overrides))
        assert expected_in_response in result["response"]


def test_rag_agent():
    _test_agent_node("agents.rag_agent", "RAG_PROMPT", "rag_generate",
                     {"context": "Vimodrone"}, "Ricoh ha sede a Vimodrone")


def test_fallback_agent():
    _test_agent_node("agents.fallback", "FALLBACK_PROMPT", "fallback",
                     {"route": "fallback"}, "Ciao! Sono l'assistente Ricoh.")


def test_interview_coach():
    _test_agent_node("agents.interview_coach", "COACH_PROMPT", "interview_coach",
                     {"route": "interview", "context": "RAG"}, "Usa il metodo STAR")


def test_architect():
    _test_agent_node("agents.architect", "ARCH_PROMPT", "architecture_advisor",
                     {"route": "architecture", "context": "Azure"}, "Container Apps → pgvector")


def test_comparator():
    _test_agent_node("agents.comparator", "COMPARE_PROMPT", "compare",
                     {"route": "compare", "context": "info"}, "| Feature | A | B |")


def test_summarizer():
    _test_agent_node("agents.summarizer", "SUMMARIZE_PROMPT", "summarize",
                     {"route": "summarize", "context": "Ricoh info"}, "Ricoh è leader nel digital workplace")
