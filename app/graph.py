"""LangGraph multi-agent graph for Ricoh AI."""

from langgraph.graph import StateGraph, END
from agents import AgentState
from agents.router import route
from agents.retriever import retrieve
from agents.rag_agent import rag_generate
from agents.summarizer import summarize
from agents.fallback import fallback
from agents.interview_coach import interview_coach
from agents.architect import architecture_advisor
from agents.comparator import compare

RETRIEVAL_ROUTES = {"rag", "summarize", "interview", "architecture", "compare"}


def _pick_agent(state: AgentState) -> str:
    if state["route"] in RETRIEVAL_ROUTES:
        return "retrieve"
    return "fallback"


def _post_retrieve(state: AgentState) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", route)
    g.add_node("retrieve", retrieve)
    g.add_node("rag", rag_generate)
    g.add_node("summarize", summarize)
    g.add_node("fallback", fallback)
    g.add_node("interview", interview_coach)
    g.add_node("architecture", architecture_advisor)
    g.add_node("compare", compare)

    g.set_entry_point("router")
    g.add_conditional_edges("router", _pick_agent, {
        "retrieve": "retrieve",
        "fallback": "fallback",
    })

    g.add_conditional_edges("retrieve", _post_retrieve, {
        "rag": "rag",
        "summarize": "summarize",
        "interview": "interview",
        "architecture": "architecture",
        "compare": "compare",
    })

    # Specialists → END (quality check runs async outside the graph)
    for node in ("rag", "summarize", "fallback", "interview", "architecture", "compare"):
        g.add_edge(node, END)

    return g.compile()
