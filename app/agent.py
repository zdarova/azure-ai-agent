"""Ricoh AI Agent - LangGraph multi-agent orchestration."""

from graph import build_graph

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class RicohAgent:
    def __init__(self):
        self.graph = get_graph()

    def run(self, query: str) -> str:
        result = self.graph.invoke({
            "question": query,
            "context": "",
            "route": "rag",
            "reasoning": "",
            "response": "",
            "quality": None,
            "session_id": "",
        })
        return result["response"]
