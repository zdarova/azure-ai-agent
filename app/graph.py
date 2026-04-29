"""LangGraph multi-agent graph with Send fan-out for multi-route support."""

import threading
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from agents import AgentState
from agents.router import route
from agents.retriever import retrieve
from agents.rag_agent import rag_generate
from agents.summarizer import summarize
from agents.fallback import fallback
from agents.interview_coach import interview_coach
from agents.architect import architecture_advisor
from agents.comparator import compare
from agents.diagram import diagram
from agents.lineage_agent import lineage_query
from agents.web_search import web_search
from agents.quality_checker import quality_check
from guardrails import check_input
from memory import save_turn, get_history
from longterm_memory import get_memories, extract_facts, save_memories

RETRIEVAL_ROUTES = {"rag", "summarize", "interview", "architecture", "compare", "diagram", "web_search"}

SPECIALISTS = {
    "rag": rag_generate,
    "summarize": summarize,
    "interview": interview_coach,
    "architecture": architecture_advisor,
    "compare": compare,
    "diagram": diagram,
    "lineage": lineage_query,
    "web_search": web_search,
    "fallback": fallback,
}


# --- Node functions ---

def guardrails_node(state: AgentState) -> AgentState:
    """Check input safety and PII, enrich question with history + memories."""
    guard = check_input(state["question"])
    if not guard["safe"]:
        return {
            "response": f"🛡️ {guard['reason']}",
            "routes": ["__blocked__"],
            "pii_detected": guard["pii_detected"],
        }

    # Enrich question with conversation history (keep concise)
    history = get_history(state["session_id"], limit=2)
    history_ctx = "\n".join(
        f"User: {h.get('question','')}\nAssistant: {h.get('response','')[:100]}"
        for h in history[-2:] if h.get('question')
    ) if history else ""

    question = state["question"]
    if history_ctx:
        question = f"[Conversazione precedente:\n{history_ctx}]\n\nNuova domanda: {question}"

    memories = get_memories(state["session_id"])
    if memories:
        question = f"[{memories}]\n\n{question}"

    return {"question": question, "pii_detected": guard["pii_detected"]}


def after_guardrails(state: AgentState) -> str:
    """Route to router or END if blocked."""
    if state.get("routes") == ["__blocked__"]:
        return END
    return "router"


def after_router(state: AgentState) -> str:
    """Decide whether retrieval is needed."""
    routes = state.get("routes", ["rag"])
    if any(r in RETRIEVAL_ROUTES for r in routes):
        return "retrieve"
    return "fan_out"


def fan_out_node(state: AgentState):
    """Fan-out: send state to each specialist via Send API."""
    routes = state.get("routes", ["rag"])
    return [Send("specialist", {**state, "route": r, "agent_responses": []}) for r in routes]


def specialist_node(state: AgentState) -> AgentState:
    """Execute the specialist for the current route."""
    agent_fn = SPECIALISTS.get(state["route"], rag_generate)
    result = agent_fn(state)
    return {
        "agent_responses": [{
            "agent": state["route"],
            "text": result["response"],
        }],
    }


def merge_node(state: AgentState) -> AgentState:
    """Merge all specialist responses into a single response."""
    parts = [ar["text"] for ar in state.get("agent_responses", [])]
    return {"response": "\n\n---\n\n".join(parts)}


def memory_node(state: AgentState) -> AgentState:
    """Extract and save long-term memory facts + persist turn — fire-and-forget."""
    def _background():
        try:
            routes = state.get("routes", ["rag"])
            facts = extract_facts(state["question"], state["response"], ",".join(routes))
            if facts:
                save_memories(state["session_id"], facts)
        except Exception:
            pass
        # Also persist turn here (was a separate node)
        if not state.get("pii_detected"):
            try:
                routes = state.get("routes", ["rag"])
                save_turn(
                    session_id=state["session_id"],
                    question=state["question"],
                    route=",".join(routes),
                    reasoning=state.get("reasoning", ""),
                    response=state["response"],
                    quality=state.get("quality"),
                )
            except Exception:
                pass
    threading.Thread(target=_background, daemon=True).start()
    return {}


def persist_node(state: AgentState) -> AgentState:
    """No-op — persist is now handled in memory_node background thread."""
    return {}


# --- Build graph ---

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guardrails", guardrails_node)
    g.add_node("router", route)
    g.add_node("retrieve", retrieve)
    g.add_node("fan_out", lambda state: {})  # passthrough, edges do the work
    g.add_node("specialist", specialist_node)
    g.add_node("merge", merge_node)
    g.add_node("quality_check", quality_check)
    g.add_node("memory", memory_node)
    g.add_node("persist", persist_node)

    g.add_edge(START, "guardrails")
    g.add_conditional_edges("guardrails", after_guardrails, {"router": "router", END: END})
    g.add_conditional_edges("router", after_router, {"retrieve": "retrieve", "fan_out": "fan_out"})
    g.add_edge("retrieve", "fan_out")
    g.add_conditional_edges("fan_out", fan_out_node)
    g.add_edge("specialist", "merge")
    g.add_edge("merge", "quality_check")
    g.add_edge("quality_check", "memory")
    g.add_edge("memory", "persist")
    g.add_edge("persist", END)

    return g.compile()
