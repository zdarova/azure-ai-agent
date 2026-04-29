"""Router node - classifies query intent, supports multi-route for complex queries."""

import os
import json
from observability import track
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from agents import AgentState

_llm = None

VALID_ROUTES = {"rag", "summarize", "interview", "architecture", "compare", "diagram", "lineage", "web_search", "fallback"}


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
            api_key=os.environ["AZURE_AI_KEY"],
            base_url=os.environ["AZURE_AI_ENDPOINT"],
            temperature=0, max_tokens=150,
        )
    return _llm


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Classify this user question. You can select 1 to 3 agents if the question needs multiple perspectives.\n\n"
    "Available agents:\n"
    "- rag: factual questions about Ricoh, its products, services, people, or technology\n"
    "- summarize: requests to summarize, condense, or give an overview\n"
    "- interview: interview preparation, coaching, STAR method\n"
    "- architecture: designing systems, proposing architectures, technical design\n"
    "- diagram: draw, visualize, create a diagram, flowchart, or schema\n"
    "- compare: comparing two or more solutions or technologies\n"
    "- lineage: data lineage, data sources, pipeline runs, data provenance\n"
    "- web_search: questions requiring current/external information not in the knowledge base\n"
    "- fallback: greetings, off-topic, unrelated to Ricoh\n\n"
    "MULTI-ROUTE EXAMPLES:\n"
    "- 'Design a RAG system and show the diagram' -> [\"architecture\", \"diagram\"]\n"
    "- 'Compare pgvector vs Pinecone and summarize' -> [\"compare\", \"summarize\"]\n"
    "- 'How does the data pipeline work? Show lineage' -> [\"rag\", \"lineage\"]\n"
    "- 'What are the latest AI trends for Ricoh?' -> [\"web_search\", \"rag\"]\n\n"
    "Question: {question}\n"
    'Reply ONLY with JSON: {{"routes": ["agent1", "agent2"], "reasoning": "<one sentence in Italian>"}}'
)


@track("router")
def route(state: AgentState) -> AgentState:
    result = (ROUTER_PROMPT | _get_llm()).invoke({"question": state["question"]})
    raw = result.content.strip()

    try:
        parsed = json.loads(raw)
        routes = parsed.get("routes", ["rag"])
        reasoning = parsed.get("reasoning", "")
        # Handle old format
        if isinstance(routes, str):
            routes = [routes]
    except Exception:
        # Fallback: try to extract from raw
        routes = ["rag"]
        reasoning = ""

    # Validate routes
    routes = [r for r in routes if r in VALID_ROUTES]
    if not routes:
        routes = ["rag"]

    # Max 3 routes
    routes = routes[:3]

    return {**state, "routes": routes, "route": routes[0], "reasoning": reasoning}
