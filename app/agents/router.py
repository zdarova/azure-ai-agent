from observability import track
"""Router node - classifies query intent with reasoning."""

import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from agents import AgentState

_llm = None

VALID_ROUTES = {"rag", "summarize", "interview", "architecture", "compare", "diagram", "fallback"}


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
    "Classify this user question and explain your reasoning.\n"
    "Categories:\n"
    "- rag: factual questions about Ricoh, its products, services, people, or technology\n"
    "- summarize: requests to summarize, condense, or give an overview of a topic\n"
    "- interview: interview preparation, how to answer questions, coaching for job interviews\n"
    "- architecture: designing systems, proposing architectures, technical design questions\n"
    "- diagram: requests to draw, visualize, create a diagram, flowchart, or schema\n"
    "- compare: comparing two or more solutions, technologies, or approaches\n"
    "- fallback: greetings, off-topic, or questions unrelated to Ricoh\n\n"
    "Question: {question}\n"
    'Reply ONLY with JSON: {{"route": "<category>", "reasoning": "<one sentence in Italian explaining why>"}}'
)


@track("router")
def route(state: AgentState) -> AgentState:
    result = (ROUTER_PROMPT | _get_llm()).invoke({"question": state["question"]})
    raw = result.content.strip()

    try:
        parsed = json.loads(raw)
        category = parsed.get("route", "rag").lower()
        reasoning = parsed.get("reasoning", "")
    except Exception:
        category = raw.split('"route"')[-1].split('"')[1] if '"route"' in raw else "rag"
        reasoning = ""

    if category not in VALID_ROUTES:
        category = "rag"

    return {**state, "route": category, "reasoning": reasoning}
