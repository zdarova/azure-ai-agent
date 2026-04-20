"""Router node - classifies query intent and picks the right agent."""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from agents import AgentState

_llm = None

VALID_ROUTES = {"rag", "summarize", "interview", "architecture", "compare", "fallback"}


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
            api_key=os.environ["AZURE_AI_KEY"],
            base_url=os.environ["AZURE_AI_ENDPOINT"],
            temperature=0, max_tokens=20,
        )
    return _llm


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Classify this user question into one category.\n"
    "Categories:\n"
    "- rag: factual questions about Ricoh, its products, services, people, or technology\n"
    "- summarize: requests to summarize, condense, or give an overview of a topic\n"
    "- interview: interview preparation, how to answer questions, coaching for job interviews\n"
    "- architecture: designing systems, proposing architectures, technical design questions\n"
    "- compare: comparing two or more solutions, technologies, or approaches\n"
    "- fallback: greetings, off-topic, or questions unrelated to Ricoh\n\n"
    "Question: {question}\n"
    "Reply with ONLY one word: rag, summarize, interview, architecture, compare, or fallback."
)


def route(state: AgentState) -> AgentState:
    result = (ROUTER_PROMPT | _get_llm()).invoke({"question": state["question"]})
    category = result.content.strip().lower()
    if category not in VALID_ROUTES:
        category = "rag"
    return {**state, "route": category}
