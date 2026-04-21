from observability import track
"""Summarizer agent node - condenses retrieved context into a concise summary."""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from agents import AgentState

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
            api_key=os.environ["AZURE_AI_KEY"],
            base_url=os.environ["AZURE_AI_ENDPOINT"],
            temperature=0.3, max_tokens=2048,
        )
    return _llm


SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    "Riassumi in italiano in modo chiaro e conciso le seguenti informazioni su Ricoh, "
    "rispondendo alla richiesta dell'utente.\n\n"
    "Contesto:\n{context}\n\n"
    "Richiesta: {question}"
)


@track("summarizer")
def summarize(state: AgentState) -> AgentState:
    result = (SUMMARIZE_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {**state, "response": result.content}
