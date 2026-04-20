"""Fallback agent node - handles greetings and off-topic queries."""

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
            temperature=0.5, max_tokens=512,
        )
    return _llm


FALLBACK_PROMPT = ChatPromptTemplate.from_template(
    "Sei l'assistente AI di Ricoh Italia. L'utente ha fatto una domanda fuori tema "
    "o un saluto. Rispondi in modo cordiale in italiano, presentati brevemente e "
    "suggerisci cosa puoi fare (rispondere su soluzioni Ricoh, AI, servizi, persone).\n\n"
    "Messaggio utente: {question}"
)


def fallback(state: AgentState) -> AgentState:
    result = (FALLBACK_PROMPT | _get_llm()).invoke({"question": state["question"]})
    return {**state, "response": result.content}
