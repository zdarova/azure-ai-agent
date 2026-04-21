from observability import track
"""Comparator agent - compares Ricoh solutions/technologies side by side."""

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
            temperature=0.2, max_tokens=2048,
        )
    return _llm


COMPARE_PROMPT = ChatPromptTemplate.from_template(
    "Sei un analista tecnico di Ricoh Italia.\n\n"
    "Contesto dalla knowledge base:\n{context}\n\n"
    "L'utente vuole un confronto. Crea una tabella comparativa in markdown con:\n"
    "- Righe: caratteristiche chiave (funzionalità, target, tecnologia, costi, pro/contro)\n"
    "- Colonne: gli elementi da confrontare\n\n"
    "Aggiungi una **Raccomandazione** finale con il caso d'uso ideale per ciascuno.\n\n"
    "Rispondi in italiano.\n\n"
    "Richiesta: {question}"
)


@track("comparator")
def compare(state: AgentState) -> AgentState:
    result = (COMPARE_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {**state, "response": result.content}
