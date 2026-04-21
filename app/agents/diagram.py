"""Diagram agent - generates Mermaid.js architecture diagrams from prompts."""

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


DIAGRAM_PROMPT = ChatPromptTemplate.from_template(
    "Sei un Solutions Architect. Genera un diagramma Mermaid.js per il seguente scenario.\n\n"
    "Contesto dalla knowledge base Ricoh:\n{context}\n\n"
    "REGOLE:\n"
    "- Usa la sintassi Mermaid valida (flowchart, sequence, o graph)\n"
    "- Preferisci flowchart TD o LR per architetture\n"
    "- Usa emoji nei nodi per renderli visivi (🤖, 🗄️, 🌐, ☁️, 📄, 🔍, etc.)\n"
    "- Includi i servizi Azure dove rilevante\n"
    "- Il codice Mermaid DEVE essere in un blocco ```mermaid\n"
    "- Dopo il diagramma, aggiungi una breve spiegazione in italiano dei componenti\n\n"
    "Richiesta: {question}"
)


def diagram(state: AgentState) -> AgentState:
    result = (DIAGRAM_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {**state, "response": result.content}
