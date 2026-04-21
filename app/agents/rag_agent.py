from observability import track
"""RAG agent node - generates answers using retrieved context."""

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


RAG_PROMPT = ChatPromptTemplate.from_template(
    "Sei un assistente AI esperto di Ricoh Italia e delle sue soluzioni enterprise.\n"
    "Rispondi in italiano in modo professionale e conciso.\n"
    "Usa il contesto fornito per rispondere. Se non hai informazioni sufficienti, dillo chiaramente.\n\n"
    "Contesto:\n{context}\n\n"
    "Domanda: {question}"
)


@track("rag_agent")
def rag_generate(state: AgentState) -> AgentState:
    result = (RAG_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {**state, "response": result.content}
