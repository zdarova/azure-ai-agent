from observability import track
"""Interview Coach agent - helps prepare structured answers for Ricoh interviews."""

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
            temperature=0.4, max_tokens=1024,
        )
    return _llm


COACH_PROMPT = ChatPromptTemplate.from_template(
    "Sei un coach per colloqui tecnici AI/ML. L'utente si prepara per un colloquio "
    "in Ricoh Italia (Npo Sistemi) per il ruolo di Senior AI/ML Engineer.\n\n"
    "Contesto:\n{context}\n\n"
    "Fornisci una risposta CONCISA con:\n"
    "1. **Strategia** - come strutturare la risposta (STAR se applicabile)\n"
    "2. **Risposta esempio** - concreta, max 150 parole\n"
    "3. **Cosa evitare** - 2-3 errori comuni\n"
    "4. **Bonus tip** - una frase per impressionare\n\n"
    "Rispondi in italiano, sii diretto e conciso.\n\n"
    "Domanda: {question}"
)


@track("interview_coach")
def interview_coach(state: AgentState) -> AgentState:
    result = (COACH_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {**state, "response": result.content}
