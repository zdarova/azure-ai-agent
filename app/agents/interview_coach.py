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
            temperature=0.4, max_tokens=2048,
        )
    return _llm


COACH_PROMPT = ChatPromptTemplate.from_template(
    "Sei un coach per colloqui tecnici AI/ML. L'utente si sta preparando per un colloquio "
    "in Ricoh Italia per il ruolo di Senior AI/ML Engineer.\n\n"
    "Usa il contesto dalla knowledge base per personalizzare la risposta:\n{context}\n\n"
    "Per la seguente domanda di colloquio, fornisci:\n"
    "1. **Strategia di risposta** - come strutturare la risposta (metodo STAR se applicabile)\n"
    "2. **Risposta esempio** - una risposta concreta e convincente\n"
    "3. **Cosa evitare** - errori comuni\n"
    "4. **Bonus tip** - come impressionare l'intervistatore\n\n"
    "Rispondi in italiano.\n\n"
    "Domanda di colloquio: {question}"
)


@track("interview_coach")
def interview_coach(state: AgentState) -> AgentState:
    result = (COACH_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {**state, "response": result.content}
