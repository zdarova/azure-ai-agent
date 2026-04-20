"""Architecture Advisor agent - proposes Azure AI architectures using Ricoh's stack."""

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


ARCH_PROMPT = ChatPromptTemplate.from_template(
    "Sei un Solutions Architect esperto di Azure e AI. Lavori per Ricoh Italia.\n\n"
    "Contesto sulle tecnologie Ricoh:\n{context}\n\n"
    "Per il seguente scenario, proponi un'architettura Azure completa:\n"
    "1. **Diagramma** - descrivi i componenti e il flusso (usa frecce →)\n"
    "2. **Servizi Azure** - lista ogni servizio con il suo ruolo\n"
    "3. **Perché questa scelta** - giustifica le decisioni architetturali\n"
    "4. **Stima costi** - indicazione qualitativa (basso/medio/alto)\n\n"
    "Usa le tecnologie dello stack Ricoh dove possibile: Claude Sonnet 4, Azure AI Foundry, "
    "Azure OpenAI, LangChain, pgvector, Azure ML, Azure Functions, Container Apps.\n\n"
    "Rispondi in italiano.\n\n"
    "Scenario: {question}"
)


def architecture_advisor(state: AgentState) -> AgentState:
    result = (ARCH_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {**state, "response": result.content}
