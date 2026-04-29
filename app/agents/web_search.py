"""Web Search agent - searches DuckDuckGo and summarizes results."""

import os
import logging
from observability import track
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
            temperature=0.3, max_tokens=1024,
        )
    return _llm


def _search_ddg(query: str, max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        logging.warning(f"DuckDuckGo search failed: {e}")
        return []


SEARCH_PROMPT = ChatPromptTemplate.from_template(
    "Sei un assistente AI di Ricoh Italia. L'utente ha fatto una domanda che richiede "
    "informazioni esterne. Ecco i risultati della ricerca web.\n\n"
    "Risultati di ricerca:\n{search_results}\n\n"
    "Contesto dalla knowledge base Ricoh:\n{context}\n\n"
    "Domanda: {question}\n\n"
    "Rispondi in italiano, citando le fonti quando possibile. "
    "Integra le informazioni web con il contesto Ricoh se rilevante."
)


@track("web_search")
def web_search(state: AgentState) -> AgentState:
    # Extract a clean search query
    question = state["question"]
    # Remove conversation history prefix if present
    if "Nuova domanda:" in question:
        question = question.split("Nuova domanda:")[-1].strip()

    results = _search_ddg(question, max_results=3)

    if not results:
        search_text = "Nessun risultato trovato dalla ricerca web."
    else:
        search_text = "\n\n".join(
            f"**{r.get('title', 'N/A')}**\n{r.get('body', '')}\nFonte: {r.get('href', '')}"
            for r in results
        )

    result = (SEARCH_PROMPT | _get_llm()).invoke({
        "search_results": search_text[:3000],
        "context": state["context"][:500],
        "question": state["question"],
    })

    # Add source links at the bottom
    sources = "\n\n---\n📎 **Fonti web:**\n" + "\n".join(
        f"- [{r.get('title', 'Link')}]({r.get('href', '')})"
        for r in results[:5]
    ) if results else ""

    return {**state, "response": result.content + sources}
