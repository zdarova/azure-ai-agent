from observability import track
"""Diagram agent - generates Mermaid.js architecture diagrams with syntax validation."""

import os
import re
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
    "REGOLE SINTASSI MERMAID IMPORTANTI:\n"
    "- Usa SOLO flowchart TD o flowchart LR\n"
    "- NON usare virgolette nei nodi, usa parentesi: A[Testo del nodo]\n"
    "- NON usare caratteri speciali come & dentro i nodi, scrivi 'e' invece di '&'\n"
    "- Per subgraph usa: subgraph NomeGruppo e end\n"
    "- Per le frecce usa: A --> B oppure A -->|etichetta| B\n"
    "- NON usare apici doppi nei nodi o nelle etichette\n"
    "- Ogni ID nodo deve essere alfanumerico senza spazi: A1, NodeName, ecc.\n"
    "- Il codice Mermaid DEVE essere in un blocco ```mermaid\n"
    "- Dopo il diagramma, aggiungi una breve spiegazione in italiano\n\n"
    "Richiesta: {question}"
)

FIX_PROMPT = ChatPromptTemplate.from_template(
    "The following Mermaid.js code has syntax errors. Fix it to be valid Mermaid syntax.\n"
    "Common issues: quotes inside nodes, special characters like &, invalid arrow syntax.\n"
    "Rules:\n"
    "- Use square brackets for nodes: A[Text here]\n"
    "- No double quotes inside node labels\n"
    "- Replace & with 'e' or 'and'\n"
    "- Use --> for arrows, -->|label| for labeled arrows\n"
    "- subgraph Name ... end\n"
    "- Node IDs must be alphanumeric without spaces\n\n"
    "Return ONLY the fixed mermaid code, no explanation, no ```mermaid wrapper.\n\n"
    "{code}"
)


def _extract_mermaid(text: str) -> list[str]:
    return re.findall(r'```mermaid\s*\n(.*?)```', text, re.DOTALL)


def _validate_mermaid(code: str) -> bool:
    """Basic syntax validation for common Mermaid errors."""
    lines = code.strip().split('\n')
    if not lines:
        return False
    first = lines[0].strip().lower()
    if not any(first.startswith(k) for k in ('flowchart', 'graph', 'sequencediagram', 'sequence', 'classDiagram', 'gantt', 'pie', 'erdiagram')):
        if not first.startswith('graph') and not first.startswith('flowchart'):
            return False
    # Check for common syntax issues
    for line in lines:
        if '["' in line or '("' in line or '{"' in line:
            return False
        if '"' in line and '-->' in line:
            return False
    return True


def _fix_mermaid_basic(code: str) -> str:
    """Quick regex fixes for common issues."""
    # Replace ["text"] with [text]
    code = re.sub(r'\["([^"]*?)"\]', r'[\1]', code)
    code = re.sub(r'\("([^"]*?)"\)', r'(\1)', code)
    code = re.sub(r'\{"([^"]*?)"\}', r'{\1}', code)
    # Replace "label" in arrows with |label|
    code = re.sub(r'-->\s*"([^"]*?)"', r'-->|\1|', code)
    # Replace & with 'e'
    code = code.replace(' & ', ' e ')
    code = code.replace('&amp;', 'e')
    return code


@track("diagram")
def diagram(state: AgentState) -> AgentState:
    result = (DIAGRAM_PROMPT | _get_llm()).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    response = result.content

    # Extract and validate each mermaid block
    blocks = _extract_mermaid(response)
    for original_code in blocks:
        fixed = _fix_mermaid_basic(original_code)
        if fixed != original_code:
            response = response.replace(original_code, fixed)

        # If still has issues, ask LLM to fix
        if not _validate_mermaid(fixed):
            try:
                fix_result = (FIX_PROMPT | _get_llm()).invoke({"code": fixed})
                fixed_code = fix_result.content.strip()
                # Remove any wrapper the LLM might add
                fixed_code = re.sub(r'^```mermaid\s*\n?', '', fixed_code)
                fixed_code = re.sub(r'\n?```$', '', fixed_code)
                response = response.replace(fixed, fixed_code)
            except Exception:
                pass

    return {**state, "response": response}
