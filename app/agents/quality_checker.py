from observability import track
"""Quality Checker agent - evaluates response quality using LLM-as-judge."""

import os
import json
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
            temperature=0, max_tokens=256,
        )
    return _llm


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict quality evaluator for an AI assistant.\n"
    "The assistant may respond with text, tables, code, or Mermaid diagrams.\n\n"
    "Rate the response on 4 criteria (1-5 each):\n"
    "- relevance: does it address the user's question?\n"
    "- accuracy: is the information correct?\n"
    "- completeness: does it cover the topic adequately?\n"
    "- clarity: is it well-structured and easy to understand?\n\n"
    "For diagrams: evaluate if the diagram correctly represents the requested architecture/flow, "
    "uses appropriate components, and includes a clear explanation.\n\n"
    "User question: {question}\n"
    "Context available: {context}\n"
    "AI response (may contain markdown/mermaid): {response}\n\n"
    "Reply with ONLY valid JSON, no other text:\n"
    '{{"relevance": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "overall": <1-5>, "note": "<one sentence evaluation in Italian>"}}'
)


def _parse_scores(raw: str) -> dict:
    # Try direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


@track("quality_checker")
def quality_check(state: AgentState) -> AgentState:
    result = (JUDGE_PROMPT | _get_llm()).invoke({
        "question": state["question"],
        "context": state["context"][:500],
        "response": state["response"][:1500],
    })

    scores = _parse_scores(result.content.strip())
    if scores is None:
        scores = {"relevance": 3, "accuracy": 3, "completeness": 3, "clarity": 3, "overall": 3, "note": "Parsing della valutazione non riuscito."}

    # Ensure all required keys exist
    for key in ("relevance", "accuracy", "completeness", "clarity", "overall"):
        if key not in scores or not isinstance(scores[key], (int, float)):
            scores[key] = 3
    if "note" not in scores or not isinstance(scores["note"], str):
        scores["note"] = ""

    return {**state, "quality": scores}
