"""Quality Checker agent - evaluates response quality using LLM-as-judge."""

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
            temperature=0, max_tokens=256,
        )
    return _llm


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict quality evaluator. Rate the following AI response on 4 criteria.\n"
    "Each criterion is scored 1-5. Reply ONLY with this exact JSON format, nothing else:\n"
    '{{"relevance": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "overall": <1-5>, "note": "<one sentence in Italian>"}}\n\n'
    "User question: {question}\n"
    "Context available: {context}\n"
    "AI response: {response}"
)


def quality_check(state: AgentState) -> AgentState:
    result = (JUDGE_PROMPT | _get_llm()).invoke({
        "question": state["question"],
        "context": state["context"],
        "response": state["response"],
    })
    raw = result.content.strip()

    # Parse the JSON score
    try:
        import json
        scores = json.loads(raw)
    except Exception:
        scores = {"relevance": 3, "accuracy": 3, "completeness": 3, "clarity": 3, "overall": 3, "note": "Valutazione non disponibile"}

    return {**state, "quality": scores}
