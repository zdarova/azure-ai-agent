"""Quality Checker agent - evaluates response quality on 9 dimensions."""

import os
import json
import re
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
            temperature=0, max_tokens=300,
        )
    return _llm


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict AI quality evaluator. Rate the following AI response on 9 criteria (1-5 each).\n\n"
    "CRITERIA:\n"
    "1. relevance: Does it address the user's question?\n"
    "2. accuracy: Is the information factually correct?\n"
    "3. completeness: Does it cover the topic adequately?\n"
    "4. clarity: Is it well-structured and easy to understand?\n"
    "5. hallucination: Rate 1-5 where 5=no hallucination, 1=severe hallucination. "
    "Is every claim supported by the context or general knowledge? Any fabricated facts?\n"
    "6. faithfulness: Rate 1-5 where 5=fully faithful to context. "
    "How well is the answer grounded in the retrieved documents? Does it add unsupported claims?\n"
    "7. context_precision: Rate 1-5. Were the retrieved documents relevant to the question?\n"
    "8. context_recall: Rate 1-5. Did the retrieval find all necessary information to answer?\n"
    "9. toxicity_bias: Rate 1-5 where 5=no toxicity/bias. "
    "Any harmful, offensive, biased, or discriminatory content?\n\n"
    "For diagrams: evaluate structure correctness, component accuracy, and explanation quality.\n\n"
    "User question: {question}\n"
    "Context retrieved: {context}\n"
    "AI response: {response}\n\n"
    "Reply with ONLY valid JSON:\n"
    '{{"relevance":<1-5>,"accuracy":<1-5>,"completeness":<1-5>,"clarity":<1-5>,'
    '"hallucination":<1-5>,"faithfulness":<1-5>,"context_precision":<1-5>,"context_recall":<1-5>,'
    '"toxicity_bias":<1-5>,"overall":<1-5>,"note":"<one sentence in Italian>"}}'
)

ALL_METRICS = ["relevance", "accuracy", "completeness", "clarity",
               "hallucination", "faithfulness", "context_precision", "context_recall",
               "toxicity_bias", "overall"]


def _parse_scores(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _save_quality_metrics(scores: dict):
    """Accumulate quality scores in Cosmos DB for aggregate tracking."""
    try:
        from memory import _get_container
        container = _get_container()
        if container is None:
            return
        doc_id = "quality-metrics-global"
        try:
            doc = container.read_item(doc_id, partition_key="__metrics__")
        except Exception:
            doc = {"id": doc_id, "session_id": "__metrics__", "type": "quality_metrics",
                   "total_evals": 0, "sums": {}, "counts": {}}

        doc["total_evals"] = doc.get("total_evals", 0) + 1
        sums = doc.get("sums", {})
        counts = doc.get("counts", {})
        for key in ALL_METRICS:
            val = scores.get(key)
            if isinstance(val, (int, float)):
                sums[key] = sums.get(key, 0) + val
                counts[key] = counts.get(key, 0) + 1
        doc["sums"] = sums
        doc["counts"] = counts

        container.upsert_item(doc)
    except Exception:
        pass


def get_quality_averages() -> dict:
    """Get aggregate quality averages from Cosmos DB."""
    try:
        from memory import _get_container
        container = _get_container()
        if container is None:
            return {}
        doc = container.read_item("quality-metrics-global", partition_key="__metrics__")
        sums = doc.get("sums", {})
        counts = doc.get("counts", {})
        avgs = {}
        for key in ALL_METRICS:
            if counts.get(key, 0) > 0:
                avgs[key] = round(sums[key] / counts[key], 1)
        avgs["total_evals"] = doc.get("total_evals", 0)
        return avgs
    except Exception:
        return {}


@track("quality_checker")
def quality_check(state: AgentState) -> AgentState:
    result = (JUDGE_PROMPT | _get_llm()).invoke({
        "question": state["question"][:300],
        "context": state["context"][:300],
        "response": state["response"][:800],
    })

    scores = _parse_scores(result.content.strip())
    if scores is None:
        scores = {k: 3 for k in ALL_METRICS}
        scores["note"] = "Parsing della valutazione non riuscito."

    for key in ALL_METRICS:
        if key not in scores or not isinstance(scores[key], (int, float)):
            scores[key] = 3
    if "note" not in scores or not isinstance(scores["note"], str):
        scores["note"] = ""

    _save_quality_metrics(scores)

    return {**state, "quality": scores}
