"""Long-term memory - extracts and stores user facts/preferences to improve responses over time."""

import os
import json
import logging
from datetime import datetime, timezone
from memory import _get_container

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_anthropic import ChatAnthropic
        _llm = ChatAnthropic(
            model=os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
            api_key=os.environ["AZURE_AI_KEY"],
            base_url=os.environ["AZURE_AI_ENDPOINT"],
            temperature=0, max_tokens=300,
        )
    return _llm


EXTRACT_PROMPT = (
    "Extract key facts about the user from this conversation turn. "
    "Focus on: skills, interests, role, preferences, goals, topics they care about.\n"
    "If no new facts, return empty array.\n\n"
    "User question: {question}\n"
    "AI response: {response}\n"
    "Route used: {route}\n\n"
    'Reply ONLY with JSON array: [{{"fact": "...", "category": "skill|interest|preference|goal|context"}}]'
)


def extract_facts(question: str, response: str, route: str) -> list:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(EXTRACT_PROMPT)
        result = (prompt | _get_llm()).invoke({
            "question": question[:500],
            "response": response[:500],
            "route": route,
        })
        raw = result.content.strip()
        # Parse JSON array
        import re
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logging.warning(f"Failed to extract facts: {e}")
    return []


def save_memories(session_id: str, facts: list):
    container = _get_container()
    if container is None or not facts:
        return
    try:
        # Upsert a single memory document per session (accumulates facts)
        mem_id = f"memory-{session_id}"
        try:
            existing = container.read_item(mem_id, partition_key=session_id)
            existing_facts = existing.get("facts", [])
        except Exception:
            existing_facts = []

        # Deduplicate by fact text
        existing_texts = {f["fact"] for f in existing_facts}
        new_facts = [f for f in facts if f.get("fact") and f["fact"] not in existing_texts]

        if not new_facts:
            return

        all_facts = existing_facts + new_facts
        # Keep last 20 facts max
        all_facts = all_facts[-20:]

        container.upsert_item({
            "id": mem_id,
            "session_id": session_id,
            "type": "memory",
            "facts": all_facts,
            "updated": datetime.now(timezone.utc).isoformat(),
        })
        logging.info(f"Saved {len(new_facts)} new memories for session {session_id[:8]}")
    except Exception as e:
        logging.warning(f"Failed to save memories: {e}")


def get_memories(session_id: str) -> str:
    container = _get_container()
    if container is None:
        return ""
    try:
        mem_id = f"memory-{session_id}"
        doc = container.read_item(mem_id, partition_key=session_id)
        facts = doc.get("facts", [])
        if not facts:
            return ""
        return "Informazioni note sull'utente:\n" + "\n".join(
            f"- [{f.get('category','info')}] {f['fact']}" for f in facts
        )
    except Exception:
        return ""
