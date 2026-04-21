"""Shared state for the LangGraph multi-agent system."""

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    question: str
    context: str
    routes: list[str]
    route: str  # current active route (for backward compat)
    reasoning: str
    response: str
    quality: Optional[dict]
    session_id: str
