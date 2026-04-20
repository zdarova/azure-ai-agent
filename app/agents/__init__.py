"""Shared state for the LangGraph multi-agent system."""

from typing import Literal, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    question: str
    context: str
    route: Literal["rag", "summarize", "interview", "architecture", "compare", "fallback"]
    response: str
    quality: Optional[dict]
