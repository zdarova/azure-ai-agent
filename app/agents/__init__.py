"""Shared state for the LangGraph multi-agent system."""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    question: str
    context: str
    routes: list[str]
    route: str
    reasoning: str
    agent_responses: Annotated[list[dict], operator.add]
    response: str
    quality: Optional[dict]
    session_id: str
    pii_detected: list[str]
