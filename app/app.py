"""Ricoh AI Knowledge Agent - FastAPI endpoints for Container Apps."""

import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent import RicohAgent
from memory import save_turn, get_history
import logging

app = FastAPI(title="Ricoh AI Architect Selection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = RicohAgent()


class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        # Get conversation history for context
        history = get_history(req.session_id)
        history_ctx = "\n".join(
            f"User: {h['question']}\nAssistant: {h['response'][:200]}"
            for h in history[-3:]
        ) if history else ""

        result = agent.graph.invoke({
            "question": req.query if not history_ctx else f"[Conversazione precedente:\n{history_ctx}]\n\nNuova domanda: {req.query}",
            "context": "",
            "route": "rag",
            "reasoning": "",
            "response": "",
            "quality": None,
            "session_id": req.session_id,
        })

        # Save to Cosmos DB
        save_turn(
            session_id=req.session_id,
            question=req.query,
            route=result["route"],
            reasoning=result.get("reasoning", ""),
            response=result["response"],
            quality=result.get("quality"),
        )

        return {
            "response": result["response"],
            "agent": result["route"],
            "reasoning": result.get("reasoning", ""),
            "quality": result.get("quality"),
            "session_id": req.session_id,
        }
    except Exception as e:
        logging.error(f"Agent error: {e}")
        return {"error": str(e)}


@app.get("/api/health")
def health():
    return {"status": "ok"}
