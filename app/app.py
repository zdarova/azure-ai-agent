"""Ricoh AI Knowledge Agent - FastAPI endpoints for Container Apps."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import RicohAgent
import logging

app = FastAPI(title="Ricoh AI Architect Selection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = RicohAgent()


class ChatRequest(BaseModel):
    query: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        result = agent.graph.invoke({
            "question": req.query,
            "context": "",
            "route": "rag",
            "response": "",
            "quality": None,
        })
        return {
            "response": result["response"],
            "agent": result["route"],
            "quality": result.get("quality"),
        }
    except Exception as e:
        logging.error(f"Agent error: {e}")
        return {"error": str(e)}


@app.get("/api/health")
def health():
    return {"status": "ok"}
