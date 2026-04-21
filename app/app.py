"""Ricoh AI Knowledge Agent - FastAPI with SSE streaming."""

import uuid
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from agent import RicohAgent
from agents.quality_checker import quality_check
from memory import save_turn, get_history
import logging

app = FastAPI(title="Ricoh AI Architect Selection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = RicohAgent()


class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    def stream():
        try:
            # Get conversation history
            history = get_history(req.session_id)
            history_ctx = "\n".join(
                f"User: {h['question']}\nAssistant: {h['response'][:200]}"
                for h in history[-3:]
            ) if history else ""

            question = req.query if not history_ctx else f"[Conversazione precedente:\n{history_ctx}]\n\nNuova domanda: {req.query}"

            # Run the main graph (router → retrieve → specialist)
            result = agent.graph.invoke({
                "question": question,
                "context": "",
                "route": "rag",
                "reasoning": "",
                "response": "",
                "quality": None,
                "session_id": req.session_id,
            })

            # Send routing + reasoning immediately
            yield _sse("routing", {
                "agent": result["route"],
                "reasoning": result.get("reasoning", ""),
            })

            # Send the full response
            yield _sse("response", {
                "text": result["response"],
            })

            # Run quality check (async, after response is already sent)
            try:
                checked = quality_check(result)
                quality = checked.get("quality")
            except Exception:
                quality = None

            yield _sse("quality", {
                "quality": quality,
            })

            # Save to Cosmos DB
            save_turn(
                session_id=req.session_id,
                question=req.query,
                route=result["route"],
                reasoning=result.get("reasoning", ""),
                response=result["response"],
                quality=quality,
            )

            yield _sse("done", {"session_id": req.session_id})

        except Exception as e:
            logging.error(f"Agent error: {e}")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")


# Keep the old endpoint for backward compat / health
@app.get("/api/health")
def health():
    return {"status": "ok"}
