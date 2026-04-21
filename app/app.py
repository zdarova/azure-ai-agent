"""Ricoh AI Knowledge Agent - FastAPI with progressive SSE streaming."""

import uuid
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from agents.router import route as run_router
from agents.retriever import retrieve as run_retriever
from agents.rag_agent import rag_generate
from agents.summarizer import summarize
from agents.fallback import fallback
from agents.interview_coach import interview_coach
from agents.architect import architecture_advisor
from agents.comparator import compare
from agents.diagram import diagram
from agents.quality_checker import quality_check
from memory import save_turn, get_history
import logging

app = FastAPI(title="Ricoh AI Architect Selection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SPECIALISTS = {
    "rag": rag_generate,
    "summarize": summarize,
    "interview": interview_coach,
    "architecture": architecture_advisor,
    "compare": compare,
    "diagram": diagram,
    "fallback": fallback,
}

RETRIEVAL_ROUTES = {"rag", "summarize", "interview", "architecture", "compare", "diagram"}


class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    def stream():
        try:
            # Build initial state
            history = get_history(req.session_id)
            history_ctx = "\n".join(
                f"User: {h['question']}\nAssistant: {h['response'][:200]}"
                for h in history[-3:]
            ) if history else ""

            question = req.query if not history_ctx else f"[Conversazione precedente:\n{history_ctx}]\n\nNuova domanda: {req.query}"

            state = {
                "question": question,
                "context": "",
                "route": "rag",
                "reasoning": "",
                "response": "",
                "quality": None,
                "session_id": req.session_id,
            }

            # Step 1: Router — send immediately
            state = run_router(state)
            yield _sse("routing", {
                "agent": state["route"],
                "reasoning": state.get("reasoning", ""),
            })

            # Step 2: Retrieve (if needed)
            if state["route"] in RETRIEVAL_ROUTES:
                state = run_retriever(state)

            # Step 3: Specialist — send response
            specialist = SPECIALISTS.get(state["route"], rag_generate)
            state = specialist(state)
            yield _sse("response", {"text": state["response"]})

            # Step 4: Quality check — async after response
            try:
                state = quality_check(state)
                yield _sse("quality", {"quality": state.get("quality")})
            except Exception:
                yield _sse("quality", {"quality": None})

            # Save to Cosmos
            save_turn(
                session_id=req.session_id,
                question=req.query,
                route=state["route"],
                reasoning=state.get("reasoning", ""),
                response=state["response"],
                quality=state.get("quality"),
            )

            yield _sse("done", {"session_id": req.session_id})

        except Exception as e:
            logging.error(f"Agent error: {e}")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/health")
def health():
    return {"status": "ok"}
