"""Ricoh AI Knowledge Agent - FastAPI with SSE streaming + observability."""

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
from agents.lineage_agent import lineage_query
from agents.quality_checker import quality_check
from memory import save_turn, get_history
from guardrails import check_input
from feedback import save_feedback, get_feedback_stats
from observability import get_metrics
from agents.quality_checker import get_quality_averages
from longterm_memory import get_memories, extract_facts, save_memories
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
    "lineage": lineage_query,
    "fallback": fallback,
}

RETRIEVAL_ROUTES = {"rag", "summarize", "interview", "architecture", "compare", "diagram"}


class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: str  # "thumbs_up" or "thumbs_down"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    def stream():
        try:
            # Guardrails check
            guard = check_input(req.query)
            if not guard["safe"]:
                yield _sse("error", {"message": f"🛡️ {guard['reason']}"})
                return
            if guard["pii_detected"]:
                yield _sse("warning", {"message": f"⚠️ PII rilevato: {', '.join(guard['pii_detected'])}. Procedo senza memorizzare."})

            # Build initial state
            history = get_history(req.session_id)
            history_ctx = "\n".join(
                f"User: {h.get('question','')}\nAssistant: {h.get('response','')[:200]}"
                for h in history[-3:] if h.get('question')
            ) if history else ""

            question = req.query if not history_ctx else f"[Conversazione precedente:\n{history_ctx}]\n\nNuova domanda: {req.query}"

            # Inject long-term memories
            memories = get_memories(req.session_id)
            if memories:
                question = f"[{memories}]\n\n{question}"

            msg_id = str(uuid.uuid4())[:8]

            state = {
                "question": question,
                "context": "",
                "route": "rag",
                "reasoning": "",
                "response": "",
                "quality": None,
                "session_id": req.session_id,
            }

            # Step 1: Router
            state = run_router(state)
            yield _sse("routing", {
                "agent": state["route"],
                "reasoning": state.get("reasoning", ""),
                "message_id": msg_id,
            })

            # Step 2: Retrieve
            if state["route"] in RETRIEVAL_ROUTES:
                state = run_retriever(state)

            # Step 3: Specialist
            specialist = SPECIALISTS.get(state["route"], rag_generate)
            state = specialist(state)
            yield _sse("response", {"text": state["response"]})

            # Step 4: Quality check (async)
            try:
                state = quality_check(state)
                yield _sse("quality", {"quality": state.get("quality")})
            except Exception:
                yield _sse("quality", {"quality": None})

            # Step 5: Tracing info
            metrics = get_metrics()
            yield _sse("trace", {"metrics": metrics})

            # Step 6: Extract and save long-term memories (async, non-blocking)
            try:
                facts = extract_facts(req.query, state["response"], state["route"])
                if facts:
                    save_memories(req.session_id, facts)
                    yield _sse("memory", {"new_facts": len(facts)})
            except Exception:
                pass

            # Save to Cosmos (skip if PII detected)
            if not guard["pii_detected"]:
                save_turn(
                    session_id=req.session_id,
                    question=req.query,
                    route=state["route"],
                    reasoning=state.get("reasoning", ""),
                    response=state["response"],
                    quality=state.get("quality"),
                )

            yield _sse("done", {"session_id": req.session_id, "message_id": msg_id})

        except Exception as e:
            logging.error(f"Agent error: {e}")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    save_feedback(req.session_id, req.message_id, req.rating)
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    return {
        "agents": get_metrics(),
        "feedback": get_feedback_stats(),
        "quality": get_quality_averages(),
    }


@app.get("/api/architecture")
def architecture():
    """Returns the live system architecture as a Mermaid diagram."""
    return {"diagram": """flowchart TD
    User[Browser / Static Web App] -->|POST /api/chat SSE| CA[Container Apps - FastAPI + LangGraph]

    subgraph Security[Security Layer]
        GR[Guardrails - Anti-injection + PII Detection]
    end

    subgraph Orchestration[LangGraph Orchestration]
        Router[Router Agent - Intent Classification + Reasoning]
        LTM[Long-term Memory - Cosmos DB]
    end

    subgraph Specialists[11 Specialist Agents]
        RAG[RAG Knowledge]
        SUM[Summarizer]
        INT[Interview Coach]
        ARCH[Architecture Advisor]
        CMP[Comparator]
        DIA[Diagram Generator]
        LIN[Data Lineage]
        FB[Fallback]
    end

    subgraph QualityLayer[Quality Assurance]
        QC[Quality Checker - LLM-as-Judge]
        FBK[Feedback Loop]
        OBS[Observability - Tracing + Metrics]
    end

    subgraph DataLayer[Data Layer]
        PGV[(pgvector - PostgreSQL)]
        Cosmos[(Cosmos DB - Conversations + Memory)]
    end

    subgraph Ingestion[Data Ingestion]
        Blob[Azure Blob Storage - Data Lake]
        Func[Azure Function - Blob Trigger]
        ADF[Azure Data Factory - ETL + Lineage]
        CSV[CSV Pipeline - GitHub Actions]
    end

    subgraph MLOps[MLOps]
        MLPipe[Azure ML Pipeline - RAG Evaluation]
        MLW[ML Workspace - MLflow Tracking]
        Gate[Quality Gate - Deploy Block/Allow]
    end

    subgraph CICD[CI/CD + IaC]
        GHA[GitHub Actions - 4 Repos]
        ACR[Azure Container Registry]
        Bicep[Bicep IaC - 15+ Azure Resources]
    end

    subgraph AI[AI Services]
        Claude[Claude Sonnet 4 - Azure AI Foundry]
        OAI[Azure OpenAI - text-embedding-3-small]
    end

    User -->|Query| CA
    CA --> GR
    GR --> Router
    Router -->|Inject memories| LTM
    Router -->|Route| RAG
    Router -->|Route| SUM
    Router -->|Route| INT
    Router -->|Route| ARCH
    Router -->|Route| CMP
    Router -->|Route| DIA
    Router -->|Route| LIN
    Router -->|Route| FB

    RAG -->|Retrieve| PGV
    SUM -->|Retrieve| PGV
    INT -->|Retrieve| PGV
    ARCH -->|Retrieve| PGV
    CMP -->|Retrieve| PGV
    DIA -->|Retrieve| PGV
    LIN -->|Query lineage| PGV

    Specialists --> QC
    QC --> OBS
    QC -->|SSE Stream| User
    FBK -->|Save| Cosmos
    CA -->|Save turns| Cosmos
    LTM -->|Read/Write| Cosmos

    Blob -->|Trigger| Func
    Func -->|Embed + Store| PGV
    ADF -->|Orchestrate| Blob
    CSV -->|Ingest + Lineage| PGV

    MLPipe -->|Evaluate| PGV
    MLPipe --> MLW
    MLPipe --> Gate

    GHA -->|Build| ACR
    ACR -->|Deploy| CA
    GHA -->|Deploy| Bicep

    Specialists -->|Call| Claude
    RAG -->|Embed| OAI"""}


@app.get("/api/health")
def health():
    return {"status": "ok"}
