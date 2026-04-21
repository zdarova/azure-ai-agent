"""Ricoh AI Knowledge Agent - FastAPI with multi-route SSE streaming."""

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
from agents.web_search import web_search
from agents.quality_checker import quality_check, get_quality_averages
from memory import save_turn, get_history
from guardrails import check_input
from feedback import save_feedback, get_feedback_stats
from observability import get_metrics
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
    "web_search": web_search,
    "fallback": fallback,
}

RETRIEVAL_ROUTES = {"rag", "summarize", "interview", "architecture", "compare", "diagram", "web_search"}
NO_RETRIEVE_ROUTES = {"lineage", "fallback"}


class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: str


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    def stream():
        try:
            # Guardrails
            guard = check_input(req.query)
            if not guard["safe"]:
                yield _sse("error", {"message": f"🛡️ {guard['reason']}"})
                return
            if guard["pii_detected"]:
                yield _sse("warning", {"message": f"⚠️ PII rilevato: {', '.join(guard['pii_detected'])}. Procedo senza memorizzare."})

            # Build state
            history = get_history(req.session_id)
            history_ctx = "\n".join(
                f"User: {h.get('question','')}\nAssistant: {h.get('response','')[:200]}"
                for h in history[-3:] if h.get('question')
            ) if history else ""

            question = req.query if not history_ctx else f"[Conversazione precedente:\n{history_ctx}]\n\nNuova domanda: {req.query}"

            memories = get_memories(req.session_id)
            if memories:
                question = f"[{memories}]\n\n{question}"

            msg_id = str(uuid.uuid4())[:8]

            state = {
                "question": question,
                "context": "",
                "routes": ["rag"],
                "route": "rag",
                "reasoning": "",
                "response": "",
                "quality": None,
                "session_id": req.session_id,
            }

            # Step 1: Router (multi-route)
            state = run_router(state)
            routes = state.get("routes", [state.get("route", "rag")])
            yield _sse("routing", {
                "agents": routes,
                "agent": routes[0],
                "reasoning": state.get("reasoning", ""),
                "message_id": msg_id,
                "multi": len(routes) > 1,
            })

            # Step 2: Retrieve once (shared across agents that need it)
            needs_retrieval = any(r in RETRIEVAL_ROUTES for r in routes)
            if needs_retrieval:
                state = run_retriever(state)

            # Step 3: Execute each specialist and stream progressively
            all_responses = []
            for i, agent_route in enumerate(routes):
                state["route"] = agent_route
                specialist = SPECIALISTS.get(agent_route, rag_generate)
                agent_state = specialist(state)

                agent_response = agent_state["response"]
                all_responses.append(agent_response)

                # Send each agent's response as it completes
                yield _sse("agent_response", {
                    "agent": agent_route,
                    "text": agent_response,
                    "index": i,
                    "total": len(routes),
                })

            # Merge all responses
            merged = "\n\n---\n\n".join(all_responses)
            state["response"] = merged
            yield _sse("response", {"text": merged})

            # Step 4: Quality check
            try:
                state = quality_check(state)
                yield _sse("quality", {"quality": state.get("quality")})
            except Exception:
                yield _sse("quality", {"quality": None})

            # Step 5: Trace
            yield _sse("trace", {"metrics": get_metrics()})

            # Step 6: Long-term memory
            try:
                facts = extract_facts(req.query, state["response"], ",".join(routes))
                if facts:
                    save_memories(req.session_id, facts)
                    yield _sse("memory", {"new_facts": len(facts)})
            except Exception:
                pass

            # Save to Cosmos
            if not guard["pii_detected"]:
                save_turn(
                    session_id=req.session_id,
                    question=req.query,
                    route=",".join(routes),
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
    return {"diagram": """flowchart LR
    User[Browser / Static Web App] -->|POST /api/chat SSE| CA[Container Apps - FastAPI + LangGraph]

    subgraph Security[Security Layer]
        GR[Guardrails - Anti-injection + PII Detection]
    end

    subgraph Orchestration[LangGraph Orchestration]
        Router[Router Agent - Multi-Route Classification]
        LTM[Long-term Memory - Cosmos DB]
    end

    subgraph Specialists[12 Specialist Agents]
        RAG[RAG Knowledge]
        SUM[Summarizer]
        INT[Interview Coach]
        ARCH[Architecture Advisor]
        CMP[Comparator]
        DIA[Diagram Generator]
        LIN[Data Lineage]
        WEB[Web Search - DuckDuckGo]
        FB[Fallback]
    end

    subgraph QualityLayer[Quality Assurance - 9 Dimensions]
        QC[Quality Checker - LLM-as-Judge]
        FBK[Feedback Loop]
        OBS[Observability - Tracing + Metrics]
    end

    subgraph DataLayer[Data Layer]
        PGV[(pgvector - PostgreSQL)]
        Cosmos[(Cosmos DB - Conversations + Memory + Metrics)]
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
        DDG[DuckDuckGo Search API]
    end

    User -->|Query| CA
    CA --> GR
    GR --> Router
    Router -->|Inject memories| LTM
    Router -->|Multi-route 1-3 agents| RAG
    Router -->|Multi-route| SUM
    Router -->|Multi-route| INT
    Router -->|Multi-route| ARCH
    Router -->|Multi-route| CMP
    Router -->|Multi-route| DIA
    Router -->|Multi-route| LIN
    Router -->|Multi-route| WEB
    Router -->|Multi-route| FB

    RAG -->|Retrieve| PGV
    SUM -->|Retrieve| PGV
    INT -->|Retrieve| PGV
    ARCH -->|Retrieve| PGV
    CMP -->|Retrieve| PGV
    DIA -->|Retrieve| PGV
    LIN -->|Query lineage| PGV
    WEB -->|Search| DDG

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
