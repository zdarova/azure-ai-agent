"""Ricoh AI Knowledge Agent - FastAPI + LangGraph with SSE streaming."""

import uuid
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from graph import build_graph
from agents.quality_checker import get_quality_averages
from feedback import save_feedback, get_feedback_stats
from observability import get_metrics
import logging

app = FastAPI(title="Ricoh AI Architect Selection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


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
        msg_id = str(uuid.uuid4())[:8]
        initial_state = {
            "question": req.query,
            "context": "",
            "routes": ["rag"],
            "route": "rag",
            "reasoning": "",
            "agent_responses": [],
            "response": "",
            "quality": None,
            "session_id": req.session_id,
            "pii_detected": [],
        }

        try:
            for event in _get_graph().stream(initial_state):
                for node_name, node_output in event.items():

                    if node_name == "guardrails":
                        # Blocked by guardrails
                        if node_output.get("routes") == ["__blocked__"]:
                            yield _sse("error", {"message": node_output.get("response", "Blocked")})
                            return
                        # PII warning
                        pii = node_output.get("pii_detected", [])
                        if pii:
                            yield _sse("warning", {"message": f"⚠️ PII rilevato: {', '.join(pii)}. Procedo senza memorizzare."})

                    elif node_name == "router":
                        routes = node_output.get("routes", ["rag"])
                        yield _sse("routing", {
                            "agents": routes,
                            "agent": routes[0],
                            "reasoning": node_output.get("reasoning", ""),
                            "message_id": msg_id,
                            "multi": len(routes) > 1,
                        })

                    elif node_name == "specialist":
                        for ar in node_output.get("agent_responses", []):
                            yield _sse("agent_response", {
                                "agent": ar["agent"],
                                "text": ar["text"],
                            })

                    elif node_name == "merge":
                        yield _sse("response", {"text": node_output.get("response", "")})

                    elif node_name == "quality_check":
                        yield _sse("quality", {"quality": node_output.get("quality")})
                        yield _sse("trace", {"metrics": get_metrics()})

                    elif node_name == "memory":
                        # Memory node returns {} but we can check state
                        pass

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
