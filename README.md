# Ricoh AI Agent

Multi-agent RAG system powered by LangGraph, Claude Sonnet 4 (Azure AI Foundry), and pgvector.

## Architecture

```
User Query → FastAPI (SSE) → LangGraph StateGraph:
    ├─ Guardrails (anti-injection + PII detection)
    ├─ Router (multi-route: selects 1-3 agents per query)
    ├─ Retriever (pgvector similarity search, shared context)
    ├─ Send() fan-out → Specialist agents (parallel execution)
    ├─ Merge (combine responses)
    ├─ Quality Checker (LLM-as-judge, 9 dimensions)
    ├─ Long-term Memory (extract + store user facts)
    └─ Persist (save turn to Cosmos DB)
```

### LangGraph Flow

```
START → guardrails → router → retrieve (conditional) → fan_out → Send(specialist) → merge → quality_check → memory → persist → END
                 ↘ END (if blocked)                                    ↑ (1-3 parallel)
```

## Agents

| Agent | File | Description |
|-------|------|-------------|
| Router | `agents/router.py` | Multi-route classification (1-3 agents), LLM-based |
| Retriever | `agents/retriever.py` | pgvector similarity search (k=4) |
| RAG Knowledge | `agents/rag_agent.py` | Factual Q&A grounded in Ricoh knowledge base |
| Summarizer | `agents/summarizer.py` | Condense and overview topics |
| Interview Coach | `agents/interview_coach.py` | STAR method coaching for job interviews |
| Architecture Advisor | `agents/architect.py` | Azure AI/ML architecture proposals |
| Comparator | `agents/comparator.py` | Side-by-side technology comparisons |
| Diagram Generator | `agents/diagram.py` | Mermaid.js diagrams with syntax validation + auto-fix |
| Data Lineage | `agents/lineage_agent.py` | Query data provenance and pipeline runs |
| Web Search | `agents/web_search.py` | DuckDuckGo search + LLM summarization with source links |
| Fallback | `agents/fallback.py` | Greetings and off-topic handling |
| Quality Checker | `agents/quality_checker.py` | LLM-as-judge scoring on 9 dimensions |

## Multi-Route Examples

The router can activate multiple agents for complex queries:

- *"Design a RAG system and show the diagram"* → `architecture` + `diagram`
- *"Compare pgvector vs Pinecone and summarize"* → `compare` + `summarize`
- *"What are the latest AI trends for Ricoh?"* → `web_search` + `rag`

Each agent's response streams progressively via SSE as it completes.

## Project Structure

```
├── app/
│   ├── agents/              # 12 specialist agents
│   │   ├── __init__.py      # AgentState (TypedDict with reducers)
│   │   ├── router.py        # Multi-route LLM classifier
│   │   ├── retriever.py     # pgvector retrieval
│   │   ├── rag_agent.py     # RAG knowledge
│   │   ├── summarizer.py    # Summarization
│   │   ├── interview_coach.py
│   │   ├── architect.py     # Azure architecture advisor
│   │   ├── comparator.py    # Technology comparisons
│   │   ├── diagram.py       # Mermaid.js generation + validation
│   │   ├── lineage_agent.py # Data lineage queries
│   │   ├── web_search.py    # DuckDuckGo + LLM summarization
│   │   ├── fallback.py      # Off-topic handling
│   │   └── quality_checker.py # 9-dimension LLM-as-judge
│   ├── app.py               # FastAPI + SSE streaming via graph.stream()
│   ├── graph.py             # LangGraph StateGraph with Send fan-out
│   ├── guardrails.py        # Prompt injection + PII detection
│   ├── memory.py            # Cosmos DB conversation history
│   ├── longterm_memory.py   # Fact extraction + persistent user memory
│   ├── feedback.py          # Thumbs up/down tracking
│   ├── observability.py     # Agent latency metrics (Cosmos DB)
│   ├── Dockerfile
│   └── requirements.txt
├── web/
│   ├── index.html           # Chat UI structure
│   ├── style.css            # Styles
│   └── app.js               # SSE client, multi-route rendering, Mermaid zoom/drag
├── tests/
│   └── test_agents.py       # 37 tests (agents, graph, guardrails, fan-out, merge)
└── .github/workflows/
    ├── deploy-agent.yml     # CI/CD: ACR build → Container Apps
    └── deploy-swa.yml       # CI/CD: Static Web App deploy
```

## SSE Events

The `/api/chat` endpoint streams these events as the LangGraph executes:

| Event | Payload | When |
|-------|---------|------|
| `warning` | `{message}` | PII detected in input |
| `routing` | `{agents, agent, reasoning, message_id, multi}` | Router selected agents |
| `agent_response` | `{agent, text}` | Each specialist completes (progressive) |
| `response` | `{text}` | Final merged response |
| `quality` | `{quality}` | 9-dimension quality scores |
| `trace` | `{metrics}` | Agent latency metrics |
| `memory` | `{new_facts}` | Long-term facts extracted |
| `done` | `{session_id, message_id}` | Stream complete |
| `error` | `{message}` | Guardrails block or exception |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | SSE streaming chat (LangGraph pipeline) |
| POST | `/api/feedback` | Save thumbs up/down rating |
| GET | `/api/metrics` | Agent performance + quality averages + feedback stats |
| GET | `/api/architecture` | Live Mermaid architecture diagram |
| GET | `/api/health` | Health check |

## Local Development

```bash
export AZURE_AI_ENDPOINT="https://ai-ricoh-xxx.services.ai.azure.com/anthropic"
export AZURE_AI_KEY="<key>"
export AZURE_AI_CHAT_DEPLOYMENT="claude-sonnet-4-6"
export AZURE_OPENAI_ENDPOINT="https://oai-ricoh-xxx.openai.azure.com/"
export AZURE_OPENAI_KEY="<key>"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
export PG_CONNECTION_STRING="host=pg-ricoh-xxx.postgres.database.azure.com port=5432 dbname=ricoh_kb user=pgadmin password=<pass> sslmode=require"
export COSMOS_ENDPOINT="https://cosmos-ricoh-xxx.documents.azure.com:443/"
export COSMOS_KEY="<key>"

cd app
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Tests

```bash
python -m pytest tests/test_agents.py -v
```

37 tests covering: state schema, router (single/multi/invalid/malformed), graph structure, guardrails (injection/PII/safe), fan-out (Send API), specialist dispatch, merge (single/multi), all 9 specialist agents, quality checker, web search, state isolation.

## Deploy

Push to `main` triggers GitHub Actions → Docker build → ACR push → Container Apps update.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude Sonnet 4 (Azure AI Foundry) |
| Embeddings | text-embedding-3-small (Azure OpenAI) |
| Orchestration | LangGraph (StateGraph + Send fan-out) |
| Framework | LangChain |
| Vector DB | pgvector (PostgreSQL Flexible Server) |
| Conversation Memory | Azure Cosmos DB (serverless) |
| API | FastAPI + SSE streaming |
| Web Search | DuckDuckGo (ddgs) |
| Frontend | Static Web App (vanilla HTML/CSS/JS) |
| Container | Azure Container Apps (consumption) |
| Registry | Azure Container Registry |
| CI/CD | GitHub Actions |
| IaC | Bicep (azure-bicep-mlops repo) |

## Related Repos

- **azure-ai-data** — KB data ETL, CSV ingestion, Azure Functions blob trigger, data lineage
- **azure-ai-ml-pipeline** — RAG evaluation pipeline, MLflow metrics, quality gates
- **azure-bicep-mlops** — Bicep IaC for all Azure infrastructure (15+ resources)
