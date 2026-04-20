# Ricoh AI Agent

LangChain RAG agent powered by Claude Sonnet 4 (Azure AI Services) + pgvector.

## Architecture

```
User Query → FastAPI → LangChain Chain:
                         ├─ Embed query (Azure OpenAI)
                         ├─ Retrieve docs (pgvector)
                         ├─ Augment prompt with context
                         └─ Generate response (Claude Sonnet 4)
```

## Local Development

```bash
# Set env vars
export AZURE_AI_ENDPOINT="https://ai-ricoh-xxx.services.ai.azure.com/anthropic"
export AZURE_AI_KEY="<key>"
export AZURE_AI_CHAT_DEPLOYMENT="claude-sonnet-4-6"
export AZURE_OPENAI_ENDPOINT="https://oai-ricoh-xxx.openai.azure.com/"
export AZURE_OPENAI_KEY="<key>"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
export PG_CONNECTION_STRING="host=pg-ricoh-xxx.postgres.database.azure.com port=5432 dbname=ricoh_kb user=pgadmin password=<pass> sslmode=require"

# Run
cd app
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | `{"query": "..."}` → `{"response": "..."}` |
| GET | `/api/health` | Health check |

## Deploy

Push to `main` triggers GitHub Actions → ACR build → Container App update.
