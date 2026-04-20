"""Retriever node - fetches relevant documents from pgvector."""

import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from agents import AgentState


def _pg_conn() -> str:
    pg = os.environ["PG_CONNECTION_STRING"]
    parts = dict(p.split("=", 1) for p in pg.split() if "=" in p)
    return f"postgresql+psycopg://{parts['user']}:{parts['password']}@{parts['host']}:{parts['port']}/{parts['dbname']}?sslmode={parts['sslmode']}"


_embeddings = None
_vectorstore = None


def _get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            api_version="2024-06-01",
        )
        _vectorstore = PGVector(
            connection=_pg_conn(),
            embeddings=_embeddings,
            collection_name="ricoh_knowledge",
        )
    return _vectorstore


def retrieve(state: AgentState) -> AgentState:
    docs = _get_vectorstore().similarity_search(state["question"], k=4)
    context = "\n\n".join(d.page_content for d in docs)
    return {**state, "context": context}
