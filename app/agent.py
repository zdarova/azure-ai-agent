"""Ricoh AI Knowledge Agent - LangChain + Claude Sonnet 4 (Azure AI Services) + pgvector RAG."""

import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

SYSTEM_PROMPT = """Sei un assistente AI esperto di Ricoh Italia e delle sue soluzioni enterprise.
Rispondi in italiano in modo professionale e conciso.
Usa il contesto fornito per rispondere alle domande. Se non hai informazioni sufficienti, dillo chiaramente.

Contesto recuperato dalla knowledge base Ricoh:
{context}

Domanda: {question}"""


def pg_conn_to_sqlalchemy(pg_str: str) -> str:
    parts = dict(p.split("=", 1) for p in pg_str.split() if "=" in p)
    return f"postgresql+psycopg://{parts['user']}:{parts['password']}@{parts['host']}:{parts['port']}/{parts['dbname']}?sslmode={parts['sslmode']}"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RicohAgent:
    def __init__(self):
        # Claude via Azure AI Services (Anthropic native API)
        self.llm = ChatAnthropic(
            model=os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
            api_key=os.environ["AZURE_AI_KEY"],
            base_url=os.environ["AZURE_AI_ENDPOINT"],
            temperature=0.3,
            max_tokens=2048,
        )

        # Embeddings via Azure OpenAI
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            api_version="2024-06-01",
        )

        conn_str = pg_conn_to_sqlalchemy(os.environ["PG_CONNECTION_STRING"])
        self.vectorstore = PGVector(
            connection=conn_str,
            embeddings=self.embeddings,
            collection_name="ricoh_knowledge",
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def run(self, query: str) -> str:
        return self.chain.invoke(query)
