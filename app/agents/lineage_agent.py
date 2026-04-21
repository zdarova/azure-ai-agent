"""Lineage agent - queries data lineage from pgvector metadata and data_lineage table."""

import os
import psycopg
from observability import track
from agents import AgentState


def _get_dsn():
    pg = os.environ["PG_CONNECTION_STRING"]
    parts = dict(p.split("=", 1) for p in pg.split() if "=" in p)
    return f"host={parts['host']} port={parts['port']} dbname={parts['dbname']} user={parts['user']} password={parts['password']} sslmode={parts['sslmode']}"


def _query_pipeline_runs() -> str:
    try:
        with psycopg.connect(_get_dsn()) as conn:
            rows = conn.execute(
                "SELECT pipeline_run_id, source_file, source_type, rows_extracted, chunks_created, status, timestamp "
                "FROM data_lineage ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()
            if not rows:
                return "Nessun pipeline run trovato nella tabella data_lineage."
            lines = ["| Run ID | Source | Type | Rows | Chunks | Status | Timestamp |",
                     "|--------|--------|------|------|--------|--------|-----------|"]
            for r in rows:
                lines.append(f"| {r[0][:8]}... | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6][:19]} |")
            return "\n".join(lines)
    except Exception as e:
        return f"Errore query pipeline runs: {e}"


def _trace_document(search_term: str) -> str:
    try:
        with psycopg.connect(_get_dsn()) as conn:
            rows = conn.execute(
                "SELECT cmetadata->>'title', cmetadata->>'source_file', cmetadata->>'source_row', "
                "cmetadata->>'pipeline_run_id', cmetadata->>'ingestion_timestamp', "
                "cmetadata->>'lineage', cmetadata->>'chunk_index' "
                "FROM langchain_pg_embedding "
                "WHERE cmetadata->>'title' ILIKE %s OR cmetadata->>'lineage' ILIKE %s "
                "LIMIT 10",
                (f"%{search_term}%", f"%{search_term}%")
            ).fetchall()
            if not rows:
                return f"Nessun documento trovato per '{search_term}'."
            lines = [f"### Lineage per '{search_term}' ({len(rows)} risultati)\n"]
            for r in rows:
                lines.append(f"**{r[0]}** (chunk {r[6]})")
                lines.append(f"- Source: `{r[1]}`")
                lines.append(f"- Row: {r[2]}")
                lines.append(f"- Pipeline: `{r[3][:8] if r[3] else 'N/A'}...`")
                lines.append(f"- Timestamp: {r[4][:19] if r[4] else 'N/A'}")
                lines.append(f"- Lineage: `{r[5]}`")
                lines.append("")
            return "\n".join(lines)
    except Exception as e:
        return f"Errore trace: {e}"


def _get_stats() -> str:
    try:
        with psycopg.connect(_get_dsn()) as conn:
            total = conn.execute("SELECT count(*) FROM langchain_pg_embedding").fetchone()[0]
            with_lineage = conn.execute(
                "SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'pipeline_run_id' IS NOT NULL"
            ).fetchone()[0]
            sources = conn.execute(
                "SELECT cmetadata->>'source_file', count(*) FROM langchain_pg_embedding "
                "WHERE cmetadata->>'source_file' IS NOT NULL "
                "GROUP BY cmetadata->>'source_file' ORDER BY count(*) DESC LIMIT 10"
            ).fetchall()

            lines = [f"### Knowledge Base Stats\n",
                     f"- **Totale documenti**: {total}",
                     f"- **Con lineage**: {with_lineage}",
                     f"- **Senza lineage**: {total - with_lineage}\n",
                     "### Fonti dati\n",
                     "| Source | Chunks |",
                     "|--------|--------|"]
            for s in sources:
                lines.append(f"| {s[0]} | {s[1]} |")
            if not sources:
                lines.append("| (nessuna fonte tracciata) | - |")
            return "\n".join(lines)
    except Exception as e:
        return f"Errore stats: {e}"


@track("lineage")
def lineage_query(state: AgentState) -> AgentState:
    question = state["question"].lower()

    # Determine what kind of lineage query
    if any(w in question for w in ["pipeline", "run", "esecuzioni", "ingestion"]):
        result = "## Pipeline Runs\n\n" + _query_pipeline_runs()
    elif any(w in question for w in ["stat", "overview", "panoramica", "quanti", "totale"]):
        result = _get_stats()
    else:
        # Extract search term — take the most meaningful words
        search = state["question"]
        # Try to find a quoted term or the last noun
        import re
        quoted = re.findall(r'"([^"]+)"', search)
        if quoted:
            term = quoted[0]
        else:
            # Use context from retriever if available
            term = search.split("lineage")[-1].split("traccia")[-1].split("origine")[-1].strip(" ?.,")
            if len(term) < 2:
                term = search

        result = _trace_document(term)
        result += "\n\n---\n" + _query_pipeline_runs()

    return {**state, "response": result}
