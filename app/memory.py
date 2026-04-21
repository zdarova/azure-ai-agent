"""Cosmos DB conversation memory."""

import os
import logging
from datetime import datetime, timezone

_client = None
_container = None


def _get_container():
    global _client, _container
    if _container is None:
        try:
            from azure.cosmos import CosmosClient
            endpoint = os.environ.get("COSMOS_ENDPOINT")
            key = os.environ.get("COSMOS_KEY")
            if not endpoint or not key:
                return None
            _client = CosmosClient(endpoint, key)
            db = _client.get_database_client("ricoh_agent")
            _container = db.get_container_client("conversations")
        except Exception as e:
            logging.warning(f"Cosmos DB not available: {e}")
            return None
    return _container


def save_turn(session_id: str, question: str, route: str, reasoning: str,
              response: str, quality: dict = None):
    container = _get_container()
    if container is None:
        return
    try:
        container.upsert_item({
            "id": f"{session_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "route": route,
            "reasoning": reasoning,
            "response": response[:2000],
            "quality": quality,
        })
    except Exception as e:
        logging.warning(f"Failed to save to Cosmos: {e}")


def get_history(session_id: str, limit: int = 5) -> list:
    container = _get_container()
    if container is None:
        return []
    try:
        query = "SELECT c.question, c.response, c.route FROM c WHERE c.session_id = @sid AND NOT IS_DEFINED(c.type) ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        items = list(container.query_items(
            query=query,
            parameters=[
                {"name": "@sid", "value": session_id},
                {"name": "@limit", "value": limit},
            ],
            enable_cross_partition_query=False,
            partition_key=session_id,
        ))
        return list(reversed(items))
    except Exception as e:
        logging.warning(f"Failed to read from Cosmos: {e}")
        return []
