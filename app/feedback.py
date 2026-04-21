"""Feedback - saves user thumbs up/down ratings to Cosmos DB."""

import logging
from datetime import datetime, timezone
from memory import _get_container


def save_feedback(session_id: str, message_id: str, rating: str):
    """Save feedback (thumbs_up or thumbs_down) to Cosmos DB."""
    container = _get_container()
    if container is None:
        return
    try:
        container.upsert_item({
            "id": f"feedback-{message_id}",
            "session_id": session_id,
            "type": "feedback",
            "message_id": message_id,
            "rating": rating,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logging.warning(f"Failed to save feedback: {e}")


def get_feedback_stats() -> dict:
    """Get aggregate feedback stats."""
    container = _get_container()
    if container is None:
        return {"thumbs_up": 0, "thumbs_down": 0}
    try:
        query = "SELECT VALUE COUNT(1) FROM c WHERE c.type = 'feedback' AND c.rating = @r"
        up = list(container.query_items(query, parameters=[{"name": "@r", "value": "thumbs_up"}], enable_cross_partition_query=True))
        down = list(container.query_items(query, parameters=[{"name": "@r", "value": "thumbs_down"}], enable_cross_partition_query=True))
        return {"thumbs_up": up[0] if up else 0, "thumbs_down": down[0] if down else 0}
    except Exception:
        return {"thumbs_up": 0, "thumbs_down": 0}
