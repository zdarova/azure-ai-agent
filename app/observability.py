"""Observability - tracks agent latency with Cosmos DB persistence."""

import time
import logging
from collections import defaultdict
from functools import wraps

_metrics = defaultdict(lambda: {"calls": 0, "total_ms": 0, "errors": 0, "last_ms": 0})
_loaded = False
_COSMOS_DOC_ID = "agent-metrics-global"


def _get_container():
    try:
        from memory import _get_container as get_cosmos
        return get_cosmos()
    except Exception:
        return None


def _load_from_cosmos():
    global _metrics, _loaded
    if _loaded:
        return
    _loaded = True
    container = _get_container()
    if container is None:
        return
    try:
        doc = container.read_item(_COSMOS_DOC_ID, partition_key="__metrics__")
        for name, m in doc.get("agents", {}).items():
            _metrics[name]["calls"] = m.get("calls", 0)
            _metrics[name]["total_ms"] = m.get("total_ms", 0)
            _metrics[name]["errors"] = m.get("errors", 0)
            _metrics[name]["last_ms"] = m.get("last_ms", 0)
        logging.info(f"Loaded metrics from Cosmos DB ({len(_metrics)} agents)")
    except Exception:
        pass


def _save_to_cosmos():
    container = _get_container()
    if container is None:
        return
    try:
        container.upsert_item({
            "id": _COSMOS_DOC_ID,
            "session_id": "__metrics__",
            "type": "metrics",
            "agents": {name: dict(m) for name, m in _metrics.items()},
        })
    except Exception as e:
        logging.warning(f"Failed to save metrics: {e}")


def track(agent_name: str):
    """Decorator to track agent execution metrics."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(state):
            _load_from_cosmos()
            start = time.perf_counter()
            try:
                result = fn(state)
                elapsed = (time.perf_counter() - start) * 1000
                _metrics[agent_name]["calls"] += 1
                _metrics[agent_name]["total_ms"] += elapsed
                _metrics[agent_name]["last_ms"] = round(elapsed)
                logging.info(f"[{agent_name}] {elapsed:.0f}ms")
                _save_to_cosmos()
                return result
            except Exception as e:
                _metrics[agent_name]["errors"] += 1
                _save_to_cosmos()
                raise
        return wrapper
    return decorator


def get_metrics() -> dict:
    _load_from_cosmos()
    result = {}
    for name, m in _metrics.items():
        avg = round(m["total_ms"] / m["calls"]) if m["calls"] > 0 else 0
        result[name] = {
            "calls": m["calls"],
            "avg_ms": avg,
            "last_ms": m["last_ms"],
            "errors": m["errors"],
        }
    return result


def get_trace(state: dict) -> dict:
    return {
        "agents": get_metrics(),
        "total_calls": sum(m["calls"] for m in _metrics.values()),
        "total_errors": sum(m["errors"] for m in _metrics.values()),
    }
