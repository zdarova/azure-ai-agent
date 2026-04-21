"""Observability - tracks agent latency, token usage, and call counts."""

import time
import logging
from collections import defaultdict
from functools import wraps

_metrics = defaultdict(lambda: {"calls": 0, "total_ms": 0, "errors": 0, "last_ms": 0})


def track(agent_name: str):
    """Decorator to track agent execution metrics."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            try:
                result = fn(state)
                elapsed = (time.perf_counter() - start) * 1000
                _metrics[agent_name]["calls"] += 1
                _metrics[agent_name]["total_ms"] += elapsed
                _metrics[agent_name]["last_ms"] = round(elapsed)
                logging.info(f"[{agent_name}] {elapsed:.0f}ms")
                return result
            except Exception as e:
                _metrics[agent_name]["errors"] += 1
                raise
        return wrapper
    return decorator


def get_metrics() -> dict:
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
    """Build a trace summary from the current metrics."""
    return {
        "agents": get_metrics(),
        "total_calls": sum(m["calls"] for m in _metrics.values()),
        "total_errors": sum(m["errors"] for m in _metrics.values()),
    }
