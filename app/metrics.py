"""
In-memory metrics for /metrics endpoint.

WHY IN-MEMORY:
Same reasoning as rate_limiter.py — one Railway instance, one process,
counters shared naturally. Fast, zero dependencies. Resets when the
container restarts, which is fine for basic operational health (Sentry
covers cross-restart error visibility).

WHY THREAD-SAFE:
FastAPI runs sync endpoints in a threadpool, so multiple requests can
update these counters concurrently. A Lock guarantees no lost updates
or corrupt reads.

WHY BOUNDED LATENCY HISTORY:
Keeping every latency ever recorded would grow memory forever. A deque
with maxlen=1000 keeps the last 1000 requests only — enough for a
stable p95 estimate, tiny memory footprint (~8KB).

WHAT WE DO NOT TRACK HERE:
- CPU/memory (Railway shows this in its dashboard)
- Per-endpoint breakdowns (overkill for v1, we only have /chat that matters)
- Historical charts (needs Prometheus/Grafana later)

UPGRADE PATH:
When scaling to multiple instances, replace with Prometheus client
library — the shape of the data (counts, per-status, latency histogram)
is identical, just backed by real Prometheus counters instead of dicts.
"""

import time
from collections import defaultdict, deque
from statistics import mean
from threading import Lock


_started_at = time.monotonic()

_total_requests = 0
_requests_by_status: dict[int, int] = defaultdict(int)
_requests_by_store: dict[str, int] = defaultdict(int)
_latency_history: deque = deque(maxlen=1000)

_lock = Lock()


def record_request(status_code: int, store_id: str | None, latency_ms: float) -> None:
    """Called by middleware after each request finishes."""
    global _total_requests
    with _lock:
        _total_requests += 1
        _requests_by_status[status_code] += 1
        if store_id:
            _requests_by_store[store_id] += 1
        _latency_history.append(latency_ms)


def get_metrics() -> dict:
    """Snapshot the current metrics for the /metrics endpoint."""
    with _lock:
        uptime = int(time.monotonic() - _started_at)
        latencies = list(_latency_history)

        error_count = sum(
            count for status, count in _requests_by_status.items()
            if status >= 500
        )
        error_rate = (error_count / _total_requests) if _total_requests else 0.0

        return {
            "uptime_seconds": uptime,
            "total_requests": _total_requests,
            "requests_by_status": dict(_requests_by_status),
            "requests_by_store": dict(_requests_by_store),
            "avg_latency_ms": int(mean(latencies)) if latencies else 0,
            "p95_latency_ms": int(_percentile(latencies, 95)) if latencies else 0,
            "error_rate": round(error_rate, 4),
            "latency_sample_size": len(latencies),
        }


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile from a list of numbers."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100)
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)