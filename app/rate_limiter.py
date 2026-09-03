"""
Simple in-memory per-IP rate limiter.

WHY IN-MEMORY (NOT REDIS):
This is v1 running on a single Railway instance. In-memory counters work
correctly when there's only one process serving requests — every request
lands on the same process, so the same counter sees them all.

WHY NOT slowapi:
Tested and confirmed silently ignored on this project's FastAPI/Pydantic
combination. Replaced with a direct implementation because rate limiting
logic is small enough that a proven library isn't worth a mystery bug.

UPGRADE PATH (later):
When we scale to multiple Railway instances, replace the in-memory dict
with Redis — each instance would have its own counter otherwise, and a
'20/minute' cap would silently become '20 * N instances / minute'.
"""

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

# Sliding window: for each IP, we keep timestamps of recent requests.
_requests: dict[str, deque] = defaultdict(deque)
_lock = Lock()

_WINDOW_SECONDS = 150
_MAX_REQUESTS = 20


def rate_limit(request: Request) -> None:
    """
    FastAPI dependency: raises 429 if the caller has exceeded
    _MAX_REQUESTS in the last _WINDOW_SECONDS.
    """
    
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    cutoff = now - _WINDOW_SECONDS

    with _lock:
        timestamps = _requests[ip]

        # Drop timestamps older than the window.
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= _MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down.",
            )

        timestamps.append(now)