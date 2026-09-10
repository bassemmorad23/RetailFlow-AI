"""
FastAPI application — HTTP entrypoint for the agent.

WHY THIS IS A SEPARATE FILE FROM main.py:
main.py is the CLI entrypoint (a dev/testing tool). This is the HTTP
entrypoint. Both are thin wrappers around the SAME orchestrator —
handle_message() is called identically by both. Keeping them in separate
files means running the API never depends on CLI code and vice versa.

WHY SYNC ENDPOINTS (def, not async def):
The pipeline underneath is synchronous (pymongo, local transformer models,
blocking HTTP to OpenRouter). FastAPI runs plain `def` endpoints in a
threadpool automatically, so blocking code doesn't freeze the server.
This lets us expose the fully-tested sync pipeline over HTTP with ZERO
changes to any module below this layer. Converting to true async is a
later step, justified by real traffic — not now.

WHY THE SCHEMAS ARE REUSED AS-IS:
CustomerMessage and AgentReply are already Pydantic models. FastAPI uses
Pydantic natively for request and response bodies, so they plug straight
in — request validation (including all the input-validation rules we added)
happens automatically, and responses are serialized automatically.

CORS:
The widget will be loaded from many different store websites, each on
its own domain. Browsers block cross-origin requests unless the API
explicitly allows them. `allow_origins=["*"]` is fine for development;
tighten this to the real store domains before onboarding real customers.
"""

import logging

from fastapi import FastAPI , Request , Depends
from fastapi.middleware.cors import CORSMiddleware

from app.core.orchestrator import handle_message
from app.logging_config import setup_logging
from app.schemas.models import AgentReply, CustomerMessage
from app.rate_limiter import rate_limit
import uuid
from app.log_context import set_request_context, clear_request_context
import sentry_sdk
from app.config import settings
import time
from fastapi.responses import JSONResponse
from app.metrics import record_request, get_metrics
from app.log_context import set_request_context, clear_request_context, get_request_context




if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        send_default_pii=False,
        traces_sample_rate=0.1 if settings.APP_ENV == "production" else 1.0,
    )


# Configure logging once, at import time, before any request is served.
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="StoreFlow AI",
    description="AI sales agent for clothing stores.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """
    Per-request setup:
      - Assign a unique request_id (visible in logs and returned via header)
      - Clear context on the way out so nothing leaks to the next request

    Note: Metrics are recorded inside the /chat endpoint itself, not here,
    because store_id is only known once the request body is parsed by
    FastAPI — and contextvars set inside sync endpoints don't propagate
    back to this async middleware.
    """
    request_id = str(uuid.uuid4())
    set_request_context(request_id=request_id)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_context()



@app.get("/health")
def health() -> dict:
    """
    Liveness check. Returns 200 if the service is up.
    Used by load balancers, uptime monitors, and deployment health checks.
    Deliberately does NOT touch MongoDB or any model — it only answers
    "is the web server responding?", which is what a liveness probe needs.
    """
    return {"status": "ok"}

@app.get("/metrics")
def metrics() -> dict:
    """
    Basic operational metrics: request count, latency, error rate.
    Public (no auth) — same pattern as Prometheus. Contains only counts
    and latencies, never customer data.
    """
    return get_metrics()


@app.post("/chat", response_model=AgentReply)
def chat(request: Request, message: CustomerMessage, _rate_limit: None = Depends(rate_limit)) -> AgentReply:
    """
    Main endpoint: receive a customer message, return the agent's reply.
    """
    set_request_context(
        store_id=message.store_id,
        conversation_id=message.conversation_id,
    )
    logger.info("Received message", extra={"channel": message.channel})

    started = time.monotonic()
    status_code = 200
    try:
        return handle_message(message)
    except Exception:
        status_code = 500
        raise
    finally:
        latency_ms = (time.monotonic() - started) * 1000
        record_request(status_code=status_code, store_id=message.store_id, latency_ms=latency_ms)
