import logging 
from fastapi import FastAPI, Request , HTTPException 


from app.core.orchestrator import handle_message
from app.logging_config import setup_logging
from app.schemas.models import CustomerMessage, AgentReply





app=FastAPI(
    title="StoreFlow AI",
    description="AI sales agent for clothing stores.",
    version="0.1.0",

)
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
"""

import logging

from fastapi import FastAPI

from app.core.orchestrator import handle_message
from app import logging_config
from app.schemas.models import AgentReply, CustomerMessage

# Configure logging once, at import time, before any request is served.
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="StoreFlow AI",
    description="AI sales agent for clothing stores.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """
    Liveness check. Returns 200 if the service is up.
    Used by load balancers, uptime monitors, and deployment health checks.
    Deliberately does NOT touch MongoDB or any model — it only answers
    "is the web server responding?", which is what a liveness probe needs.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=AgentReply)
def chat(message: CustomerMessage) -> AgentReply:
    """
    Main endpoint: receive a customer message, return the agent's reply.

    FastAPI automatically:
      - parses the JSON body into a CustomerMessage (running all our
        input-validation rules; invalid input returns 422 without ever
        reaching the pipeline)
      - serializes the returned AgentReply back to JSON

    The orchestrator is called exactly as the CLI calls it — this endpoint
    adds no business logic of its own.
    """
    logger.info(
        "Received message on conversation %s (channel=%s)",
        message.conversation_id,
        message.channel,
    )
    return handle_message(message)

