"""
Fact extraction from a single customer message.

WHY THIS EXISTS:
Alongside the visible reply, we quietly build up a small profile of what
the customer said they want — their size, color preference, budget, and
which products they've mentioned. Doing this with a compact prompt to a
capable LLM handles both Arabic and English input without maintaining
regex/keyword rules per language.

WHY THIS RETURNS A DICT[str, ANY]:
The extractor returns a raw dict rather than a KnownFacts object because
the orchestrator layer decides how to MERGE the new facts with existing
memory. Keeping merge policy out of this module keeps extraction pure —
"look at the message, output what was stated." Nothing else.

WHY WE CHECK EACH FIELD BEFORE USING IT:
The LLM may return partial JSON, extra keys, or the wrong type. We only
accept known keys and validate types before returning them. Anything
malformed silently becomes {} — never a crash, never corrupted memory.

RESILIENCE:
Same two-layer model as response_generator.py:
  1. Per-model retries with backoff for transient errors
  2. Chain-level failover if a whole model is unusable

WHY A FAILURE HERE RETURNS {} INSTEAD OF RAISING:
Fact extraction is a background enrichment step — a failure should not
break the customer conversation. If we can't extract facts this turn,
we return {} and keep going. Memory just doesn't grow this turn.

STRUCTURED LOGGING NOTES:
Successful extraction: one log line with model + latency_ms + fact count.
Failed model call: one log line with failed_model + reason + latency,
before moving to the next model.
"""

import json
import logging
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings


logger = logging.getLogger(__name__)


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_ALLOWED_FIELDS = {
    "preferred_size",
    "preferred_color",
    "budget_max",
    "mentioned_products",
}

_client = OpenAI(
    base_url=_OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
    timeout=30.0,
)

_SYSTEM_PROMPT = """You are an extraction assistant. Read the customer's message
and extract only the facts they explicitly stated about their preferences.

Return a valid JSON object with any of these optional keys:
- preferred_size: string (e.g., "M", "L", "XL")
- preferred_color: string (lowercase, e.g., "red")
- budget_max: number (in the store's currency)
- mentioned_products: list of product-type strings the customer named

Rules:
- Only include a key if the message clearly states it.
- Do NOT guess. Do NOT invent.
- If nothing is stated, return exactly: {}
- Output ONLY the JSON. No prose, no code fences.
"""


# ---------------------------------------------------------------------------
# LOW-LEVEL: single-model call with retry
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return 500 <= exc.status_code < 600
    if isinstance(exc, APIError):
        return True
    return False


def _classify_failure(exc: BaseException) -> str:
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, APIConnectionError):
        return "connection_error"
    if isinstance(exc, APIStatusError):
        return f"http_{exc.status_code}"
    if isinstance(exc, ValueError):
        return "empty_response"
    if isinstance(exc, APIError):
        return "api_error"
    return type(exc).__name__


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
)
def _call_single_model(model: str, text: str) -> str:
    """One extraction call to one model, with retry-on-transient-error."""
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    if not response.choices:
        raise ValueError(f"Model {model} returned no choices in response.")

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# PUBLIC
# ---------------------------------------------------------------------------

def extract_facts(text: str) -> dict[str, Any]:
    """
    Extract stated facts from a single customer message. Walks the
    configured model chain and stops at the first success.
    """
    if not text or not text.strip():
        return {}

    for model in settings.response_model_chain:
        started = time.monotonic()
        try:
            raw = _call_single_model(model, text)
            latency_ms = int((time.monotonic() - started) * 1000)

            parsed = _parse_facts(raw)
            logger.info(
                "Facts extracted",
                extra={
                    "model": model,
                    "latency_ms": latency_ms,
                    "fact_count": len(parsed),
                },
            )
            return parsed

        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "Fact extraction model failed, trying next in chain",
                extra={
                    "failed_model": model,
                    "failure_reason": _classify_failure(exc),
                    "failure_detail": str(exc)[:200],
                    "latency_ms": latency_ms,
                },
            )
            continue

    logger.error(
        "All models in chain failed for fact extraction; returning {}",
        extra={"chain_length": len(settings.response_model_chain)},
    )
    return {}


# ---------------------------------------------------------------------------
# PARSING HELPERS
# ---------------------------------------------------------------------------

def _parse_facts(raw: str) -> dict[str, Any]:
    """
    Turn the model's raw text into a safe dict. Handles common failure
    modes (code fences, prose leading up to JSON, malformed output) by
    silently returning {} rather than raising.
    """
    if not raw:
        return {}

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match is None:
        logger.debug("Fact extraction returned non-JSON; ignoring")
        return {}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.debug("Fact extraction returned malformed JSON; ignoring")
        return {}

    if not isinstance(parsed, dict):
        return {}

    result: dict[str, Any] = {}
    for key, value in parsed.items():
        if key not in _ALLOWED_FIELDS:
            continue
        if key == "mentioned_products":
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                result[key] = value
        elif key == "budget_max":
            if isinstance(value, (int, float)):
                result[key] = float(value)
        elif key in {"preferred_size", "preferred_color"}:
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()

    return result