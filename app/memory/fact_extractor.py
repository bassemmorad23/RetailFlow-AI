"""
Fact extraction from a single customer message.

Returns {"hard_constraints": {...}, "soft_preferences": {...}} with only
known fields, normalized and validated. Failures return empty buckets and
never break the conversation.

- Fields = industry fields + built-in fields (e.g. `price` = budget)
- Numbers are read from any text: "500 EGP", "٥٠٠ جنيه", "1,500", "8 GB"
- Lists are allowed ("red or blue" -> ["red", "blue"]); each item validated
- merge_facts(): a newly stated field replaces that field in BOTH buckets,
  so customers can change their mind ("actually, size L")

Resilience: per-model retries for transient errors, then chain failover.
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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.industries.fields import get_fields_for_industry

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_EMPTY = {"hard_constraints": {}, "soft_preferences": {}}

_client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=settings.OPENROUTER_API_KEY, timeout=30.0)


def _empty() -> dict[str, dict[str, Any]]:
    return {"hard_constraints": {}, "soft_preferences": {}}


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------

def _build_extraction_prompt(industry_id: str) -> str:
    field_lines = []
    for f in get_fields_for_industry(industry_id):
        line = f"- {f.canonical_name}: {f.field_type}"
        if f.unit:
            line += f" ({f.unit})"
        if f.allowed_values:
            line += f" — one of {f.allowed_values}"
        if f.extraction_hint:
            line += f". {f.extraction_hint}"
        field_lines.append(line)
    fields_block = "\n".join(field_lines)

    return f"""You are an extraction assistant. Read the customer's message
and extract facts they stated about their preferences. The message may be
in English, Arabic or Franco-Arabic.

For each fact, classify it as HARD or SOFT:
- HARD = mandatory constraint. Words like "must", "need", "only",
  "at least", "under X", "no more than". Filter must enforce it.
- SOFT = preference that influences ranking but doesn't exclude.
  Words like "prefer", "would like", "if possible", "nice to have".

Return a valid JSON object with this exact structure:
{{
  "hard_constraints": {{ ... }},
  "soft_preferences": {{ ... }}
}}

Where each inner dict may include any of these keys:
{fields_block}

Rules:
- Only include a key if the message clearly states it.
- Do NOT guess. Do NOT invent.
- If the customer names alternatives ("red or blue"), use a JSON list.
- If unsure whether hard or soft, treat as soft (safer — no exclusion).
- If nothing is stated, return: {{"hard_constraints": {{}}, "soft_preferences": {{}}}}
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
    return isinstance(exc, APIError)


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
def _call_single_model(model: str, text: str, industry_id: str) -> tuple[str, dict]:
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _build_extraction_prompt(industry_id)},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    if not response.choices:
        raise ValueError(f"Model {model} returned no choices in response.")

    reply_text = (response.choices[0].message.content or "").strip()
    usage: dict = {}
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return reply_text, usage


# ---------------------------------------------------------------------------
# PUBLIC
# ---------------------------------------------------------------------------

def extract_facts(text: str, industry_id: str | None) -> dict[str, Any]:
    """Extract stated facts from one customer message. Never raises."""
    if industry_id is None or not text or not text.strip():
        return _empty()

    for model in settings.response_model_chain:
        started = time.monotonic()
        try:
            raw, usage = _call_single_model(model, text, industry_id)
            parsed = _parse_facts(raw, industry_id)
            logger.info("Facts extracted", extra={
                "model": model,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "fact_count": len(parsed["hard_constraints"]) + len(parsed["soft_preferences"]),
                "total_tokens": usage.get("total_tokens"),
            })
            return parsed
        except Exception as exc:
            logger.warning("Fact extraction model failed, trying next in chain", extra={
                "failed_model": model,
                "failure_reason": _classify_failure(exc),
                "failure_detail": str(exc)[:200],
                "latency_ms": int((time.monotonic() - started) * 1000),
            })

    logger.error("All models in chain failed for fact extraction; returning {}",
                 extra={"chain_length": len(settings.response_model_chain)})
    return _empty()


def merge_facts(
    old_hard: dict[str, Any], old_soft: dict[str, Any], new: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Merge newly extracted facts into memory. A field stated again replaces
    the previous value in BOTH buckets (the customer changed their mind).
    """
    new_hard = new.get("hard_constraints", {}) or {}
    new_soft = new.get("soft_preferences", {}) or {}
    restated = set(new_hard) | set(new_soft)

    hard = {k: v for k, v in (old_hard or {}).items() if k not in restated}
    soft = {k: v for k, v in (old_soft or {}).items() if k not in restated}
    hard.update(new_hard)
    soft.update(new_soft)
    return hard, soft


# ---------------------------------------------------------------------------
# VALUE NORMALIZATION & VALIDATION
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")  # \d also matches Arabic-Indic digits


def _parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = _NUMBER_RE.search(value)
    if match is None:
        return None
    token = match.group(0)
    # "1,500" -> thousands separator; "6.1" -> decimal
    token = token.replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def _normalize_value(value: Any, field) -> Any:
    """Coerce a raw scalar to the field's type. None if impossible."""
    rules = field.normalization_rules or {}

    if field.field_type == "int":
        number = _parse_number(value)
        return int(number) if number is not None else None

    if field.field_type == "float":
        return _parse_number(value)

    if field.field_type == "boolean":
        return value if isinstance(value, bool) else None

    if field.field_type in ("string", "enum"):
        if not isinstance(value, str):
            return None
        s = value.strip() if rules.get("trim", True) else value

        case = rules.get("case")
        if case == "lower":
            s = s.lower()
        elif case == "upper":
            s = s.upper()
        elif case == "title":
            s = s.title()

        for alias, canonical in (rules.get("aliases") or {}).items():
            if alias.lower() == s.lower():
                s = canonical
                break
        return s or None

    return None  # unknown field_type: refuse rather than guess


def _validate_value(value: Any, field) -> bool:
    rules = field.validation_rules or {}

    if field.field_type == "enum" and field.allowed_values and value not in field.allowed_values:
        return False
    if field.field_type in ("int", "float"):
        if "min" in rules and value < rules["min"]:
            return False
        if "max" in rules and value > rules["max"]:
            return False
    if field.field_type == "string":
        if "min_length" in rules and len(value) < rules["min_length"]:
            return False
        if "max_length" in rules and len(value) > rules["max_length"]:
            return False
    return True


def _clean_value(value: Any, field) -> Any:
    """Normalize + validate a scalar or a list. None if nothing valid remains."""
    if isinstance(value, list):
        items: list = []
        for item in value:
            cleaned = _clean_value(item, field)
            if cleaned is not None and not isinstance(cleaned, list) and cleaned not in items:
                items.append(cleaned)
        if not items:
            return None
        return items[0] if len(items) == 1 else items

    normalized = _normalize_value(value, field)
    if normalized is None or not _validate_value(normalized, field):
        return None
    return normalized


# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------

def _parse_facts(raw: str, industry_id: str) -> dict[str, dict[str, Any]]:
    if not raw:
        return _empty()

    cleaned = re.sub(r"^```(?:json)?", "", raw.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match is None:
        return _empty()
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _empty()
    if not isinstance(parsed, dict):
        return _empty()

    fields = {f.canonical_name: f for f in get_fields_for_industry(industry_id)}

    def bucket(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        out: dict[str, Any] = {}
        for key, value in data.items():
            field = fields.get(key)
            if field is None:
                continue
            cleaned_value = _clean_value(value, field)
            if cleaned_value is not None:
                out[key] = cleaned_value
        return out

    return {
        "hard_constraints": bucket(parsed.get("hard_constraints", {})),
        "soft_preferences": bucket(parsed.get("soft_preferences", {})),
    }