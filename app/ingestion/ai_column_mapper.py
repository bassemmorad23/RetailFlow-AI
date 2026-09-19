"""
AI-assisted column mapping — fallback for columns the deterministic
mapper couldn't handle.

WHY THIS EXISTS:
Merchants export from many tools. Column names vary wildly:
"Storage Space GB", "colour", "size (EU)", "المقاس". Deterministic
aliases catch most, but not all. LLM handles the tail semantically.

WHY DETERMINISTIC FIRST:
- Known columns (name, price, sku) don't need LLM — instant match
- LLM only runs on the unmapped tail
- Cheaper, faster, more reliable

WHY BEST-EFFORT (never raises):
Ingestion must succeed even if the LLM is down. AI mapping is
enrichment, not a hard dependency.
"""

import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.config import settings
from app.industries.registry import get_fields_for_industry


logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_client = OpenAI(
    base_url=_OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
    timeout=30.0,
)

_MAX_SAMPLES_PER_COLUMN = 3


_BUILTIN_FIELDS = [
    ("product_id", "Unique identifier per product (SKU, ID, or code)"),
    ("name", "Product name or title"),
    ("description", "Product description or details"),
    ("price", "Product price in the store's currency"),
    ("category", "Product category or type"),
    ("image_url", "URL to product image"),
]


def ai_map_columns(
    unmapped_columns: list[str],
    rows: list[dict[str, Any]],
    industry_id: str,
) -> dict[str, str]:
    """
    Ask LLM to map unmapped source columns to canonical industry fields.

    Returns {source_column: canonical_field} for columns the LLM was
    confident about. Columns the LLM can't confidently map are omitted.
    """
    if not unmapped_columns or not rows or industry_id is None:
        return {}

    try:
        fields = get_fields_for_industry(industry_id)
    except Exception:
        logger.warning("Could not load industry fields for %s", industry_id)
        return {}

    if not fields:
        return {}

    # Sample values per unmapped column (helps LLM disambiguate)
    samples = _collect_samples(unmapped_columns, rows)

    prompt = _build_prompt(unmapped_columns, fields, samples)

    for model in settings.response_model_chain:
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            parsed = _parse_mapping(raw)
            valid = _validate_mapping(parsed, unmapped_columns, fields)
            if valid:
                logger.info(
                    "AI mapping succeeded: %d columns mapped via LLM",
                    len(valid),
                )
                return valid
        except Exception as exc:
            logger.warning(
                "AI mapping failed on model %s, trying next",
                model,
                extra={"error": str(exc)[:200]},
            )
            continue

    logger.warning("All models failed for AI mapping; returning empty")
    return {}


_SYSTEM_PROMPT = """You map merchant CSV column names to canonical product fields.

Rules:
- Return ONLY a JSON object. No prose, no code fences, no explanation.
- Keys = merchant's source column names (exact, as given).
- Values = canonical field names from the provided list, or null.
- If uncertain about a mapping, use null. Never guess.
- Never invent field names not in the provided list.
"""


def _build_prompt(
    unmapped_columns: list[str],
    fields: list,
    samples: dict[str, list],
) -> str:
    lines = []
    lines.append("Canonical fields (with descriptions):")
    for name, desc in _BUILTIN_FIELDS:
        lines.append(f"  - {name}: {desc}")
    for f in fields:
        lines.append(f"  - {f.canonical_name}: {f.description}")

    lines.append("\nUnmapped source columns from merchant (with sample values):")
    for col in unmapped_columns:
        col_samples = samples.get(col, [])
        sample_str = ", ".join(str(s) for s in col_samples[:_MAX_SAMPLES_PER_COLUMN])
        lines.append(f"  - '{col}' — samples: [{sample_str}]")

    lines.append("\nReturn JSON mapping each source column to a canonical field or null.")
    lines.append('Example: {"Storage Space GB": "storage_gb", "Weird Column": null}')

    return "\n".join(lines)


def _collect_samples(
    columns: list[str], rows: list[dict[str, Any]]
) -> dict[str, list]:
    samples: dict[str, list] = {c: [] for c in columns}
    for row in rows[:20]:  # scan up to 20 rows for samples
        for col in columns:
            val = row.get(col)
            if val is not None and str(val).strip():
                if len(samples[col]) < _MAX_SAMPLES_PER_COLUMN:
                    samples[col].append(val)
    return samples


def _parse_mapping(raw: str) -> dict:
    """Extract JSON object from LLM output. Handles code fences, prose."""
    if not raw:
        return {}

    cleaned = re.sub(r"^```(?:json)?", "", raw.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match is None:
        return {}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _validate_mapping(
    parsed: dict, unmapped_columns: list[str], fields: list
) -> dict[str, str]:
    """
    Drop any mapping that:
    - references a source column we didn't ask about
    - maps to a canonical field name not in industry config
    - has null/empty target (LLM couldn't confidently map)
    """
    
    valid_targets = {f.canonical_name for f in fields}
    valid_targets.update(name for name, _ in _BUILTIN_FIELDS)  # ← ADD THIS LINE
    unmapped_set = set(unmapped_columns)

    result: dict[str, str] = {}
    for src, tgt in parsed.items():
        if src not in unmapped_set:
            continue
        if tgt is None or not isinstance(tgt, str):
            continue
        if tgt not in valid_targets:
            logger.warning("LLM invented unknown field: %s", tgt)
            continue
        result[src] = tgt
    return result