"""
Fact extraction evaluation script.

WHAT THIS MEASURES, PER FIELD (preferred_size, preferred_color, budget_max,
mentioned_products):
  - Precision: of the facts the extractor claimed, how many were correct?
  - Recall:    of the facts actually present, how many did the extractor find?
  - False positives specifically matter here more than in a typical
    classification eval: an extractor that INVENTS a fact silently
    corrupts customer memory. This script reports false positives (the
    "hallucination count") as its own explicit number, not buried inside
    precision, because a low hallucination count is arguably more
    important than a high recall for this component's safety.

WHY NORMALIZATION IS APPLIED BEFORE SCORING:
The model may return a correct fact in a different but equivalent surface
form ("large" vs "L"). Scoring raw string equality would unfairly penalize
correct extractions. Normalization (lowercasing, a small size-alias map,
naive de-pluralization for product names) is applied ONLY in this eval
script, never in the application code — it exists purely to make scoring
fair, documented here so the methodology is visible.

WHY THIS MAKES REAL NETWORK CALLS AND WHY THAT MATTERS FOR INTERPRETING
RESULTS:
Unlike test_fact_extractor.py (which tests the deterministic JSON parser
against synthetic model output), this script calls the REAL extract_facts()
function, which hits OpenRouter. This means:
  - Results are NOT perfectly deterministic between runs.
  - A missed fact could mean the model genuinely failed to extract it, OR
    that the API call itself failed (rate limit, timeout) and returned {}.
  - fact_extractor.py now has retry + failover across the model chain
    (same pattern as response_generator.py), so a single rate-limited
    model should no longer silently tank recall the way it could before.

HOW TO RUN:
    python -m eval.evaluate_fact_extraction
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.memory.fact_extractor import extract_facts

_DATA_PATH = Path(__file__).parent / "data" / "fact_extraction_eval.json"
_RESULTS_DIR = Path(__file__).parent / "results"

_SCALAR_FIELDS = ["preferred_size", "preferred_color", "budget_max"]
_LIST_FIELD = "mentioned_products"

_SIZE_ALIASES = {
    "small": "s",
    "medium": "m",
    "large": "l",
    "extra large": "xl",
    "extra-large": "xl",
    "xlarge": "xl",
    "x-large": "xl",
}


def _normalize_size(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    return _SIZE_ALIASES.get(v, v)


def _normalize_color(value):
    if value is None:
        return None
    return str(value).strip().lower()


def _normalize_product(value):
    v = str(value).strip().lower()
    return v[:-1] if v.endswith("s") and len(v) > 3 else v


def _normalize_budget(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_NORMALIZERS = {
    "preferred_size": _normalize_size,
    "preferred_color": _normalize_color,
    "budget_max": _normalize_budget,
}


def run_evaluation() -> None:
    with open(_DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    examples = dataset["examples"]
    print(f"Loaded {len(examples)} labeled examples from {_DATA_PATH.name}")
    print("Calling extract_facts() for each example (real API calls)...\n")

    counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in _SCALAR_FIELDS + [_LIST_FIELD]}
    per_example_results = []

    for i, example in enumerate(examples, start=1):
        text = example["text"]
        expected = example["expected"]

        got = extract_facts(text)

        print(f"({i}/{len(examples)}) {text!r}")
        print(f"    expected: {expected}")
        print(f"    got:      {got}")

        example_detail = {"text": text, "expected": expected, "got": got, "fields": {}}

        for field in _SCALAR_FIELDS:
            norm = _NORMALIZERS[field]
            exp_val = norm(expected.get(field))
            got_val = norm(got.get(field))

            if exp_val is not None and got_val == exp_val:
                verdict = "tp"
            elif exp_val is not None:
                verdict = "fn"
            elif got_val is not None:
                verdict = "fp"
            else:
                verdict = "tn"

            if verdict in ("tp", "fp", "fn"):
                counts[field][verdict] += 1
            example_detail["fields"][field] = verdict

        exp_products = {_normalize_product(p) for p in (expected.get(_LIST_FIELD) or [])}
        got_products = {_normalize_product(p) for p in (got.get(_LIST_FIELD) or [])}
        tp = len(exp_products & got_products)
        fp = len(got_products - exp_products)
        fn = len(exp_products - got_products)
        counts[_LIST_FIELD]["tp"] += tp
        counts[_LIST_FIELD]["fp"] += fp
        counts[_LIST_FIELD]["fn"] += fn
        example_detail["fields"][_LIST_FIELD] = {"tp": tp, "fp": fp, "fn": fn}

        per_example_results.append(example_detail)
        print()

    print("=" * 60)
    print("RESULTS  (per field)")
    print("=" * 60)

    total_fp = 0
    field_summary = {}

    for field in _SCALAR_FIELDS + [_LIST_FIELD]:
        tp, fp, fn = counts[field]["tp"], counts[field]["fp"], counts[field]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        total_fp += fp

        p_str = f"{precision:.2f}" if precision is not None else "n/a"
        r_str = f"{recall:.2f}" if recall is not None else "n/a"
        print(f"{field:>20}: precision={p_str}  recall={r_str}  (tp={tp}, fp={fp}, fn={fn})")

        field_summary[field] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}

    print(f"\nTotal hallucinations (false positives across all fields): {total_fp}")
    print("A hallucination here means a fact was extracted that the message did not state.")

    _save_results(field_summary, total_fp, per_example_results)


def _save_results(field_summary, total_fp, per_example_results) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = _RESULTS_DIR / f"fact_extraction_eval_{timestamp}.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_examples": len(per_example_results),
        "field_summary": field_summary,
        "total_hallucinations": total_fp,
        "per_example_results": per_example_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()