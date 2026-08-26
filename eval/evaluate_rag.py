"""
RAG retrieval evaluation script.

WHAT THIS MEASURES:
For each labeled query, calls the real retrieve_context() and checks
whether a correct document was found, and how highly it ranked:
  - recall@1: was a correct document the TOP result?
  - recall@3: was a correct document anywhere in the top 3?
(computed only over the "positive" examples — ones with a real expected
answer in the catalog)

For "negative" examples (expected_sources == [], meaning nothing in the
catalog should match), this reports whether retrieval returned anything
with a suspiciously high score anyway — a possible false positive.
_CONFIDENT_MATCH_BAR (0.50) is a judgment call for what counts as
"suspiciously confident" for a query that shouldn't match anything; it's
well above the retriever's own MIN_SIMILARITY_SCORE filter (0.35), so a
result clearing this bar is a stronger signal than just "cleared the
noise floor."

WHY THIS RUNS ENTIRELY LOCALLY:
retrieve_context() uses only the local sentence-transformers model —
no network calls, no API cost, no rate limits. Safe to re-run as often
as needed while iterating.

HOW TO RUN:
    python -m eval.evaluate_rag
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.rag.retriever import retrieve_context

_DATA_PATH = Path(__file__).parent / "data" / "rag_eval.json"
_RESULTS_DIR = Path(__file__).parent / "results"

_TOP_K = 5  # retrieve enough to measure both recall@1 and recall@3 from one call
_CONFIDENT_MATCH_BAR = 0.50


def run_evaluation() -> None:
    with open(_DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    examples = dataset["examples"]
    positives = [e for e in examples if e["expected_sources"]]
    negatives = [e for e in examples if not e["expected_sources"]]

    print(f"Loaded {len(examples)} queries ({len(positives)} positive, {len(negatives)} negative)")
    print("Running retrieve_context() for each (local, no network)...\n")

    per_example_results = []
    hit_at_1 = 0
    hit_at_3 = 0
    false_positives = 0

    for i, example in enumerate(examples, start=1):
        text = example["text"]
        expected = set(example["expected_sources"])
        is_negative = len(expected) == 0

        chunks = retrieve_context(text, top_k=_TOP_K)
        returned_sources = [c.source for c in chunks]
        returned_scores = [round(c.score, 3) for c in chunks]

        detail = {
            "text": text,
            "expected_sources": list(expected),
            "returned_sources": returned_sources,
            "returned_scores": returned_scores,
        }

        if is_negative:
            top_score = returned_scores[0] if returned_scores else 0.0
            is_false_positive = top_score >= _CONFIDENT_MATCH_BAR
            if is_false_positive:
                false_positives += 1
            marker = "FALSE-POS" if is_false_positive else "OK       "
            detail["false_positive"] = is_false_positive
            print(f"[{marker}] ({i}/{len(examples)}) {text!r}  (negative example)")
            print(f"           returned: {list(zip(returned_sources, returned_scores))}")
        else:
            rank = next((idx + 1 for idx, s in enumerate(returned_sources) if s in expected), None)
            found_at_1 = rank == 1
            found_at_3 = rank is not None and rank <= 3
            hit_at_1 += int(found_at_1)
            hit_at_3 += int(found_at_3)
            detail["rank_found"] = rank

            marker = "OK  " if found_at_3 else "MISS"
            print(f"[{marker}] ({i}/{len(examples)}) {text!r}")
            print(f"       expected any of: {list(expected)}")
            print(f"       returned: {list(zip(returned_sources, returned_scores))}  (rank={rank})")

        per_example_results.append(detail)
        print()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    n_pos = len(positives)
    recall_at_1 = hit_at_1 / n_pos if n_pos else None
    recall_at_3 = hit_at_3 / n_pos if n_pos else None
    fp_rate = false_positives / len(negatives) if negatives else None

    print(f"\nPositive examples: n={n_pos}")
    print(f"  recall@1: {recall_at_1:.2%}" if recall_at_1 is not None else "  recall@1: n/a")
    print(f"  recall@3: {recall_at_3:.2%}" if recall_at_3 is not None else "  recall@3: n/a")

    print(f"\nNegative examples: n={len(negatives)}")
    print(f"  false positives (score >= {_CONFIDENT_MATCH_BAR}): {false_positives}")
    if fp_rate is not None:
        print(f"  false-positive rate: {fp_rate:.2%}")

    _save_results(recall_at_1, recall_at_3, false_positives, n_pos, len(negatives), per_example_results)


def _save_results(recall_at_1, recall_at_3, false_positives, n_pos, n_neg, per_example_results) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = _RESULTS_DIR / f"rag_eval_{timestamp}.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "false_positives": false_positives,
        "per_example_results": per_example_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()