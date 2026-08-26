"""
Recommendation evaluation script.

WHAT THIS MEASURES:
For each example, runs the REAL retrieve_context() to get real RAG
results, then feeds them into the REAL recommend_products() along with a
MANUALLY ASSIGNED intent (not the real intent classifier — see below for
why). Checks whether the correct product was recommended, or, for
negative examples, whether NO recommendation was incorrectly produced.

WHY THE INTENT IS MANUALLY ASSIGNED, NOT DETECTED:
detect_intent() has its own measured error rate (see eval/evaluate_intent.py).
Chaining through it here would mean a miss could be caused by EITHER a
recommendation bug OR an intent misclassification, with no way to tell
which from the result alone. Assigning the intent that a message SHOULD
produce isolates recommend_products()'s own correctness — its filtering
logic, its similarity threshold, its source-prefix matching — from
intent-classification noise that already has its own honest number
elsewhere.

WHAT THE NEGATIVE EXAMPLES ARE ACTUALLY TESTING:
  - Non-product queries (shipping, hours): retrieval shouldn't surface a
    "product_"-prefixed chunk at all, so nothing should be recommended.
  - Out-of-catalog queries (shoes): same as the RAG eval's negative check,
    now verified doesn't leak through as a bad recommendation either.
  - The complaint safety case: RAG WILL find a real, relevant product
    (the red dress) for "this red dress arrived damaged" — the important
    thing is that the intent gate blocks a recommendation anyway. This is
    the one case here that's a genuine safety property, not just accuracy.

WHY THIS RUNS ENTIRELY LOCALLY:
Both retrieve_context() and recommend_products() are local, no network
calls involved. Safe and fast to re-run repeatedly.

HOW TO RUN:
    python -m eval.evaluate_recommendation
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.rag.retriever import retrieve_context
from app.recommendation.product_recommender import recommend_products
from app.schemas.models import IntentLabel, IntentResult, KnownFacts, MemoryState

_DATA_PATH = Path(__file__).parent / "data" / "recommendation_eval.json"
_RESULTS_DIR = Path(__file__).parent / "results"

_BLANK_MEMORY = MemoryState(conversation_id="eval", history=[], known_facts=KnownFacts())


def run_evaluation() -> None:
    with open(_DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    examples = dataset["examples"]
    positives = [e for e in examples if e["expected_product_id"] is not None]
    negatives = [e for e in examples if e["expected_product_id"] is None]

    print(f"Loaded {len(examples)} examples ({len(positives)} positive, {len(negatives)} negative)")
    print("Running retrieve_context() + recommend_products() for each (local, no network)...\n")

    per_example_results = []
    correct_recommendations = 0
    unwanted_recommendations = 0

    for i, example in enumerate(examples, start=1):
        text = example["text"]
        expected = example["expected_product_id"]
        intent = IntentResult(label=IntentLabel(example["intent_label"]), confidence=1.0)

        retrieved = retrieve_context(text)
        recommendations = recommend_products(
            intent=intent,
            memory=_BLANK_MEMORY,
            retrieved_context=retrieved,
        )
        recommended_ids = [r.product_id for r in recommendations]

        detail = {
            "text": text,
            "intent_used": example["intent_label"],
            "expected_product_id": expected,
            "recommended_ids": recommended_ids,
            "retrieved_sources": [(c.source, round(c.score, 3)) for c in retrieved],
        }

        if expected is not None:
            is_correct = expected in recommended_ids
            correct_recommendations += int(is_correct)
            marker = "OK  " if is_correct else "MISS"
            detail["correct"] = is_correct
            print(f"[{marker}] ({i}/{len(examples)}) {text!r}  (intent={example['intent_label']})")
            print(f"       expected: {expected}   recommended: {recommended_ids}")
            print(f"       retrieved: {detail['retrieved_sources']}")
        else:
            has_unwanted = len(recommended_ids) > 0
            unwanted_recommendations += int(has_unwanted)
            marker = "UNWANTED" if has_unwanted else "OK      "
            detail["unwanted"] = has_unwanted
            print(f"[{marker}] ({i}/{len(examples)}) {text!r}  (intent={example['intent_label']}, expects NO recommendation)")
            print(f"       recommended: {recommended_ids}")
            print(f"       retrieved: {detail['retrieved_sources']}")

        per_example_results.append(detail)
        print()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    n_pos = len(positives)
    n_neg = len(negatives)
    recall = correct_recommendations / n_pos if n_pos else None
    unwanted_rate = unwanted_recommendations / n_neg if n_neg else None

    print(f"\nPositive examples: n={n_pos}")
    print(f"  correct recommendation rate: {recall:.2%}" if recall is not None else "  n/a")

    print(f"\nNegative examples: n={n_neg}")
    print(f"  unwanted recommendations: {unwanted_recommendations}")
    print(f"  unwanted rate: {unwanted_rate:.2%}" if unwanted_rate is not None else "  n/a")

    _save_results(recall, unwanted_rate, n_pos, n_neg, per_example_results)


def _save_results(recall, unwanted_rate, n_pos, n_neg, per_example_results) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = _RESULTS_DIR / f"recommendation_eval_{timestamp}.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "correct_recommendation_rate": recall,
        "unwanted_recommendation_rate": unwanted_rate,
        "per_example_results": per_example_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()