"""
Intent classification evaluation script.

WHY THIS IS SEPARATE FROM pytest:
Unit tests (tests/) assert deterministic pass/fail behavior. This script
measures a MODEL'S accuracy against labeled examples — the output is a
number, not a pass/fail. Different purpose, different tool, so it lives
in its own eval/ directory rather than tests/.

WHAT THIS MEASURES AND WHAT IT DOESN'T:
This evaluates ONLY the intent classifier (detect_intent), in isolation,
against the hand-labeled set in eval/data/intent_eval.json. It says
nothing about emotion accuracy, RAG relevance, or response quality —
those need their own eval sets and scripts, built the same way.

HONESTY NOTE ON THE DATASET:
eval/data/intent_eval.json is a small, hand-crafted starter set (40
examples, 5 per intent label), not extracted from real customer traffic.
Treat this number as a first, honest baseline — not a rigorous benchmark.
Report the sample size (n) alongside any accuracy number quoted anywhere,
including in the README.

HOW TO RUN:
    python -m eval.evaluate_intent

OUTPUT:
    - Printed accuracy, macro F1, and per-class precision/recall/F1
    - Printed confusion matrix
    - A timestamped JSON results file saved to eval/results/, so a
      specific measured run can be cited later (e.g. in the README)
      instead of re-stating a number that may drift as the model or
      dataset changes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from app.intent.intent_detector import detect_intent
from app.schemas.models import IntentLabel

_DATA_PATH = Path(__file__).parent / "data" / "intent_eval.json"
_RESULTS_DIR = Path(__file__).parent / "results"

_ALL_LABELS = [label.value for label in IntentLabel]


def run_evaluation() -> None:
    with open(_DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    examples = dataset["examples"]
    print(f"Loaded {len(examples)} labeled examples from {_DATA_PATH.name}")
    print("Running intent classifier on each example...\n")

    y_true = []
    y_pred = []
    per_example_results = []

    for i, example in enumerate(examples, start=1):
        text = example["text"]
        expected = example["label"]

        result = detect_intent(text)
        predicted = result.label.value

        y_true.append(expected)
        y_pred.append(predicted)

        is_correct = predicted == expected
        marker = "OK  " if is_correct else "MISS"
        print(f"[{marker}] ({i}/{len(examples)}) {text!r}")
        print(f"       expected={expected}  predicted={predicted}  confidence={result.confidence:.2f}")

        per_example_results.append({
            "text": text,
            "expected": expected,
            "predicted": predicted,
            "confidence": result.confidence,
            "correct": is_correct,
        })

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy: {acc:.2%}  (n={len(examples)})")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=_ALL_LABELS, zero_division=0))

    print("Confusion matrix (rows=expected, columns=predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=_ALL_LABELS)
    header = "".join(f"{label[:8]:>10}" for label in _ALL_LABELS)
    print(f"{'':>18}{header}")
    for label, row in zip(_ALL_LABELS, cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{label:>18}{row_str}")

    _save_results(acc, y_true, y_pred, per_example_results)


def _save_results(acc, y_true, y_pred, per_example_results) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = _RESULTS_DIR / f"intent_eval_{timestamp}.json"

    report = classification_report(
        y_true, y_pred, labels=_ALL_LABELS, zero_division=0, output_dict=True
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_examples": len(y_true),
        "accuracy": acc,
        "per_class_report": report,
        "per_example_results": per_example_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()