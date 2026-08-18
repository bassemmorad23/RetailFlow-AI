import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from app.emotion.emotion_detector import detect_emotion
from app.schemas.models import EmotionLabel


_DATA_PATH = Path(__file__).parent / "data" / "emotion_eval.json"
_RESULTS_DIR = Path(__file__).parent / "results"

_ALL_LABELS = [label.value for label in EmotionLabel]


def run_evaluation() -> None:
    with open(_DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    examples = dataset["examples"]

    print(f"Loaded {len(examples)} labeled examples from {_DATA_PATH.name}")
    print("Running emotion classifier on each example...\n")

    y_true = []
    y_pred = []
    per_example_results = []

    for i, example in enumerate(examples, start=1):
        text = example["text"]
        expected = example["label"]

        result = detect_emotion(text)
        predicted = result.label.value

        y_true.append(expected)
        y_pred.append(predicted)

        is_correct = predicted == expected
        marker = "OK  " if is_correct else "MISS"

        print(f"[{marker}] ({i}/{len(examples)}) {text!r}")
        print(
            f"       expected={expected} "
            f"predicted={predicted} "
            f"confidence={result.confidence:.2f}"
        )

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

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nOverall accuracy: {accuracy:.2%}  (n={len(examples)})")

    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=_ALL_LABELS,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=expected, columns=predicted):")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=_ALL_LABELS,
    )

    header = "".join(f"{label[:8]:>10}" for label in _ALL_LABELS)

    print(f"{'':>18}{header}")

    for label, row in zip(_ALL_LABELS, cm):
        row_str = "".join(f"{value:>10}" for value in row)
        print(f"{label:>18}{row_str}")

    save_results(
        accuracy,
        y_true,
        y_pred,
        per_example_results,
    )


def save_results(
    accuracy,
    y_true,
    y_pred,
    per_example_results,
) -> None:

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        _RESULTS_DIR
        / f"emotion_eval_{timestamp}.json"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=_ALL_LABELS,
        zero_division=0,
        output_dict=True,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_examples": len(y_true),
        "accuracy": accuracy,
        "per_class_report": report,
        "per_example_results": per_example_results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()