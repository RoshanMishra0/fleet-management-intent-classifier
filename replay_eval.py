"""Replay recorded model outputs through the rule layer.

Reads the per-case predictions saved by evaluate.py in metrics_report.json
(the raw model outputs from before the rule layer existed) and applies
pre_classify / postprocess to each one. This measures what the rule layer
changes without re-running the model.

It does NOT replace a fresh run: after changing the service, start the server
and run evaluate.py again, ideally on a held-out test set.
"""

import json

from postprocess import pre_classify, postprocess


def normalize_value(value):
    if isinstance(value, list):
        return sorted(value)
    return value


def score(results, predict):
    total = len(results)
    intent_ok = exact_ok = slots = slots_ok = 0
    failures = []
    for r in results:
        expected = r["expected"]
        prediction = predict(r)
        if prediction.get("intent") == expected.get("intent"):
            intent_ok += 1
        case_ok = 0
        for key, value in expected.items():
            slots += 1
            if normalize_value(prediction.get(key)) == normalize_value(value):
                slots_ok += 1
                case_ok += 1
        if case_ok == len(expected):
            exact_ok += 1
        else:
            failures.append({"query": r["query"], "expected": expected, "prediction": prediction})
    return {
        "total_test_cases": total,
        "intent_accuracy": round(intent_ok / total, 4),
        "slot_accuracy": round(slots_ok / slots, 4),
        "exact_match_accuracy": round(exact_ok / total, 4),
    }, failures


def main():
    with open("metrics_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    results = [r for r in report["detailed_results"] if "prediction" in r]

    skipped = 0

    def with_rules(r):
        nonlocal skipped
        ruled = pre_classify(r["query"])
        if ruled is not None:
            skipped += 1
            return ruled
        return postprocess(r["query"], r["prediction"])

    before, _ = score(results, lambda r: r["prediction"])
    after, failures = score(results, with_rules)
    after["answered_without_model"] = skipped

    print("Model output only:   ", json.dumps(before))
    print("With the rule layer: ", json.dumps(after))
    if failures:
        print("\nSTILL FAILING:")
        for f in failures:
            print("-" * 80)
            print("Query:", f["query"])
            print("Expected:", f["expected"])
            print("Prediction:", f["prediction"])

    with open("metrics_replay.json", "w", encoding="utf-8") as f:
        json.dump({"model_output_only": before, "with_rule_layer": after, "failures": failures}, f, indent=2)


if __name__ == "__main__":
    main()
