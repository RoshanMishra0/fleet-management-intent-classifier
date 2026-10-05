"""Metrics shared by the evaluation scripts.

A record is a dict with "query", "expected" and "prediction" (None when the
request failed), plus "latency_seconds" when it was measured. Only the fields
listed in "expected" are compared.
"""


def normalize_value(value):
    if isinstance(value, list):
        return sorted(value)
    return value


def summarize(records: list) -> dict:
    """Score the records in place (adds "exact_match") and return the metrics."""
    total = len(records)
    answered = intent_ok = exact_ok = slots = slots_ok = 0
    latencies = []

    for r in records:
        expected, prediction = r["expected"], r.get("prediction")
        slots += len(expected)
        if prediction is None:
            r["exact_match"] = False
            continue
        answered += 1
        if "latency_seconds" in r:
            latencies.append(r["latency_seconds"])
        if prediction.get("intent") == expected.get("intent"):
            intent_ok += 1
        case_ok = sum(
            normalize_value(prediction.get(key)) == normalize_value(value)
            for key, value in expected.items()
        )
        slots_ok += case_ok
        r["exact_match"] = case_ok == len(expected)
        exact_ok += r["exact_match"]

    metrics = {
        "total_test_cases": total,
        "intent_accuracy": round(intent_ok / total, 4) if total else 0,
        "slot_accuracy": round(slots_ok / slots, 4) if slots else 0,
        "exact_match_accuracy": round(exact_ok / total, 4) if total else 0,
        "valid_json_rate": round(answered / total, 4) if total else 0,
    }
    if latencies:
        metrics["average_latency_seconds"] = round(sum(latencies) / len(latencies), 3)
    return metrics


def print_failures(records: list) -> None:
    failures = [r for r in records if not r.get("exact_match")]
    if not failures:
        return
    print(f"\nFAILED CASES ({len(failures)}):")
    for r in failures:
        print("-" * 80)
        print("Query:     ", r["query"])
        print("Expected:  ", r["expected"])
        if r.get("prediction") is None:
            print("Error:     ", r.get("error"))
        else:
            print("Prediction:", {key: r["prediction"].get(key) for key in r["expected"]})
