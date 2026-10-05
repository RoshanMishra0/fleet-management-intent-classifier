import json
import sys
import time
import requests


API_URL = "http://127.0.0.1:8000/classify"

# Usage:
#   python evaluate.py                          -> test_cases.json, metrics_report.json
#   python evaluate.py test_cases_heldout.json  -> metrics_report_heldout.json
TEST_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_cases.json"
REPORT_FILE = (
    "metrics_report.json"
    if TEST_FILE == "test_cases.json"
    else "metrics_report_" + TEST_FILE.replace("test_cases_", "").replace(".json", "") + ".json"
)


def normalize_value(value):
    if isinstance(value, list):
        return sorted(value)
    return value


def main():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Support both formats:
    # 1. [ {...}, {...} ]
    # 2. { "test_cases": [ {...}, {...} ] }
    if isinstance(test_cases, dict) and "test_cases" in test_cases:
        test_cases = test_cases["test_cases"]

    total = len(test_cases)

    intent_correct = 0
    exact_match_correct = 0
    total_slots = 0
    correct_slots = 0
    valid_json = 0
    latencies = []
    detailed_results = []

    for case in test_cases:
        query = case.get("query")
        expected = case.get("expected", {})

        if not query:
            print("Skipping case without query:", case)
            continue

        if not expected:
            print("Skipping case without expected:", case)
            continue

        start = time.time()

        try:
            response = requests.post(API_URL, json={"query": query}, timeout=120)
            latency = time.time() - start
            latencies.append(latency)

            prediction = response.json()
            valid_json += 1

        except Exception as e:
            detailed_results.append({
                "query": query,
                "error": str(e),
                "passed": False
            })
            continue

        if prediction.get("intent") == expected.get("intent"):
            intent_correct += 1

        case_slot_correct = 0
        case_total_slots = 0

        for key, expected_value in expected.items():
            case_total_slots += 1
            total_slots += 1

            predicted_value = prediction.get(key)

            if normalize_value(predicted_value) == normalize_value(expected_value):
                correct_slots += 1
                case_slot_correct += 1

        exact_match = case_slot_correct == case_total_slots

        if exact_match:
            exact_match_correct += 1

        detailed_results.append({
            "query": query,
            "expected": expected,
            "prediction": prediction,
            "latency_seconds": round(latency, 3),
            "exact_match": exact_match
        })

    metrics = {
        "total_test_cases": total,
        "intent_accuracy": round(intent_correct / total, 4) if total else 0,
        "slot_accuracy": round(correct_slots / total_slots, 4) if total_slots else 0,
        "exact_match_accuracy": round(exact_match_correct / total, 4) if total else 0,
        "valid_json_rate": round(valid_json / total, 4) if total else 0,
        "average_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None
    }

    print(json.dumps(metrics, indent=2))

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "detailed_results": detailed_results
        }, f, indent=2)
    print("\nFAILED CASES:")
    for result in detailed_results:
        if result.get("exact_match") is False:
            print("-" * 80)
            print("Query:", result["query"])
            print("Expected:", result["expected"])
            print("Prediction:", result["prediction"])

if __name__ == "__main__":
    main()