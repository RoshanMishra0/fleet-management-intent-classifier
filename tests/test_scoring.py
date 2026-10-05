from evaluation.scoring import summarize


def test_only_expected_fields_are_compared():
    records = [{"query": "q", "expected": {"intent": "A", "amenities": ["x", "y"]},
                "prediction": {"intent": "A", "amenities": ["y", "x"], "action": "extra"}}]
    metrics = summarize(records)
    assert metrics["exact_match_accuracy"] == 1.0
    assert records[0]["exact_match"] is True


def test_failed_request_counts_as_wrong():
    records = [
        {"query": "a", "expected": {"intent": "A", "action": "x"}, "prediction": {"intent": "A", "action": "y"}},
        {"query": "b", "expected": {"intent": "B"}, "prediction": None, "error": "timeout"},
    ]
    metrics = summarize(records)
    assert metrics["intent_accuracy"] == 0.5
    assert metrics["slot_accuracy"] == round(1 / 3, 4)
    assert metrics["exact_match_accuracy"] == 0.0
    assert metrics["valid_json_rate"] == 0.5
