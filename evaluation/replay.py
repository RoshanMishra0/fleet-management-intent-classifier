"""Replay recorded model outputs through the rule layer.

Reads the raw model outputs recorded on the original test set before the rule
layer existed (results/model_only_test_cases.json) and applies pre_classify /
postprocess to each one. This measures what the rule layer changes without
re-running the model.

It does NOT replace a fresh run: the rules were designed from these same
failures. Use `python -m evaluation.evaluate` on the held-out set for that.

    python -m evaluation.replay
"""

import copy
import json
from pathlib import Path

from evaluation.scoring import print_failures, summarize
from fleet_intent.rules import postprocess, pre_classify

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "model_only_test_cases.json"
OUT = ROOT / "results" / "replay_test_cases.json"


def main():
    with open(SOURCE, encoding="utf-8") as f:
        report = json.load(f)
    recorded = [
        {"query": r["query"], "expected": r["expected"], "prediction": r["prediction"]}
        for r in report["detailed_results"]
        if r.get("prediction") is not None
    ]

    before = summarize(copy.deepcopy(recorded))

    skipped = 0
    replayed = []
    for r in recorded:
        ruled = pre_classify(r["query"])
        if ruled is not None:
            skipped += 1
            prediction = ruled
        else:
            prediction = postprocess(r["query"], r["prediction"])
        replayed.append({"query": r["query"], "expected": r["expected"], "prediction": prediction})

    after = summarize(replayed)
    after["answered_without_model"] = skipped

    print("Model output only:   ", json.dumps(before))
    print("With the rule layer: ", json.dumps(after))
    print_failures(replayed)

    failures = [r for r in replayed if not r["exact_match"]]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model_output_only": before, "with_rule_layer": after, "failures": failures}, f, indent=2)


if __name__ == "__main__":
    main()
