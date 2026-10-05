"""Run a labelled test set and report accuracy.

    python -m evaluation.evaluate                          # held-out set, through the running service
    python -m evaluation.evaluate data/test_cases.json     # another test set
    python -m evaluation.evaluate --model-only             # the model alone: no rule layer, no server

Results go to results/<service|model_only>_<test set name>.json unless --out is given.
"""

import argparse
import json
import time
from pathlib import Path

import requests

from evaluation.scoring import print_failures, summarize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "data" / "test_cases_heldout.json"
DEFAULT_URL = "http://127.0.0.1:8000/classify"


def load_cases(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    # Accept both [ {...} ] and { "test_cases": [ {...} ] }.
    if isinstance(cases, dict) and "test_cases" in cases:
        cases = cases["test_cases"]
    return [c for c in cases if c.get("query") and c.get("expected")]


def service_predictor(url: str):
    session = requests.Session()

    def predict(query: str) -> dict:
        response = session.post(url, json={"query": query}, timeout=180)
        response.raise_for_status()
        return response.json()

    return predict


def model_only_predictor():
    from fleet_intent.config import Settings
    from fleet_intent.llm import OllamaClient

    return OllamaClient(Settings.from_env()).extract


def run(cases: list, predict) -> list:
    records = []
    for i, case in enumerate(cases, 1):
        record = {"query": case["query"], "expected": case["expected"]}
        start = time.perf_counter()
        try:
            record["prediction"] = predict(case["query"])
            record["latency_seconds"] = round(time.perf_counter() - start, 3)
        except Exception as e:  # one failed request must not stop the run
            record["prediction"] = None
            record["error"] = str(e)
        records.append(record)
        print(f"\r{i}/{len(cases)}", end="", flush=True)
    print()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="?", type=Path, default=DEFAULT_CASES, help="test set JSON file")
    parser.add_argument("--url", default=DEFAULT_URL, help="classify endpoint of the running service")
    parser.add_argument("--model-only", action="store_true", help="call the model directly, without the rule layer")
    parser.add_argument("--out", type=Path, help="where to write the per-case report")
    args = parser.parse_args()

    mode = "model_only" if args.model_only else "service"
    predict = model_only_predictor() if args.model_only else service_predictor(args.url)
    out = args.out or ROOT / "results" / f"{mode}_{args.cases.stem}.json"

    records = run(load_cases(args.cases), predict)
    metrics = summarize(records)

    print(json.dumps(metrics, indent=2))
    print_failures(records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "detailed_results": records}, f, indent=2)
    print(f"\nReport written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
