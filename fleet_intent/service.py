"""Classification pipeline: rule pre-check, then the model, then the rule layer."""

import time
from dataclasses import dataclass
from typing import Protocol

from .rules import postprocess, pre_classify


class Extractor(Protocol):
    def extract(self, query: str) -> dict: ...


@dataclass
class Classification:
    result: dict
    source: str  # "rules" when no model call was needed, otherwise "model"
    latency_ms: float


class Classifier:
    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def classify(self, query: str) -> Classification:
        start = time.perf_counter()

        # 1. Requests that fixed rules can answer never reach the model.
        ruled = pre_classify(query)
        if ruled is not None:
            result, source = ruled, "rules"
        else:
            # 2. The model interprets the query; code enforces the business rules.
            result, source = postprocess(query, self.extractor.extract(query)), "model"

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return Classification(result=result, source=source, latency_ms=latency_ms)
