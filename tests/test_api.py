"""API tests with a fake model. No Ollama needed."""

import pytest
from fastapi.testclient import TestClient

from fleet_intent.api import create_app
from fleet_intent.config import Settings
from fleet_intent.llm import LLMUnavailableError
from fleet_intent.schemas import MAX_QUERY_LENGTH


class FakeModel:
    def __init__(self, output=None, error=None, status=None):
        self.output = output if output is not None else {}
        self.error = error
        self.status_value = status or {"ollama_reachable": True, "model_available": True}
        self.calls = []

    def extract(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.output

    def warm_up(self):
        pass

    def status(self):
        return self.status_value


def make_client(model):
    return TestClient(create_app(Settings(warm_up=False), client=model))


def test_model_answer_goes_through_rule_layer():
    model = FakeModel({"intent": "CARD_MANAGEMENT", "action": "remove", "card_number": "5555", "confidence": 0.9})
    response = make_client(model).post("/classify", json={"query": "remove card 5555"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "CARD_MANAGEMENT"
    assert body["action"] == "block"
    assert body["card_number"] == "5555"
    assert body["source"] == "model"
    assert body["latency_ms"] >= 0
    assert model.calls == ["remove card 5555"]


def test_rule_answer_skips_the_model():
    model = FakeModel()
    body = make_client(model).post("/classify", json={"query": "generate fuel usage report"}).json()
    assert body["intent"] == "UNKNOWN"
    assert body["source"] == "rules"
    assert model.calls == []


def test_query_is_stripped():
    model = FakeModel({"intent": "CARD_UNLOCK"})
    make_client(model).post("/classify", json={"query": "  unlock card 1234  "})
    assert model.calls == ["unlock card 1234"]


@pytest.mark.parametrize("payload", [{}, {"query": ""}, {"query": "   "}, {"query": "x" * (MAX_QUERY_LENGTH + 1)}])
def test_invalid_requests_are_rejected(payload):
    assert make_client(FakeModel()).post("/classify", json=payload).status_code == 422


def test_model_unavailable_returns_503_without_internal_details():
    model = FakeModel(error=LLMUnavailableError("Cannot reach Ollama at http://internal-host:11434"))
    response = make_client(model).post("/classify", json={"query": "activate card 1234"})
    assert response.status_code == 503
    assert "internal-host" not in response.text


def test_unparseable_model_output_becomes_unknown():
    body = make_client(FakeModel({})).post("/classify", json={"query": "activate card 1234"}).json()
    assert body["intent"] == "UNKNOWN"
    assert body["confidence"] == 0.0


def test_health():
    body = make_client(FakeModel()).get("/health").json()
    assert body["status"] == "ok"


def test_ready_when_model_available():
    response = make_client(FakeModel()).get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_not_ready_when_model_missing():
    model = FakeModel(status={"ollama_reachable": True, "model_available": False})
    response = make_client(model).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_demo_page_is_served():
    response = make_client(FakeModel()).get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
