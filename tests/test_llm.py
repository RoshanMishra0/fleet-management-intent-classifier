"""Ollama client tests with the HTTP session replaced. No Ollama needed."""

import json

import pytest
import requests

from fleet_intent.config import Settings
from fleet_intent.llm import GENERATION_OPTIONS, LLMUnavailableError, OllamaClient


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.sent = response, error, []

    def post(self, url, json=None, timeout=None):
        self.sent.append((url, json, timeout))
        if self.error:
            raise self.error
        return self.response

    get = post


def client_with(session, **settings):
    client = OllamaClient(Settings(**settings))
    client._session = session
    return client


def test_extract_sends_deterministic_json_request():
    session = FakeSession(FakeResponse({"response": json.dumps({"intent": "FUEL_SEARCH"})}))
    client = client_with(session, model_name="test-model", request_timeout=7)
    assert client.extract("find diesel") == {"intent": "FUEL_SEARCH"}

    url, payload, timeout = session.sent[0]
    assert url.endswith("/api/generate")
    assert payload["model"] == "test-model"
    assert payload["format"] == "json"
    assert payload["options"] == GENERATION_OPTIONS
    assert "find diesel" in payload["prompt"]
    assert timeout == 7


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_invalid_model_output_returns_empty_dict(raw):
    client = client_with(FakeSession(FakeResponse({"response": raw})))
    assert client.extract("anything") == {}


@pytest.mark.parametrize("error", [requests.ConnectionError(), requests.Timeout()])
def test_network_errors_raise_unavailable(error):
    with pytest.raises(LLMUnavailableError):
        client_with(FakeSession(error=error)).extract("anything")


def test_http_error_raises_unavailable():
    with pytest.raises(LLMUnavailableError):
        client_with(FakeSession(FakeResponse({}, status=500))).extract("anything")


def test_status_reports_missing_model():
    session = FakeSession(FakeResponse({"models": [{"name": "other:1b"}]}))
    assert client_with(session).status() == {"ollama_reachable": True, "model_available": False}


def test_status_reports_available_model():
    session = FakeSession(FakeResponse({"models": [{"name": "qwen2.5:3b"}]}))
    assert client_with(session).status() == {"ollama_reachable": True, "model_available": True}


def test_status_when_ollama_down():
    session = FakeSession(error=requests.ConnectionError())
    assert client_with(session).status() == {"ollama_reachable": False, "model_available": False}
