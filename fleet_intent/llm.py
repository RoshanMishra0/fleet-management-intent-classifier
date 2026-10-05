"""Ollama client: one JSON-mode generate call per query."""

import json
import logging

import requests

from .config import Settings
from .prompt import build_prompt

log = logging.getLogger(__name__)

# Fixed so the same query always gets the same answer.
GENERATION_OPTIONS = {"temperature": 0, "top_p": 0.1, "repeat_penalty": 1.1}

WARM_UP_TIMEOUT_SECONDS = 300


class LLMUnavailableError(RuntimeError):
    """The model server could not be reached or did not answer in time."""


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return self.settings.ollama_url + path

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        try:
            response = self._session.post(self._url(path), json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.Timeout as e:
            raise LLMUnavailableError(f"Ollama did not answer within {timeout:g} s") from e
        except requests.ConnectionError as e:
            raise LLMUnavailableError(f"Cannot reach Ollama at {self.settings.ollama_url}") from e
        except requests.HTTPError as e:
            raise LLMUnavailableError(f"Ollama returned HTTP {e.response.status_code}") from e

    def extract(self, query: str) -> dict:
        """Return the model's raw JSON object, or {} if it is not valid JSON.

        An empty dict is turned into UNKNOWN with confidence 0 by the rule layer.
        """
        payload = {
            "model": self.settings.model_name,
            "prompt": build_prompt(query),
            "stream": False,
            "format": "json",
            "keep_alive": self.settings.keep_alive,
            "options": GENERATION_OPTIONS,
        }
        data = self._post("/api/generate", payload, self.settings.request_timeout)
        try:
            parsed = json.loads(data.get("response", "{}"))
        except json.JSONDecodeError:
            log.warning("Model returned invalid JSON; answering UNKNOWN")
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def warm_up(self) -> None:
        """Load the model into memory. A generate call without a prompt only loads it."""
        payload = {"model": self.settings.model_name, "keep_alive": self.settings.keep_alive}
        self._post("/api/generate", payload, WARM_UP_TIMEOUT_SECONDS)

    def status(self) -> dict:
        """Whether Ollama is reachable and the configured model has been pulled."""
        try:
            response = self._session.get(self._url("/api/tags"), timeout=5)
            response.raise_for_status()
            names = {m.get("name") for m in response.json().get("models", [])}
        except (requests.RequestException, ValueError):
            return {"ollama_reachable": False, "model_available": False}
        wanted = self.settings.model_name
        available = wanted in names or (":" not in wanted and f"{wanted}:latest" in names)
        return {"ollama_reachable": True, "model_available": available}
