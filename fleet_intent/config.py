"""Runtime settings, read from environment variables."""

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:3b"
    # A cold model load can take most of a minute on a laptop GPU.
    request_timeout: float = 120.0
    # How long Ollama keeps the model in memory after the last request.
    keep_alive: str = "30m"
    # Load the model in the background at startup so the first request is fast.
    warm_up: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", defaults.ollama_url).rstrip("/"),
            model_name=os.getenv("MODEL_NAME", defaults.model_name),
            request_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", defaults.request_timeout)),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", defaults.keep_alive),
            warm_up=_env_bool("WARM_UP_MODEL", defaults.warm_up),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
        )
