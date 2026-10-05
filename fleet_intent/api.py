"""HTTP API and demo page.

Run with:  uvicorn fleet_intent.api:app
"""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings
from .llm import LLMUnavailableError, OllamaClient
from .schemas import (
    ClassificationResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    ReadinessResponse,
)
from .service import Classifier

STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger("fleet_intent")


def _warm_up(client: OllamaClient) -> None:
    try:
        client.warm_up()
        log.info("Model loaded")
    except LLMUnavailableError as e:
        log.warning("Model warm-up failed: %s", e)


def create_app(settings: Optional[Settings] = None, client=None) -> FastAPI:
    """Build the app. Tests pass a fake client; production uses Ollama."""
    settings = settings or Settings.from_env()
    client = client or OllamaClient(settings)
    classifier = Classifier(client)

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warm_up:
            threading.Thread(target=_warm_up, args=(client,), daemon=True).start()
        yield

    app = FastAPI(
        title="Fleet Management Intent Classifier",
        description="Intent classification and slot extraction on a local LLM, with a deterministic rule layer.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable(request: Request, exc: LLMUnavailableError):
        log.error("LLM unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "The language model is unavailable. Check that Ollama is running and the model is pulled."},
        )

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health():
        """Liveness: the API process is up."""
        return HealthResponse(status="ok", version=__version__)

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
        tags=["health"],
    )
    def ready():
        """Readiness: Ollama is reachable and the model has been pulled."""
        status = client.status()
        body = ReadinessResponse(
            ready=status["ollama_reachable"] and status["model_available"],
            model=settings.model_name,
            **status,
        )
        return JSONResponse(status_code=200 if body.ready else 503, content=body.model_dump())

    @app.post(
        "/classify",
        response_model=ClassificationResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["classification"],
    )
    def classify(request: QueryRequest):
        outcome = classifier.classify(request.query)
        # The query is not logged: it can contain card numbers and email addresses.
        log.info(
            "classified intent=%s source=%s latency_ms=%.1f",
            outcome.result["intent"], outcome.source, outcome.latency_ms,
        )
        return ClassificationResponse(**outcome.result, source=outcome.source, latency_ms=outcome.latency_ms)

    return app


app = create_app()
