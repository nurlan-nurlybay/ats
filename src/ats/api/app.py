"""FastAPI app factory.

Composition:
  - configure_logging at module import (JSON for cloud, console for local)
  - lifespan: pre-warm bge-m3 + TF-IDF state so the first request is instant
  - middleware: CORS (UI lives on a different port) + per-request logging context
  - exception handlers from ats.api.errors
  - routers: health, vacancies, candidates, recommendations
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from ats.api import errors
from ats.api.routers import candidates, health, ingestion, recommendations, vacancies
from ats.core.logger import bind_context, clear_context, configure_logging, get_logger
from ats.db.base import SessionLocal
from ats.matching import MATCHERS, TfidfMatcher

# Configure once at import — `python -m ats.api` and uvicorn-as-module both
# go through this path. JSON is the right choice in a container; the CLI
# entry points override to console output.
configure_logging(level="INFO", json_output=True)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm slow singletons so the first HTTP request isn't a 20-s wait."""
    log.info("api_lifespan_warmup_start", strategies=sorted(MATCHERS))

    # bge-m3 model load (~5 s if cached on disk; ~6 min if not).
    try:
        from ats.ingestion.parser.embed import get_embedder
        get_embedder()
        log.info("api_lifespan_embedder_loaded")
    except Exception as e:
        # The semantic matcher will fail at request time if this never works,
        # but TF-IDF and other endpoints should still serve.
        log.warning("api_lifespan_embedder_failed", err=str(e))

    # TF-IDF state — fit vectorizer + LR classifier on the joint corpus.
    try:
        async with SessionLocal() as s:
            await TfidfMatcher()._search(s, "warmup", top_k=1)
        log.info("api_lifespan_tfidf_fit")
    except Exception as e:
        log.warning("api_lifespan_tfidf_failed", err=str(e))

    log.info("api_lifespan_warmup_done")
    yield
    log.info("api_lifespan_shutdown")


class _RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request_id to every structlog line emitted during the request."""

    async def dispatch(self, request: Request, call_next):
        bind_context(request_id=str(uuid.uuid4())[:8], path=request.url.path)
        try:
            return await call_next(request)
        finally:
            clear_context()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Recruiting Agent — ATS API",
        version="0.4.0",
        description=(
            "REST API for resume / vacancy matching. Three strategies side-by-side: "
            "`semantic` (bge-m3 + pgvector), `tfidf` (TF-IDF + LR classifier), "
            "`llm` (Qwen rerank with natural-language explanations)."
        ),
        lifespan=lifespan,
    )

    # CORS — Streamlit runs in the browser at a different origin (port 8501);
    # without this the fetch from JS land would fail preflight.
    cors_origins = os.environ.get("ATS_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(_RequestContextMiddleware)

    errors.register(app)
    app.include_router(health.router)
    app.include_router(vacancies.router)
    app.include_router(candidates.router)
    app.include_router(recommendations.router)
    app.include_router(ingestion.router)
    return app


app = create_app()
