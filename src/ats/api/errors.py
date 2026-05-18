"""Exception → HTTP status mapping for the matching engine's exceptions."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ats.core.logger import get_logger
from ats.matching.base import EmbeddingMissing, VacancyNotFound

log = get_logger(__name__)


def register(app: FastAPI) -> None:
    @app.exception_handler(VacancyNotFound)
    async def _vacancy_404(_req: Request, exc: VacancyNotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(EmbeddingMissing)
    async def _embedding_412(_req: Request, exc: EmbeddingMissing) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=412)

    @app.exception_handler(ValueError)
    async def _value_error_400(_req: Request, exc: ValueError) -> JSONResponse:
        # Catch the matchers' input-validation errors (e.g., empty text).
        log.warning("api_value_error", err=str(exc))
        return JSONResponse({"detail": str(exc)}, status_code=400)
