"""Structured JSON logging via structlog.

Call `configure_logging()` once at process start. Use `get_logger(__name__)`
in every module. Use `bind_context(job_id=..., candidate_name=...)` to
attach per-request fields that will appear in every subsequent log line
within the same async task / thread, without manual plumbing.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
)
from structlog.stdlib import BoundLogger
from structlog.types import Processor


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog and route stdlib logging through it.

    JSON output is the default — appropriate for Docker / cloud deployments
    where logs are scraped by a collector. Set `json_output=False` for a
    pretty console renderer when running locally.
    """
    log_level = getattr(logging, level.upper())

    shared_processors: list[Processor] = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a structlog logger. Pass `__name__` from the calling module."""
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context fields to all subsequent logs in this async/thread context."""
    bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear bound context fields. Call at request / job boundary end."""
    clear_contextvars()
