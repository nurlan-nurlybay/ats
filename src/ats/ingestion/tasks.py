"""Celery tasks for background CV ingestion.

Beat schedule: every 5 minutes, pull new CVs from Gmail and parse them
into the database automatically. The same task can be triggered on-demand
via the ``POST /ingestion/pull`` API endpoint.

Broker: Redis (``CELERY_BROKER_URL`` env-var, defaults to
``redis://localhost:6379/0``).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from celery import Celery
from celery.schedules import crontab

from ats.core.logger import configure_logging, get_logger

# Configure logging early so structlog is set up before any other ats import
# that might call get_logger at module scope.
configure_logging(level="INFO", json_output=True)
log = get_logger(__name__)

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

app = Celery("ats_worker", broker=BROKER_URL)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_backend=BROKER_URL,
    timezone="UTC",
    enable_utc=True,
    # Beat schedule — pull + parse every 5 minutes.
    beat_schedule={
        "pull-and-parse-cvs-every-5m": {
            "task": "ats.ingestion.tasks.pull_and_parse_cvs",
            "schedule": 300.0,  # seconds
        },
    },
)


@app.on_after_configure.connect
def _eager_warmup(sender, **kwargs):
    """Eager hash-heal at worker startup.

    `EmailIngestionService.__init__` runs `_load_or_heal_hashes()` — counts
    CV files on disk vs entries in seen_hashes.json and rebuilds when they
    mismatch. Without this hook the heal would wait until the first beat
    tick (up to 5 min), which doesn't match the user's "on startup" intent.
    Lazy-imported so the api container's `from ats.ingestion.tasks import …`
    path doesn't pay this cost.
    """
    from ats.ingestion.mail_client import EmailIngestionService

    try:
        EmailIngestionService()  # side-effect: heal hashes
        log.info("worker_warmup_done")
    except Exception:
        log.exception("worker_warmup_failed")


# ── helpers (reuse parser CLI's upsert logic) ────────────────────────────

async def _upsert_parsed(parsed: "ParsedResume") -> bool:
    """Insert candidate row. Returns True if a row was written."""
    from sqlalchemy import select

    from ats.db.base import SessionLocal
    from ats.db.models import Candidate

    async with SessionLocal() as session:
        existing = await session.scalar(
            select(Candidate.id).where(
                Candidate.source_file == parsed.source_file
            )
        )
        if existing:
            log.info(
                "candidate_skipped",
                reason="already_parsed",
                source=parsed.source_file,
            )
            return False

        session.add(
            Candidate(
                name=parsed.name,
                email=parsed.email,
                source_file=parsed.source_file,
                raw_text=parsed.raw_text,
                embedding=parsed.embedding,
                parsed_json=parsed.model_dump(exclude={"raw_text", "embedding"}),
            )
        )
        await session.commit()
        log.info(
            "candidate_inserted",
            source=parsed.source_file,
            name=parsed.name,
            skills=len(parsed.skills),
        )
        return True


async def _pull_and_parse() -> dict:
    """Core async logic: pull from Gmail, parse each new file, upsert."""
    from ats.ingestion.mail_client import EmailIngestionService
    from ats.ingestion.parser.pipeline import parse_resume

    service = EmailIngestionService()
    attachments = service.fetch_new_resumes()

    pulled = len(attachments)
    parsed = 0
    errors: list[str] = []

    for att in attachments:
        try:
            result = parse_resume(att.path)
            if await _upsert_parsed(result):
                parsed += 1
        except Exception as exc:
            msg = f"{att.path.name}: {exc}"
            log.exception("parse_failed_in_task", file=att.path.name)
            errors.append(msg)

    summary = {"pulled": pulled, "parsed": parsed, "errors": errors}
    log.info("pull_and_parse_done", **summary)
    return summary


@app.task(name="ats.ingestion.tasks.pull_and_parse_cvs", bind=True)
def pull_and_parse_cvs(self) -> dict:  # noqa: ANN001
    """Celery task: pull CVs from Gmail and parse into Postgres.

    Runs the async pipeline in a fresh event loop (Celery workers are
    sync by default).
    """
    log.info("pull_and_parse_task_start", task_id=self.request.id)
    try:
        result = asyncio.run(_pull_and_parse())
    except Exception as exc:
        log.exception("pull_and_parse_task_failed")
        raise self.retry(exc=exc, countdown=60, max_retries=2) from exc
    return result
