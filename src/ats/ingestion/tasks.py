"""Celery tasks for background CV ingestion.

Beat schedule: every 5 minutes, pull new CVs from Gmail and parse them
into the database automatically. The same task can be triggered on-demand
via the ``POST /ingestion/pull`` API endpoint.

Broker: Redis (``CELERY_BROKER_URL`` env-var, defaults to
``redis://localhost:6379/0``).

Per attachment, the flow is:

  1. mail_client computes SHA-256, looks up ``candidates.content_hash``,
     and only writes a temp file when the hash is new.
  2. We parse the temp file via ``parse_resume``.
  3. Insert a Candidate row with ``source_file = temp_path`` and the hash.
  4. Rename temp → ``{candidate_id}_{sanitized}.{ext}`` and UPDATE
     ``source_file``.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from celery import Celery
from sqlalchemy import select

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
    timezone="Asia/Almaty",
    enable_utc=True,
    # Beat schedule — pull + parse every 5 minutes.
    beat_schedule={
        "pull-and-parse-cvs-every-5m": {
            "task": "ats.ingestion.tasks.pull_and_parse_cvs",
            "schedule": 300.0,  # seconds
        },
    },
)


# ── core async logic ──────────────────────────────────────────────────────


async def _ingest_one(att) -> bool:
    """Parse one SavedAttachment, insert + rename. Returns True if a row
    was written."""
    from ats.core.config import settings
    from ats.db.base import SessionLocal
    from ats.db.models import Candidate
    from ats.ingestion.cv_store import final_path_for
    from ats.ingestion.parser.pipeline import parse_resume

    temp_path: Path = att.path

    # Belt-and-suspenders: hash should have been checked in mail_client, but
    # in race conditions two attachments could pass the check simultaneously
    # before either commits. The unique index on content_hash will catch the
    # second one — we just need to clean up the temp file gracefully.
    try:
        parsed = parse_resume(temp_path)
    except Exception:
        log.exception("parse_failed_in_task", file=temp_path.name)
        temp_path.unlink(missing_ok=True)
        return False

    async with SessionLocal() as session:
        existing = await session.scalar(
            select(Candidate.id).where(Candidate.content_hash == att.content_hash)
        )
        if existing is not None:
            log.info(
                "candidate_dupe_race",
                hash=att.content_hash[:12],
                existing_id=existing,
            )
            temp_path.unlink(missing_ok=True)
            return False

        from sqlalchemy.exc import IntegrityError

        cand = Candidate(
            source_file=str(temp_path),
            content_hash=att.content_hash,
            name=parsed.name,
            email=parsed.email,
            raw_text=parsed.raw_text,
            embedding=parsed.embedding,
            parsed_json=parsed.model_dump(exclude={"raw_text", "embedding"}),
        )
        session.add(cand)
        try:
            await session.commit()
            await session.refresh(cand)
        except IntegrityError:
            log.info(
                "candidate_dupe_race_on_commit",
                hash=att.content_hash[:12],
            )
            temp_path.unlink(missing_ok=True)
            return False

        final_path = final_path_for(
            settings.paths.cvs, cand.id, att.sanitized_basename
        )
        try:
            temp_path.rename(final_path)
        except OSError as exc:
            log.error(
                "candidate_rename_failed",
                id=cand.id, temp=str(temp_path), target=str(final_path),
                err=str(exc),
            )
        else:
            cand.source_file = str(final_path)
            await session.commit()

        log.info(
            "candidate_inserted",
            id=cand.id,
            source=cand.source_file,
            name=cand.name,
            hash=att.content_hash[:12],
        )
        return True


async def _pull_and_parse() -> dict:
    """Pull all UNSEEN Gmail attachments, parse each, insert + rename."""
    from ats.ingestion.mail_client import EmailIngestionService

    service = EmailIngestionService()
    attachments = await service.fetch_new_resumes()

    pulled = len(attachments)
    parsed = 0
    errors: list[str] = []

    for att in attachments:
        try:
            if await _ingest_one(att):
                parsed += 1
        except Exception as exc:
            errors.append(f"{att.path.name}: {exc}")
            log.exception("ingest_one_failed", file=att.path.name)

    summary = {"pulled": pulled, "parsed": parsed, "errors": errors}
    log.info("pull_and_parse_done", **summary)
    return summary


# ── celery task wrapper ───────────────────────────────────────────────────


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
