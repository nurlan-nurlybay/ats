"""CLI driver for the resume parser.

Usage:
    python -m ats.ingestion.parser                          # scan data/cvs/
    python -m ats.ingestion.parser path/to/cv.pdf ...       # specific paths
    python -m ats.ingestion.parser --dry-run path/to/cv.pdf # don't write DB
    python -m ats.ingestion.parser --force                  # reparse existing
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import delete, select

from ats.core.config import settings
from ats.core.logger import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)
from ats.db.base import SessionLocal
from ats.db.models import Candidate
from ats.ingestion.cv_store import compute_hash
from ats.ingestion.parser.pipeline import parse_resume
from ats.ingestion.parser.schema import ParsedResume

log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


async def _upsert(parsed: ParsedResume, force: bool) -> bool:
    """Insert candidate row. Returns True iff a row was written."""
    source_path = Path(parsed.source_file)
    content_hash = (
        compute_hash(source_path.read_bytes()) if source_path.exists() else None
    )

    async with SessionLocal() as session:
        existing = await session.scalar(
            select(Candidate.id).where(Candidate.source_file == parsed.source_file)
        )
        if existing and not force:
            log.info(
                "candidate_skipped",
                reason="already_parsed",
                source=parsed.source_file,
            )
            return False
        if existing and force:
            await session.execute(
                delete(Candidate).where(Candidate.source_file == parsed.source_file)
            )

        session.add(
            Candidate(
                name=parsed.name,
                email=parsed.email,
                source_file=parsed.source_file,
                content_hash=content_hash,
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
            hash=content_hash[:12] if content_hash else None,
        )
        return True


async def _run(paths: list[Path], dry_run: bool, force: bool) -> int:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            for ext in ALLOWED_EXTENSIONS:
                files.extend(p.rglob(f"*{ext}"))
        elif p.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(p)
        else:
            log.warning("unsupported_file", path=str(p))

    files = sorted({f.resolve() for f in files})
    log.info("parse_batch_start", count=len(files), dry_run=dry_run, force=force)

    processed = 0
    for f in files:
        bind_context(file=f.name)
        try:
            parsed = parse_resume(f)
            if dry_run:
                log.info(
                    "dry_run_result",
                    name=parsed.name,
                    email=parsed.email,
                    phone=parsed.phone,
                    languages=parsed.languages_detected,
                    skills_count=len(parsed.skills),
                    skills_preview=parsed.skills[:5],
                    experience_count=len(parsed.experience),
                    education_count=len(parsed.education),
                    embedding_dim=len(parsed.embedding),
                )
                processed += 1
            elif await _upsert(parsed, force):
                processed += 1
        except Exception:
            log.exception("parse_failed", path=str(f))
        finally:
            clear_context()

    log.info("parse_batch_done", processed=processed)
    return processed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse resumes into the candidates table."
    )
    ap.add_argument(
        "paths", nargs="*", help="Files or directories. Default: data/cvs/"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print results, don't write to DB."
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Reparse and overwrite existing candidates.",
    )
    args = ap.parse_args()

    configure_logging(level="INFO", json_output=False)
    paths = [Path(p) for p in args.paths] if args.paths else [settings.paths.cvs]
    asyncio.run(_run(paths, dry_run=args.dry_run, force=args.force))


if __name__ == "__main__":
    main()
