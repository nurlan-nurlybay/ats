"""Load curated vacancy JSON files into the Postgres `vacancies` table.

Idempotent: dedupes by `source_filename`, so re-running inserts only new
files. Use `--force` to re-import files whose contents have changed (it
DELETE-then-INSERTs, then `embed_vacancies` (run separately) backfills
the NULL embedding).

Run:
    PYTHONPATH=src python -m ats.utils.import_vacancies
    PYTHONPATH=src python -m ats.utils.import_vacancies --force            # reimport whole dir
    PYTHONPATH=src python -m ats.utils.import_vacancies --force path.json  # reimport one file
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
from ats.db.models import Vacancy

log = get_logger(__name__)


async def import_vacancies(
    vacancies_dir: Path | None = None,
    *,
    force: bool = False,
    only_files: list[Path] | None = None,
) -> int:
    """Insert vacancy JSONs into Postgres.

    Args:
        vacancies_dir: directory to scan (defaults to settings.paths.vacancies).
            Ignored when `only_files` is provided.
        force: when True, existing rows with the same source_filename are
            DELETEd and re-INSERTed (embedding is wiped, run
            `embed_vacancies` afterward to repopulate).
        only_files: scope to a specific list of files instead of scanning
            the directory.

    Returns:
        Number of rows newly inserted (or replaced, when force=True).
    """
    if only_files is not None:
        files = sorted(only_files)
        log.info("vacancy_import_start", file_count=len(files), force=force,
                 mode="explicit_files")
    else:
        src = vacancies_dir or settings.paths.vacancies
        files = sorted(src.glob("*.json"))
        log.info("vacancy_import_start", dir=str(src), file_count=len(files),
                 force=force)

    inserted = 0
    async with SessionLocal() as session:
        result = await session.execute(select(Vacancy.source_filename))
        existing: set[str] = {row[0] for row in result.all()}

        for path in files:
            bind_context(file=path.name)
            try:
                if path.name in existing and not force:
                    log.info("vacancy_skipped", reason="already_imported")
                    continue
                if path.name in existing and force:
                    await session.execute(
                        delete(Vacancy).where(Vacancy.source_filename == path.name)
                    )
                    log.info("vacancy_deleted_for_reimport")
                payload = json.loads(path.read_text(encoding="utf-8"))
                vac = Vacancy(
                    title=payload["title"],
                    experience=payload.get("experience"),
                    description=payload["description"],
                    source_filename=path.name,
                )
                session.add(vac)
                inserted += 1
                log.info("vacancy_inserted", title=vac.title, force=force)
            except Exception:
                log.exception("vacancy_import_failed")
            finally:
                clear_context()

        await session.commit()

    log.info(
        "vacancy_import_done",
        inserted=inserted,
        skipped=len(files) - inserted,
    )
    return inserted


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Import vacancy JSON files into Postgres."
    )
    ap.add_argument(
        "files",
        type=Path,
        nargs="*",
        help="Specific JSON file(s) to import. If omitted, scans settings.paths.vacancies.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete + re-insert rows that already exist (e.g. after editing).",
    )
    return ap.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if args.files:
        only_files = [p.resolve() for p in args.files]
        await import_vacancies(only_files=only_files, force=args.force)
    else:
        await import_vacancies(force=args.force)


if __name__ == "__main__":
    configure_logging(level="INFO", json_output=False)
    asyncio.run(_run(_parse_args()))
