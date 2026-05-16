"""Load curated vacancy JSON files into the Postgres `vacancies` table.

Idempotent: dedupes by `source_filename`, so re-running inserts only new
files. To re-import an edited vacancy, delete its row first (or extend
this script with an UPDATE branch).

Run:
    PYTHONPATH=src python -m ats.utils.import_vacancies
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

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


async def import_vacancies(vacancies_dir: Path | None = None) -> int:
    src = vacancies_dir or settings.paths.vacancies
    files = sorted(src.glob("*.json"))
    log.info("vacancy_import_start", dir=str(src), file_count=len(files))

    inserted = 0
    async with SessionLocal() as session:
        result = await session.execute(select(Vacancy.source_filename))
        existing: set[str] = {row[0] for row in result.all()}

        for path in files:
            bind_context(file=path.name)
            try:
                if path.name in existing:
                    log.info("vacancy_skipped", reason="already_imported")
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                vac = Vacancy(
                    title=payload["title"],
                    experience=payload.get("experience"),
                    description=payload["description"],
                    source_filename=path.name,
                )
                session.add(vac)
                inserted += 1
                log.info("vacancy_inserted", title=vac.title)
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


if __name__ == "__main__":
    configure_logging(level="INFO", json_output=False)
    asyncio.run(import_vacancies())
