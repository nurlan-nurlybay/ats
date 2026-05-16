"""Backfill `vacancies.embedding` with bge-m3 vectors.

Idempotent: only embeds rows where `embedding IS NULL`. The Stage 3
matching engine consumes these vectors via pgvector cosine distance.

Run:
    PYTHONPATH=src python -m ats.utils.embed_vacancies
"""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from ats.core.logger import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)
from ats.db.base import SessionLocal
from ats.db.models import Vacancy
from ats.ingestion.parser.embed import embed_texts

log = get_logger(__name__)


async def embed_vacancies(batch_size: int = 8) -> int:
    """Embed every vacancy with embedding IS NULL. Returns count embedded."""
    log.info("vacancy_embed_start")

    async with SessionLocal() as session:
        result = await session.execute(
            select(Vacancy).where(Vacancy.embedding.is_(None))
        )
        vacancies = list(result.scalars().all())
        log.info("vacancy_embed_pending", count=len(vacancies))

        if not vacancies:
            log.info("vacancy_embed_done", embedded=0)
            return 0

        texts = [f"{v.title}\n\n{v.description}" for v in vacancies]
        log.info("vacancy_embed_encoding", n=len(texts), batch_size=batch_size)
        vectors = embed_texts(texts, batch_size=batch_size)

        for vac, vec in zip(vacancies, vectors):
            bind_context(vacancy_id=vac.id, title=vac.title)
            try:
                vac.embedding = vec
                log.info("vacancy_embedded")
            except Exception:
                log.exception("vacancy_embed_failed")
            finally:
                clear_context()

        await session.commit()

    log.info("vacancy_embed_done", embedded=len(vacancies))
    return len(vacancies)


if __name__ == "__main__":
    configure_logging(level="INFO", json_output=False)
    asyncio.run(embed_vacancies())
