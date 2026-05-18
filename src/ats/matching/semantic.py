"""Semantic matcher: pgvector cosine similarity over bge-m3 embeddings.

Both `Vacancy.embedding` and `Candidate.embedding` are pre-populated and
unit-normalized (see `ats.utils.embed_vacancies` and the parser pipeline).
With normalized vectors, pgvector's `cosine_distance` = `1 - cos_sim`,
so the similarity score is just `1 - distance`.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ats.core.config import settings
from ats.core.logger import get_logger
from ats.db.models import Candidate, Vacancy
from ats.ingestion.parser.embed import embed_text
from ats.matching.base import (
    RESUME_CSV_PREFIX,
    CandidateMatch,
    EmbeddingMissing,
    MatchingStrategy,
    VacancyNotFound,
)

log = get_logger(__name__)


class SemanticMatcher(MatchingStrategy):
    name = "semantic"

    async def match_by_vacancy(
        self,
        session: AsyncSession,
        job_id: int,
        top_k: int | None = None,
    ) -> list[CandidateMatch]:
        top_k = top_k or settings.matching.top_k
        vacancy = await session.get(Vacancy, job_id)
        if vacancy is None:
            raise VacancyNotFound(f"vacancy id={job_id} not found")
        if vacancy.embedding is None:
            raise EmbeddingMissing(
                f"vacancy id={job_id} has no embedding — "
                f"run `python -m ats.utils.embed_vacancies` first"
            )
        log.info(
            "match_by_vacancy",
            job_id=job_id,
            title=vacancy.title,
            top_k=top_k,
        )
        return await self._search(session, list(vacancy.embedding), top_k)

    async def match_by_text(
        self,
        session: AsyncSession,
        vacancy_text: str,
        top_k: int | None = None,
        vacancy_title: str | None = None,
    ) -> list[CandidateMatch]:
        top_k = top_k or settings.matching.top_k
        full = f"{vacancy_title}\n\n{vacancy_text}" if vacancy_title else vacancy_text
        log.info("match_by_text", chars=len(full), top_k=top_k)
        query_vec = embed_text(full)
        return await self._search(session, query_vec, top_k)

    async def _search(
        self,
        session: AsyncSession,
        query_vec: list[float],
        top_k: int,
    ) -> list[CandidateMatch]:
        dist = Candidate.embedding.cosine_distance(query_vec).label("dist")
        stmt = (
            select(
                Candidate.id,
                Candidate.name,
                Candidate.email,
                Candidate.source_file,
                Candidate.parsed_json,
                dist,
            )
            .where(Candidate.embedding.is_not(None))
            .where(~Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%"))
            .order_by(dist)
            .limit(top_k)
        )
        rows = (await session.execute(stmt)).all()
        results = [
            CandidateMatch(
                candidate_id=r.id,
                name=r.name,
                email=r.email,
                source_file=r.source_file,
                score=max(0.0, min(1.0, 1.0 - float(r.dist))),
                strategy=self.name,
                parsed_json=r.parsed_json,
            )
            for r in rows
        ]
        log.info(
            "match_done",
            results=len(results),
            top_score=results[0].score if results else None,
        )
        return results
