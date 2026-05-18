"""TF-IDF + LogisticRegression matcher.

Fits a TfidfVectorizer on the joint corpus (Resume.csv + all vacancies +
all candidate raw_texts) on first call and caches the fitted state at
module scope. Re-fits if the candidate row count changes.

A small classifier-agreement boost (+CATEGORY_BOOST) is added when the
LogisticRegression (trained on Resume.csv → 24 categories) predicts the
same category for both the vacancy and the candidate. The boost is
skipped for Russian vacancies and Russian-tagged candidates — the
classifier is English-only and would otherwise predict garbage.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ats.core.config import settings
from ats.core.logger import get_logger
from ats.db.models import Candidate, Vacancy
from ats.ingestion.parser.keywords import _stopwords
from ats.matching.base import (
    RESUME_CSV_PREFIX,
    CandidateMatch,
    MatchingStrategy,
    VacancyNotFound,
)

log = get_logger(__name__)

CATEGORY_BOOST = 0.10  # additive boost when classifier agrees on category

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RESUME_CSV_PATH = _PROJECT_ROOT / "data" / "eval" / "Resume.csv"

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def _is_russian(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


@dataclass
class _State:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression | None
    cand_ids: list[int]
    cand_names: list[str | None]
    cand_emails: list[str | None]
    cand_source_files: list[str]
    cand_parsed_jsons: list[dict | None]
    cand_vecs: sp.csr_matrix
    cand_langs: list[set[str]]
    # Index-aligned boolean: True if the candidate is a "real" CV (not from
    # the Resume.csv eval corpus). The matcher zeroes out non-real entries
    # before argsort so production callers never see eval rows.
    cand_is_real: np.ndarray
    n_candidates_at_fit: int


_state: _State | None = None
_state_lock = asyncio.Lock()


async def _count_candidates(session: AsyncSession) -> int:
    return int(
        (await session.execute(
            select(func.count(Candidate.id)).where(
                Candidate.raw_text.is_not(None),
                func.length(Candidate.raw_text) > 0,
            )
        )).scalar_one()
    )


async def _fit_state(session: AsyncSession) -> _State:
    if _RESUME_CSV_PATH.exists():
        df = pd.read_csv(_RESUME_CSV_PATH, on_bad_lines="skip")
        resume_texts = df["Resume_str"].fillna("").tolist()
        resume_cats = df["Category"].tolist()
    else:
        resume_texts, resume_cats = [], []

    vac_rows = (await session.execute(
        select(Vacancy.title, Vacancy.description)
    )).all()
    vac_texts = [f"{t}\n\n{d}" for t, d in vac_rows]

    cand_rows = (await session.execute(
        select(
            Candidate.id,
            Candidate.name,
            Candidate.email,
            Candidate.source_file,
            Candidate.raw_text,
            Candidate.parsed_json,
        )
        .where(Candidate.raw_text.is_not(None))
        .where(func.length(Candidate.raw_text) > 0)
        .order_by(Candidate.id)
    )).all()

    log.info(
        "tfidf_fit_start",
        resume_csv_rows=len(resume_texts),
        vacancies=len(vac_texts),
        candidates=len(cand_rows),
    )

    corpus = resume_texts + vac_texts + [r.raw_text for r in cand_rows]
    if not corpus:
        # Empty universe — return a no-op state. The vectorizer is fit on a
        # placeholder so subsequent transform() calls don't raise.
        vectorizer = TfidfVectorizer(stop_words=_stopwords())
        vectorizer.fit(["placeholder"])
        return _State(
            vectorizer=vectorizer,
            classifier=None,
            cand_ids=[], cand_names=[], cand_emails=[],
            cand_source_files=[], cand_parsed_jsons=[],
            cand_vecs=sp.csr_matrix((0, len(vectorizer.vocabulary_))),
            cand_langs=[],
            cand_is_real=np.zeros(0, dtype=bool),
            n_candidates_at_fit=0,
        )

    vectorizer = TfidfVectorizer(
        max_features=settings.models.tfidf.max_features,
        ngram_range=settings.models.tfidf.ngram_range,
        min_df=settings.models.tfidf.min_df,
        sublinear_tf=True,
        lowercase=True,
        stop_words=_stopwords(),
    )
    vectorizer.fit(corpus)

    classifier: LogisticRegression | None = None
    if resume_texts:
        classifier = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
        classifier.fit(vectorizer.transform(resume_texts), resume_cats)

    cand_texts = [r.raw_text for r in cand_rows]
    cand_vecs: sp.csr_matrix = (
        sp.csr_matrix(vectorizer.transform(cand_texts))
        if cand_texts
        else sp.csr_matrix((0, len(vectorizer.vocabulary_)))
    )

    log.info(
        "tfidf_fit_done",
        vocab_size=len(vectorizer.vocabulary_),
        classifier_classes=(len(classifier.classes_) if classifier else 0),
    )

    cand_source_files = [r.source_file for r in cand_rows]
    return _State(
        vectorizer=vectorizer,
        classifier=classifier,
        cand_ids=[r.id for r in cand_rows],
        cand_names=[r.name for r in cand_rows],
        cand_emails=[r.email for r in cand_rows],
        cand_source_files=cand_source_files,
        cand_parsed_jsons=[r.parsed_json for r in cand_rows],
        cand_vecs=cand_vecs,
        cand_langs=[
            set((r.parsed_json or {}).get("languages_detected", []))
            for r in cand_rows
        ],
        cand_is_real=np.array(
            [not sf.startswith(RESUME_CSV_PREFIX) for sf in cand_source_files],
            dtype=bool,
        ),
        n_candidates_at_fit=len(cand_rows),
    )


async def _get_state(session: AsyncSession) -> _State:
    global _state
    async with _state_lock:
        n = await _count_candidates(session)
        if _state is None or _state.n_candidates_at_fit != n:
            _state = await _fit_state(session)
        return _state


class TfidfMatcher(MatchingStrategy):
    name = "tfidf"

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
        text = f"{vacancy.title}\n\n{vacancy.description}"
        log.info("tfidf_match_by_vacancy", job_id=job_id, title=vacancy.title, top_k=top_k)
        return await self._search(session, text, top_k)

    async def match_by_text(
        self,
        session: AsyncSession,
        vacancy_text: str,
        top_k: int | None = None,
        vacancy_title: str | None = None,
    ) -> list[CandidateMatch]:
        top_k = top_k or settings.matching.top_k
        full = f"{vacancy_title}\n\n{vacancy_text}" if vacancy_title else vacancy_text
        log.info("tfidf_match_by_text", chars=len(full), top_k=top_k)
        return await self._search(session, full, top_k)

    async def _search(
        self,
        session: AsyncSession,
        vacancy_text: str,
        top_k: int,
    ) -> list[CandidateMatch]:
        state = await _get_state(session)
        if not state.cand_ids:
            log.info("tfidf_match_done", results=0, top_score=None)
            return []

        vac_vec = state.vectorizer.transform([vacancy_text])
        sims = cosine_similarity(vac_vec, state.cand_vecs).ravel()

        if not _is_russian(vacancy_text) and state.classifier is not None:
            vac_cat = state.classifier.predict(vac_vec)[0]
            eligible_idx = [
                i for i, langs in enumerate(state.cand_langs)
                if "ru" not in langs
            ]
            if eligible_idx:
                cand_cats = state.classifier.predict(state.cand_vecs[eligible_idx])
                for k, i in enumerate(eligible_idx):
                    if cand_cats[k] == vac_cat:
                        sims[i] += CATEGORY_BOOST

        # Mask out Resume.csv eval candidates — production callers only see
        # real CVs. Eval scripts bypass this matcher entirely.
        sims[~state.cand_is_real] = -np.inf

        top_idx = np.argsort(-sims)[:top_k]
        results = [
            CandidateMatch(
                candidate_id=state.cand_ids[i],
                name=state.cand_names[i],
                email=state.cand_emails[i],
                source_file=state.cand_source_files[i],
                score=max(0.0, min(1.0, float(sims[i]))),
                strategy=self.name,
                parsed_json=state.cand_parsed_jsons[i],
            )
            for i in top_idx
            if np.isfinite(sims[i])
        ]
        log.info(
            "tfidf_match_done",
            results=len(results),
            top_score=results[0].score if results else None,
        )
        return results
