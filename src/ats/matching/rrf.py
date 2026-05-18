"""Reciprocal Rank Fusion (RRF) over multiple matchers.

For each candidate `c` that appears in any input ranking R, RRF computes:

    score(c) = Σ_R   1 / (k + rank_R(c))

where `k` is a smoothing constant. Cormack et al. 2009 set k=60 and
showed it dominates parameter-free fusion across TREC tracks; we use the
same default. Candidates absent from a ranking contribute 0 from that
ranking. The fused ranking sorts by descending score.

RRF is a *rank-based* fusion — it ignores the raw score values of the
input matchers, only their *rank position*. That's the right property
here: semantic scores live in [0.35, 0.65] while TF-IDF lives in
[0.10, 0.45]; mixing them by score would be apples-to-oranges. Mixing
by rank is well-posed regardless of the underlying distributions.
"""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from ats.core.config import settings
from ats.core.logger import get_logger
from ats.matching.base import CandidateMatch, MatchingStrategy
from ats.matching.semantic import SemanticMatcher
from ats.matching.tfidf import TfidfMatcher

log = get_logger(__name__)

# Smoothing constant. Cormack et al. (2009) found this robust; we keep it
# fixed since this stage isn't tuning the fusion.
DEFAULT_K = 60
DEFAULT_OVERFETCH = 3  # fan-out multiplier: each matcher returns top_k * 3


class RrfMatcher(MatchingStrategy):
    name = "rrf"

    def __init__(
        self,
        matchers: list[MatchingStrategy] | None = None,
        k: int = DEFAULT_K,
        over_fetch: int = DEFAULT_OVERFETCH,
    ) -> None:
        self.matchers = matchers or [SemanticMatcher(), TfidfMatcher()]
        self.k = k
        self.over_fetch = over_fetch

    async def match_by_vacancy(
        self,
        session: AsyncSession,
        job_id: int,
        top_k: int | None = None,
    ) -> list[CandidateMatch]:
        top_k = top_k or settings.matching.top_k
        fan = max(top_k * self.over_fetch, top_k)
        log.info(
            "rrf_match_by_vacancy",
            job_id=job_id,
            top_k=top_k,
            fan_out=fan,
            sources=[m.name for m in self.matchers],
        )
        # Serial execution: a single AsyncSession is NOT safe for concurrent
        # use across two awaiters (raises IllegalStateChangeError). Each
        # matcher's individual query is sub-100ms at our scale, so parallel
        # gain is negligible — serial keeps the session contract clean.
        rankings: list[list[CandidateMatch]] = []
        for m in self.matchers:
            rankings.append(await m.match_by_vacancy(session, job_id, top_k=fan))
        return self._fuse(rankings, top_k)

    async def match_by_text(
        self,
        session: AsyncSession,
        vacancy_text: str,
        top_k: int | None = None,
        vacancy_title: str | None = None,
    ) -> list[CandidateMatch]:
        top_k = top_k or settings.matching.top_k
        fan = max(top_k * self.over_fetch, top_k)
        log.info(
            "rrf_match_by_text",
            chars=len(vacancy_text),
            top_k=top_k,
            fan_out=fan,
            sources=[m.name for m in self.matchers],
        )
        rankings: list[list[CandidateMatch]] = []
        for m in self.matchers:
            rankings.append(await m.match_by_text(
                session, vacancy_text, top_k=fan, vacancy_title=vacancy_title,
            ))
        return self._fuse(rankings, top_k)

    def _fuse(
        self,
        rankings: list[list[CandidateMatch]],
        top_k: int,
    ) -> list[CandidateMatch]:
        scores: dict[int, float] = defaultdict(float)
        by_id: dict[int, CandidateMatch] = {}
        for ranking in rankings:
            for rank, m in enumerate(ranking, start=1):
                scores[m.candidate_id] += 1.0 / (self.k + rank)
                # Keep the first match seen — name/email/source_file are
                # stable across matchers; only the score+strategy differ.
                by_id.setdefault(m.candidate_id, m)
        ordered = sorted(scores, key=lambda cid: -scores[cid])[:top_k]
        results = [
            by_id[cid].model_copy(update={
                "score": scores[cid],
                "strategy": self.name,
            })
            for cid in ordered
        ]
        log.info(
            "rrf_match_done",
            results=len(results),
            top_score=results[0].score if results else None,
        )
        return results
